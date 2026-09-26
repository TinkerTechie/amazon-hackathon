"""
inference.py — Final test inference pipeline.

Applies the trained model to the test set and generates:
  - output/candidate_pairs.tsv
  - output/matching_results.tsv
"""
import logging
import os
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from features import build_feature_dataframe, get_feature_columns, TfidfCosineScorer

logger = logging.getLogger(__name__)


def write_submission(
    matches: Dict[str, List[str]],
    candidates: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_dir: str = "output",
):
    """
    Write matching_results.tsv and candidate_pairs.tsv.
    Enforces all format rules:
      - every S1 has exactly one row
      - empty matched_entity_ids for singletons
      - no duplicate IDs
      - no S1- IDs in matched lists
    """
    os.makedirs(output_dir, exist_ok=True)

    # Validate and write matching_results.tsv
    match_rows = []
    for sid in all_s1_ids:
        matched = matches.get(sid, [])
        # Deduplicate, keep order
        seen = set()
        deduped = []
        for mid in matched:
            if mid not in seen:
                seen.add(mid)
                deduped.append(mid)
        # Remove any S1 IDs (safety check)
        deduped = [mid for mid in deduped if not mid.startswith("S1-")]
        match_rows.append({
            "source1_entity_id": sid,
            "matched_entity_ids": ",".join(deduped),
        })

    match_df = pd.DataFrame(match_rows)
    match_path = os.path.join(output_dir, "matching_results.tsv")
    match_df.to_csv(match_path, sep="\t", index=False)
    logger.info(f"Wrote {len(match_df)} rows to {match_path}")

    # Validate and write candidate_pairs.tsv
    cand_rows = []
    for sid in all_s1_ids:
        cand_set = candidates.get(sid, set())
        # Ensure matched IDs are a subset of candidates
        extra_matched = set(matches.get(sid, [])) - cand_set
        if extra_matched:
            cand_set.update(extra_matched)
            logger.warning(
                f"  Added {len(extra_matched)} matched IDs to candidates for {sid}"
            )
        cand_list = sorted(cand_set)
        cand_rows.append({
            "source1_entity_id": sid,
            "candidate_entity_ids": ",".join(cand_list),
        })

    cand_df = pd.DataFrame(cand_rows)
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    cand_df.to_csv(cand_path, sep="\t", index=False)
    logger.info(f"Wrote {len(cand_df)} rows to {cand_path}")

    # Summary stats
    n_with_matches = sum(1 for sid in all_s1_ids if matches.get(sid))
    n_singletons = len(all_s1_ids) - n_with_matches
    total_matched = sum(len(v) for v in matches.values())
    logger.info(
        f"Submission summary: {n_with_matches} S1 with matches, "
        f"{n_singletons} singletons, {total_matched} total matched IDs"
    )


def run_inference_on_test(
    s1_test: pd.DataFrame,
    s2_test: pd.DataFrame,
    s3_test: pd.DataFrame,
    blocker,
    trained_models,
    feature_cols: List[str],
    threshold: float,
    tfidf_scorer: Optional[TfidfCosineScorer],
    block_info: Optional[Dict] = None,
    output_dir: str = "output",
    batch_size: int = 50_000,
):
    """
    Full inference pipeline for test set.
    Processes S1 entities in batches to avoid OOM.
    """
    all_s1_ids = s1_test["entity_id"].tolist()
    cand_test = pd.concat([s2_test, s3_test], ignore_index=True)

    logger.info(f"Running inference for {len(all_s1_ids)} test S1 entities...")

    # ── Generate candidates ────────────────────────────────────────────────
    if block_info is None:
        logger.info("Generating test candidates...")
        block_info = blocker.generate_candidates(s1_test, cand_test)

    # ── Build pair list ────────────────────────────────────────────────────
    pair_list = []
    for sid, r in block_info.items():
        for cand_id in r["all_candidates"]:
            pair_list.append((sid, cand_id, -1))  # label=-1 means unknown

    logger.info(f"Total test candidate pairs: {len(pair_list)}")

    if not pair_list:
        logger.warning("No candidates generated for test set!")
        write_submission({}, {}, all_s1_ids, output_dir)
        return

    # ── Index TF-IDF candidates for test (transform only, no re-fitting) ──
    if tfidf_scorer is not None:
        tfidf_scorer.index_candidates(cand_test)

    # ── Compute features in chunks ─────────────────────────────────────────
    logger.info("Computing test features...")
    feat_df = build_feature_dataframe(
        pair_list, s1_test, cand_test, block_info, tfidf_scorer,
        chunk_size=batch_size,
    )

    if feat_df.empty:
        logger.warning("Empty feature dataframe for test set!")
        write_submission({}, {}, all_s1_ids, output_dir)
        return

    # ── Predict probabilities ──────────────────────────────────────────────
    logger.info("Predicting probabilities...")
    X_test = feat_df[feature_cols].values

    if isinstance(trained_models, list):
        probs = np.stack([m.predict_proba(X_test)[:, 1] for m in trained_models])
        test_probs = probs.mean(axis=0)
    else:
        test_probs = trained_models.predict_proba(X_test)[:, 1]

    feat_df["prob"] = test_probs

    # ── Apply threshold ────────────────────────────────────────────────────
    logger.info(f"Applying threshold {threshold:.3f}...")
    matches_dict = {}
    for sid in all_s1_ids:
        matches_dict[sid] = []

    for sid, grp in feat_df.groupby("s1_id"):
        above = grp[grp["prob"] >= threshold]["cand_id"].tolist()
        matches_dict[sid] = above

    # Build candidates dict for output
    candidates_dict = {}
    for sid, r in block_info.items():
        candidates_dict[sid] = r["all_candidates"]
    for sid in all_s1_ids:
        if sid not in candidates_dict:
            candidates_dict[sid] = set()

    # ── Write submission ───────────────────────────────────────────────────
    write_submission(matches_dict, candidates_dict, all_s1_ids, output_dir)
