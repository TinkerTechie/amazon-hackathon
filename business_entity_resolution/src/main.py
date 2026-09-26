"""
main.py — End-to-end business entity resolution pipeline.

Usage:
    python src/main.py [options]

Options:
    --train-only        Run only train pipeline (skip test inference)
    --test-only         Run only test inference (requires saved model)
    --n-train-sample N  Number of S1 entities to sample for training (default: 200000)
    --top-k N           Candidates per blocker per S1 (default: 50)
    --n-folds N         GroupKFold folds (default: 5)
    --output-dir PATH   Output directory (default: output/)
    --data-dir PATH     Root data directory (default: dataset/)
"""
import argparse
import gc
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_dataframe
from blocking import MultiBlocker, evaluate_candidate_recall, validate_blocking
from features import (
    build_feature_dataframe,
    get_feature_columns,
    TfidfCosineScorer,
    BLOCK_FEATURE_NAMES,
)
from model import (
    train_with_oof,
    train_final_model,
    report_feature_importance,
    build_hard_negatives,
    DEFAULT_LGB_PARAMS,
)
from evaluation import (
    optimize_threshold,
    evaluate_predictions,
    compute_macro_f05,
    diagnose_oof_errors,
)
from inference import write_submission, run_inference_on_test

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─── Config ───────────────────────────────────────────────────────────────────
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"
MODEL_DIR = "model_artifacts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--n-train-sample", type=int, default=200_000,
                   help="Number of S1 train entities to sample (0 = all)")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--cand-chunk-size", type=int, default=250_000,
                   help="Candidate reference chunk size for blocking (default: 250000)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--data-dir", type=str, default="dataset")
    p.add_argument("--no-tfidf-features", action="store_true",
                   help="Skip TF-IDF cosine features (faster, slightly lower quality)")
    p.add_argument("--validate", action="store_true",
                   help="Run blocking validation on 5000 S1 records before full pipeline run")
    p.add_argument("--validate-n", type=int, default=5_000,
                   help="Number of S1 records to use for validation (default: 5000)")
    return p.parse_args()


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_and_preprocess_train(data_dir: str) -> Dict:
    """Load and preprocess only training data."""
    train_dir = os.path.join(data_dir, "train")

    logger.info("Loading training data...")
    t0 = time.time()
    s1_train = pd.read_csv(f"{train_dir}/train_source1.tsv", sep="\t")
    s2_train = pd.read_csv(f"{train_dir}/train_source2.tsv", sep="\t")
    s3_train = pd.read_csv(f"{train_dir}/train_source3.tsv", sep="\t")
    gt_train = pd.read_csv(f"{train_dir}/train_ground_truth.tsv", sep="\t")
    logger.info(
        f"Loaded train: S1={len(s1_train)}, S2={len(s2_train)}, S3={len(s3_train)}, "
        f"GT={len(gt_train)} ({time.time()-t0:.1f}s)"
    )

    logger.info("Preprocessing training data...")
    s1_train = preprocess_dataframe(s1_train)
    s2_train = preprocess_dataframe(s2_train)
    s3_train = preprocess_dataframe(s3_train)

    # Parse ground truth
    gt_train["matched_entity_ids"] = gt_train["matched_entity_ids"].fillna("")
    gt_dict = {}
    for _, row in gt_train.iterrows():
        sid = row["source1_entity_id"]
        m = str(row["matched_entity_ids"]).strip()
        gt_dict[sid] = set(x.strip() for x in m.split(",") if x.strip())

    return {
        "s1_train": s1_train,
        "s2_train": s2_train,
        "s3_train": s3_train,
        "gt_dict": gt_dict,
        "gt_train": gt_train,
    }


def load_and_preprocess_test(data_dir: str) -> Dict:
    """Load and preprocess only test data."""
    test_dir = os.path.join(data_dir, "test")

    logger.info("Loading test data...")
    t0 = time.time()
    s1_test = pd.read_csv(f"{test_dir}/test_source1.tsv", sep="\t")
    s2_test = pd.read_csv(f"{test_dir}/test_source2.tsv", sep="\t")
    s3_test = pd.read_csv(f"{test_dir}/test_source3.tsv", sep="\t")
    logger.info(
        f"Loaded test: S1={len(s1_test)}, S2={len(s2_test)}, S3={len(s3_test)} "
        f"({time.time()-t0:.1f}s)"
    )

    logger.info("Preprocessing test data...")
    s1_test = preprocess_dataframe(s1_test)
    s2_test = preprocess_dataframe(s2_test)
    s3_test = preprocess_dataframe(s3_test)

    return {
        "s1_test": s1_test,
        "s2_test": s2_test,
        "s3_test": s3_test,
    }


def load_and_preprocess(data_dir: str) -> Dict:
    """Backward-compatible loader that loads both train and test data."""
    train_data = load_and_preprocess_train(data_dir)
    test_data = load_and_preprocess_test(data_dir)
    return {**train_data, **test_data}


# ─── Training pipeline ────────────────────────────────────────────────────────

def build_training_pairs(
    s1_sample: pd.DataFrame,
    cand_df: pd.DataFrame,
    gt_dict: Dict[str, set],
    block_info: Dict,
) -> List:
    """
    Build labeled training pairs from candidate set.
    Label=1 if pair is in ground truth, 0 otherwise.

    IMPORTANT: The blocker recall reported BEFORE calling this function is the
    honest recall: the fraction of GT positives that the blocker itself found.
    GT positives are added below ONLY to improve model training coverage; they
    must not be counted in the blocker recall figure.
    """
    pair_list = []
    blocker_missed = 0
    for sid, r in block_info.items():
        true_matches = gt_dict.get(sid, set())
        for cand_id in r["all_candidates"]:
            label = int(cand_id in true_matches)
            pair_list.append((sid, cand_id, label))

        # Add positive pairs that blocking missed — improves training only;
        # has NO effect on test-time recall (blocker is the bottleneck there).
        for true_cand in true_matches:
            if true_cand not in r["all_candidates"]:
                pair_list.append((sid, true_cand, 1))
                blocker_missed += 1

    n_pos = sum(1 for _, _, l in pair_list if l == 1)
    n_neg = sum(1 for _, _, l in pair_list if l == 0)
    logger.info(
        f"Training pairs: {len(pair_list)} total, {n_pos} pos, {n_neg} neg | "
        f"GT positives injected (missed by blocker): {blocker_missed}"
    )
    return pair_list


def run_training(
    data: Dict,
    n_train_sample: int = 200_000,
    top_k: int = 50,
    cand_chunk_size: int = 250_000,
    n_folds: int = 5,
    use_tfidf_features: bool = True,
    output_dir: str = OUTPUT_DIR,
    run_validate: bool = False,
    validate_n: int = 5_000,
) -> Dict:
    """
    Full training pipeline:
    1. Sample S1 entities for tractable training
    2. Generate candidates via multi-blocking (chunked reference processing)
    3. Evaluate blocking recall
    4. Build training pairs + features
    5. Train LightGBM with GroupKFold
    6. Optimize threshold on OOF predictions
    7. Return everything needed for test inference
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    s1_train = data["s1_train"]
    s2_train = data["s2_train"]
    s3_train = data["s3_train"]
    gt_dict = data["gt_dict"]

    # ── Sample S1 entities ─────────────────────────────────────────────────
    all_s1_ids_train = s1_train["entity_id"].tolist()
    if n_train_sample > 0 and n_train_sample < len(s1_train):
        # Stratified: include some singletons
        singleton_ids = [sid for sid in all_s1_ids_train if not gt_dict.get(sid)]
        non_singleton_ids = [sid for sid in all_s1_ids_train if gt_dict.get(sid)]

        n_singletons = min(int(n_train_sample * 0.05), len(singleton_ids))
        n_non_singletons = min(n_train_sample - n_singletons, len(non_singleton_ids))

        rng = np.random.RandomState(42)
        sampled_singleton_ids = list(rng.choice(singleton_ids, n_singletons, replace=False))
        sampled_non_singleton_ids = list(rng.choice(non_singleton_ids, n_non_singletons, replace=False))
        sampled_s1_ids = set(sampled_singleton_ids + sampled_non_singleton_ids)

        s1_sample = s1_train[s1_train["entity_id"].isin(sampled_s1_ids)].copy()
        logger.info(
            f"Sampled {len(s1_sample)} S1 entities "
            f"({n_singletons} singletons, {n_non_singletons} with matches)"
        )
    else:
        s1_sample = s1_train.copy()
        logger.info(f"Using all {len(s1_sample)} S1 training entities")

    # ── Combine S2 + S3 candidates ─────────────────────────────────────────
    cand_train = pd.concat([s2_train, s3_train], ignore_index=True)
    logger.info(f"Total candidates (S2+S3): {len(cand_train)}")
    # Release s2_train and s3_train now that they are combined
    del s2_train, s3_train
    data.pop("s2_train", None)
    data.pop("s3_train", None)
    gc.collect()

    # ── Multi-blocking (with chunked candidate processing) ─────────────────
    logger.info(f"Running multi-blocking with top_k={top_k}, cand_chunk_size={cand_chunk_size}...")
    blocker = MultiBlocker(top_k=top_k, max_candidates_per_s1=300, cand_chunk_size=cand_chunk_size)

    # ── Optional validation run before full blocking ───────────────────────
    if run_validate:
        logger.info("Running blocking validation before full pipeline...")
        validate_blocking(
            s1_train,
            cand_train,
            data["gt_train"],
            blocker,
            sample_size=validate_n,
        )

    block_info = blocker.generate_candidates(s1_sample, cand_train)

    # ── Evaluate blocking recall ───────────────────────────────────────────
    # IMPORTANT: Measure recall NOW, before build_training_pairs injects
    # missed GT positives.  The number below is the true blocker recall.
    logger.info("Evaluating blocking recall (before GT injection)...")
    gt_sample = data["gt_train"][
        data["gt_train"]["source1_entity_id"].isin(s1_sample["entity_id"])
    ].copy()
    recall_stats = evaluate_candidate_recall(block_info, gt_sample, label="Training blocking")
    logger.info(
        f"Blocker recall (unadjusted): {recall_stats['candidate_recall']:.4f} | "
        f"avg candidates: {recall_stats['avg_candidates_per_s1']:.1f}"
    )

    # ── Build training pairs ────────────────────────────────────────────────
    logger.info("Building training pairs...")
    pair_list = build_training_pairs(s1_sample, cand_train, gt_dict, block_info)

    if not pair_list:
        raise RuntimeError("No training pairs generated!")

    # ── Fit TF-IDF scorer ─────────────────────────────────────────────────
    tfidf_scorer = None
    if use_tfidf_features:
        logger.info("Fitting TF-IDF scorer on training candidates...")
        tfidf_scorer = TfidfCosineScorer()
        tfidf_scorer.fit(cand_train)

    # ── Compute features ───────────────────────────────────────────────────
    s1_sample_ids = s1_sample["entity_id"].tolist()

    # Pre-extract S1 TF-IDF texts for fold-isolated TF-IDF computation
    s1_tfidf_texts = None
    if use_tfidf_features:
        tfidf_text_keys = {
            "name_char": "name_clean",
            "name_word": "name_clean",
            "addr_char": "addr_expanded",
            "addr_word": "addr_expanded",
            "fulltext": "combined_text",
        }
        s1_tfidf_texts = {
            row["entity_id"]: {k: str(row.get(col, "") or "") for k, col in tfidf_text_keys.items()}
            for _, row in s1_sample.iterrows()
        }

    # ── Compute features ───────────────────────────────────────────────────
    logger.info("Computing training features...")
    feat_df = build_feature_dataframe(
        pair_list, s1_sample, cand_train, block_info, tfidf_scorer,
        chunk_size=100_000,
    )
    # Release pair_list
    del pair_list
    gc.collect()

    if feat_df.empty:
        raise RuntimeError("Empty feature dataframe!")

    feature_cols = get_feature_columns(feat_df)
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")

    # ── Hard negative sampling ─────────────────────────────────────────────
    logger.info("Applying hard-negative sampling...")
    feat_df = build_hard_negatives(feat_df, gt_dict, feature_cols, neg_to_pos_ratio=5.0)
    logger.info(f"After hard-negative sampling: {len(feat_df)} pairs")

    # ── GroupKFold training (TF-IDF vectorizers fitted strictly per fold) ──
    logger.info(f"Training LightGBM with {n_folds}-fold GroupKFold (group_col=s1_id)...")
    oof_probs, fold_models, best_thresh = train_with_oof(
        feat_df,
        feature_cols,
        n_folds=n_folds,
        gt_dict=gt_dict,
        all_s1_ids=s1_sample_ids,
        cand_df=cand_train,
        s1_tfidf_texts=s1_tfidf_texts,
    )

    # Release cand_train, s1_sample, and s1_tfidf_texts now that OOF is done
    del cand_train, s1_sample, s1_tfidf_texts
    gc.collect()

    # ── OOF evaluation ─────────────────────────────────────────────────────
    oof_df = feat_df[["s1_id", "cand_id"]].copy()
    oof_df["prob"] = oof_probs

    oof_matches = {}
    for sid in s1_sample_ids:
        oof_matches[sid] = []

    for sid, grp in oof_df.groupby("s1_id"):
        above = grp[grp["prob"] >= best_thresh]["cand_id"].tolist()
        oof_matches[sid] = above

    oof_eval = evaluate_predictions(
        oof_matches, gt_dict, s1_sample_ids,
        label="OOF validation"
    )
    logger.info(f"OOF F0.5: {oof_eval['macro_f05']:.5f}")

    # ── False-negative & singleton error diagnostics ──────────────────────
    diagnostics = diagnose_oof_errors(
        block_info, oof_matches, gt_dict, s1_sample_ids,
        label="OOF validation"
    )
    del block_info
    gc.collect()

    # ── Feature importance ─────────────────────────────────────────────────
    importance_df = report_feature_importance(fold_models, feature_cols)
    importance_df.to_csv(os.path.join(output_dir, "feature_importance.tsv"), sep="\t", index=False)

    # ── Train final model on all data ──────────────────────────────────────
    logger.info("Training final model on all training data...")
    avg_best_iter = int(np.mean([m.best_iteration_ for m in fold_models if hasattr(m, 'best_iteration_')]) * 1.1)
    avg_best_iter = max(avg_best_iter, 100)
    logger.info(f"Using {avg_best_iter} estimators for final model")

    final_model = train_final_model(
        feat_df, feature_cols,
        n_estimators=avg_best_iter,
    )

    # ── Save artifacts ─────────────────────────────────────────────────────
    artifacts = {
        "final_model": final_model,
        "fold_models": fold_models,
        "feature_cols": feature_cols,
        "best_threshold": best_thresh,
        "tfidf_scorer": tfidf_scorer,
        "blocker": blocker,
        "oof_f05": oof_eval["macro_f05"],
        "blocking_recall": recall_stats["candidate_recall"],
        "oof_diagnostics": diagnostics,
    }


    artifact_path = os.path.join(MODEL_DIR, "pipeline_artifacts.pkl")
    logger.info(f"Saving artifacts to {artifact_path}...")
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f, protocol=4)

    logger.info(
        f"\n{'='*60}\n"
        f"TRAINING COMPLETE\n"
        f"  Blocking recall: {recall_stats['candidate_recall']:.4f}\n"
        f"  OOF F0.5:       {oof_eval['macro_f05']:.5f}\n"
        f"  Best threshold: {best_thresh:.2f}\n"
        f"{'='*60}"
    )

    # Clean up training data structures
    del feat_df, oof_df, oof_probs
    gc.collect()

    return artifacts


# ─── Test inference pipeline ──────────────────────────────────────────────────

def run_test_inference(
    data: Dict,
    artifacts: Dict,
    output_dir: str = OUTPUT_DIR,
    batch_size: int = 50_000,
):
    """Apply trained model to the full test set."""
    logger.info("Starting test inference...")

    s1_test = data["s1_test"]
    s2_test = data["s2_test"]
    s3_test = data["s3_test"]
    all_s1_ids_test = s1_test["entity_id"].tolist()

    blocker = artifacts["blocker"]
    final_model = artifacts["final_model"]
    fold_models = artifacts.get("fold_models", [final_model])
    feature_cols = artifacts["feature_cols"]
    threshold = artifacts["best_threshold"]
    tfidf_scorer = artifacts.get("tfidf_scorer")

    cand_test = pd.concat([s2_test, s3_test], ignore_index=True)
    # Release s2_test and s3_test to save memory
    del s2_test, s3_test
    data.pop("s2_test", None)
    data.pop("s3_test", None)
    gc.collect()

    # ── Generate test candidates ───────────────────────────────────────────
    logger.info(f"Generating test candidates for {len(s1_test)} S1 entities...")
    test_block_info = blocker.generate_candidates(s1_test, cand_test)

    # ── Build test pair list ───────────────────────────────────────────────
    pair_list = []
    for sid, r in test_block_info.items():
        for cand_id in r["all_candidates"]:
            pair_list.append((sid, cand_id, -1))

    logger.info(f"Test candidate pairs: {len(pair_list)}")

    if not pair_list:
        logger.warning("No test candidates generated!")
        write_submission({}, {}, all_s1_ids_test, output_dir)
        return

    # ── Index TF-IDF candidates for test (transform only, no re-fitting) ──
    if tfidf_scorer is not None:
        logger.info("Indexing test candidates with pre-fitted training vectorizers (transform only)...")
        tfidf_scorer.index_candidates(cand_test)

    # ── Compute features ───────────────────────────────────────────────────
    logger.info("Computing test features...")
    feat_df = build_feature_dataframe(
        pair_list, s1_test, cand_test, test_block_info, tfidf_scorer,
        chunk_size=batch_size,
    )
    # Release pair_list and cand_test
    del pair_list, cand_test
    gc.collect()

    if feat_df.empty:
        logger.warning("Empty test feature dataframe!")
        write_submission({}, {}, all_s1_ids_test, output_dir)
        return

    # Align feature columns (handle any missing features)
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df = feat_df[["s1_id", "cand_id"] + feature_cols]

    # ── Predict ────────────────────────────────────────────────────────────
    logger.info("Predicting probabilities...")
    X_test = feat_df[feature_cols].values
    if fold_models:
        probs = np.stack([m.predict_proba(X_test)[:, 1] for m in fold_models])
        test_probs = probs.mean(axis=0)
    else:
        test_probs = final_model.predict_proba(X_test)[:, 1]

    del X_test
    gc.collect()

    feat_df["prob"] = test_probs

    # ── Apply threshold ────────────────────────────────────────────────────
    logger.info(f"Applying threshold {threshold:.3f}...")
    matches_dict = {sid: [] for sid in all_s1_ids_test}
    for sid, grp in feat_df.groupby("s1_id"):
        above = grp[grp["prob"] >= threshold]["cand_id"].tolist()
        matches_dict[sid] = above

    # Build candidates dict
    candidates_dict = {}
    for sid in all_s1_ids_test:
        r = test_block_info.get(sid, {})
        candidates_dict[sid] = r.get("all_candidates", set())

    # ── Write submission ───────────────────────────────────────────────────
    write_submission(matches_dict, candidates_dict, all_s1_ids_test, output_dir)
    logger.info("Test inference complete!")

    del feat_df, matches_dict, candidates_dict
    gc.collect()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info("Business Entity Resolution Pipeline")
    logger.info(f"Config: n_train_sample={args.n_train_sample}, top_k={args.top_k}, "
                f"cand_chunk_size={args.cand_chunk_size}, "
                f"n_folds={args.n_folds}, output_dir={args.output_dir}")

    output_dir = args.output_dir
    data_dir = args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    use_tfidf = not args.no_tfidf_features

    if args.test_only:
        # Load saved artifacts
        artifact_path = os.path.join(MODEL_DIR, "pipeline_artifacts.pkl")
        logger.info(f"Loading artifacts from {artifact_path}...")
        with open(artifact_path, "rb") as f:
            artifacts = pickle.load(f)
        # Load and preprocess ONLY test data
        test_data = load_and_preprocess_test(data_dir)
        run_test_inference(test_data, artifacts, output_dir)
        del test_data
        gc.collect()
        return

    if args.train_only:
        # Load and preprocess ONLY train data
        train_data = load_and_preprocess_train(data_dir)
        run_training(
            train_data,
            n_train_sample=args.n_train_sample,
            top_k=args.top_k,
            cand_chunk_size=args.cand_chunk_size,
            n_folds=args.n_folds,
            use_tfidf_features=use_tfidf,
            output_dir=output_dir,
            run_validate=args.validate,
            validate_n=args.validate_n,
        )
        del train_data
        gc.collect()
        return

    # Full pipeline: train then test sequentially (NEVER load both simultaneously)
    logger.info("── Stage 1/2: Training Pipeline ──────────────────────────────")
    train_data = load_and_preprocess_train(data_dir)
    artifacts = run_training(
        train_data,
        n_train_sample=args.n_train_sample,
        top_k=args.top_k,
        cand_chunk_size=args.cand_chunk_size,
        n_folds=args.n_folds,
        use_tfidf_features=use_tfidf,
        output_dir=output_dir,
        run_validate=args.validate,
        validate_n=args.validate_n,
    )
    # Explicitly release all training DataFrames before loading test data
    del train_data
    gc.collect()
    logger.info("Training complete. Released all training DataFrames from memory.")

    logger.info("── Stage 2/2: Test Inference Pipeline ────────────────────────")
    test_data = load_and_preprocess_test(data_dir)
    run_test_inference(test_data, artifacts, output_dir)
    del test_data
    gc.collect()
    logger.info("Test inference complete. Released all test DataFrames from memory.")

    # ── Validate submission ────────────────────────────────────────────────
    logger.info("\nRunning submission validator...")
    os.system(
        f"python3 utils/validate_submission.py "
        f"--matching {output_dir}/matching_results.tsv "
        f"--candidate {output_dir}/candidate_pairs.tsv "
        f"--test-dir {data_dir}/test"
    )


if __name__ == "__main__":
    main()

