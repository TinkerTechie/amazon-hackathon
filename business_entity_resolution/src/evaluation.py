"""
evaluation.py — F0.5 evaluation utilities.

Implements:
  - Per-entity F0.5 computation
  - Macro F0.5 over all S1 entities
  - Threshold sweep and optimization
  - OOF prediction aggregation
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BETA = 0.5
BETA2 = BETA ** 2


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def entity_f05(true_matches: set, predicted_matches: set) -> float:
    """
    Compute F0.5 for a single S1 entity.
    Both arguments are sets of matched entity IDs.
    """
    if not true_matches and not predicted_matches:
        return 1.0  # correctly predicted singleton
    if not true_matches and predicted_matches:
        return 0.0  # false merge on singleton
    if not predicted_matches:
        recall = 0.0
        precision = 0.0
    else:
        tp = len(true_matches & predicted_matches)
        precision = tp / len(predicted_matches)
        recall = tp / len(true_matches)
    return f_beta(precision, recall)


def compute_macro_f05(
    gt_dict: Dict[str, set],
    pred_dict: Dict[str, set],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute macro F0.5 over all S1 entities in gt_dict.
    
    Returns:
      (macro_f05, per_entity_scores)
    """
    per_entity = {}
    for sid, true_m in gt_dict.items():
        pred_m = pred_dict.get(sid, set())
        per_entity[sid] = entity_f05(true_m, pred_m)

    macro = np.mean(list(per_entity.values())) if per_entity else 0.0
    return float(macro), per_entity


def optimize_threshold(
    oof_df: pd.DataFrame,
    gt_dict: Dict[str, set],
    all_s1_ids: List[str],
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Sweep thresholds and return the best (threshold, macro_F0.5).
    
    Args:
      oof_df: DataFrame with columns [s1_id, cand_id, prob]
      gt_dict: {s1_id: set of true matched IDs}
      all_s1_ids: Complete list of S1 IDs (includes singletons with no candidates)
      thresholds: thresholds to try; defaults to 0.30-0.99 with step 0.01
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 1.00, 0.01)]

    # Group predictions by S1
    grouped = oof_df.groupby("s1_id")

    best_thresh = 0.5
    best_score = -1.0
    results = []

    # Build a full gt_dict for all s1 (including singletons)
    full_gt = {sid: gt_dict.get(sid, set()) for sid in all_s1_ids}

    for thresh in thresholds:
        pred_dict = {}
        for sid in all_s1_ids:
            pred_dict[sid] = set()

        for sid, grp in grouped:
            matches = set(grp.loc[grp["prob"] >= thresh, "cand_id"].tolist())
            pred_dict[sid] = matches

        score, _ = compute_macro_f05(full_gt, pred_dict)
        results.append((thresh, score))

        if score > best_score:
            best_score = score
            best_thresh = thresh

    logger.info(
        f"Threshold sweep: best threshold={best_thresh:.2f}, "
        f"macro F0.5={best_score:.5f}"
    )

    # Log top-5 thresholds
    top5 = sorted(results, key=lambda x: -x[1])[:5]
    logger.info(f"Top-5 thresholds: {top5}")

    return best_thresh, best_score


def apply_threshold_with_margin(
    pred_df: pd.DataFrame,
    threshold: float,
    margin_threshold: Optional[float] = None,
) -> Dict[str, List[str]]:
    """
    Apply threshold to produce final matches.
    
    Optional margin-based logic:
      If the top candidate has high probability but the gap to the second 
      candidate is small (ambiguous), we can apply a higher secondary threshold.
    
    Returns: {s1_id: [matched_cand_id, ...]}
    """
    matches = {}

    for sid, grp in pred_df.groupby("s1_id"):
        grp = grp.sort_values("prob", ascending=False).reset_index(drop=True)
        probs = grp["prob"].values

        matched = []
        for i, (_, row) in enumerate(grp.iterrows()):
            prob = row["prob"]
            if prob < threshold:
                break  # sorted descending; no more candidates above threshold

            if margin_threshold is not None and i > 0:
                # Check if probability margin to top is still significant
                top_prob = probs[0]
                if (top_prob - prob) > margin_threshold and prob < threshold + 0.05:
                    # Ambiguous secondary candidate — skip
                    continue

            matched.append(row["cand_id"])

        matches[sid] = matched

    return matches


def evaluate_predictions(
    matches: Dict[str, List[str]],
    gt_dict: Dict[str, set],
    all_s1_ids: List[str],
    label: str = "",
) -> Dict:
    """Evaluate final match predictions against ground truth."""
    pred_dict = {sid: set(v) for sid, v in matches.items()}
    full_gt = {sid: gt_dict.get(sid, set()) for sid in all_s1_ids}

    macro_f05, per_entity = compute_macro_f05(full_gt, pred_dict)

    # Compute precision/recall overall
    tp_total = fp_total = fn_total = 0
    singleton_correct = singleton_incorrect = 0

    for sid in all_s1_ids:
        true_m = full_gt[sid]
        pred_m = pred_dict.get(sid, set())

        if not true_m:
            # Singleton entity
            if not pred_m:
                singleton_correct += 1
            else:
                singleton_incorrect += 1
        else:
            tp = len(true_m & pred_m)
            fp = len(pred_m - true_m)
            fn = len(true_m - pred_m)
            tp_total += tp
            fp_total += fp
            fn_total += fn

    overall_prec = tp_total / max(1, tp_total + fp_total)
    overall_recall = tp_total / max(1, tp_total + fn_total)

    result = {
        "label": label,
        "macro_f05": macro_f05,
        "overall_precision": overall_prec,
        "overall_recall": overall_recall,
        "singleton_correct": singleton_correct,
        "singleton_incorrect": singleton_incorrect,
        "singleton_accuracy": singleton_correct / max(1, singleton_correct + singleton_incorrect),
    }

    logger.info(
        f"[Eval] {label}: macro_F0.5={macro_f05:.5f} | "
        f"P={overall_prec:.4f} R={overall_recall:.4f} | "
        f"Singletons: {singleton_correct} correct, {singleton_incorrect} incorrect"
    )
    return result


def diagnose_oof_errors(
    block_info: Dict[str, Dict],
    oof_matches: Dict[str, List[str]],
    gt_dict: Dict[str, set],
    all_s1_ids: List[str],
    label: str = "OOF Diagnostics",
) -> Dict:
    """
    Explicit False-Negative and Singleton Error Diagnostics.
    
    Partitions errors into three disjoint, actionable categories:
      (1) True matches missed by blocking (blocker recall bottleneck)
      (2) True matches retrieved as candidates but rejected by matcher (classifier/threshold bottleneck)
      (3) False positives on singleton S1 records (precision bottleneck / false merges)

    Args:
      block_info: {s1_id: {"all_candidates": set_of_cand_ids, ...}}
      oof_matches: {s1_id: [predicted_cand_ids, ...]}
      gt_dict: {s1_id: set_of_true_cand_ids}
      all_s1_ids: List of S1 IDs evaluated
      label: Optional logging label
    """
    total_true_matches = 0
    fn_blocking = 0         # True match not in candidate pool
    fn_matcher = 0          # True match in candidate pool, but not predicted
    tp_matches = 0          # Correctly predicted true match

    singleton_count = 0
    fp_singletons_s1 = 0    # Singletons with at least one false positive match
    fp_singletons_edges = 0 # Total false positive matches on singletons

    sample_fn_blocking = []
    sample_fn_matcher = []
    sample_fp_singletons = []

    for sid in all_s1_ids:
        true_m = gt_dict.get(sid, set())
        pred_m = set(oof_matches.get(sid, []))
        cands = block_info.get(sid, {}).get("all_candidates", set()) if block_info else set()

        if not true_m:
            singleton_count += 1
            if pred_m:
                fp_singletons_s1 += 1
                fp_singletons_edges += len(pred_m)
                if len(sample_fp_singletons) < 5:
                    sample_fp_singletons.append({
                        "s1_id": sid,
                        "false_matches": sorted(pred_m),
                    })
        else:
            total_true_matches += len(true_m)
            for cid in true_m:
                if cid not in cands:
                    fn_blocking += 1
                    if len(sample_fn_blocking) < 5:
                        sample_fn_blocking.append({"s1_id": sid, "true_cand_id": cid})
                elif cid not in pred_m:
                    fn_matcher += 1
                    if len(sample_fn_matcher) < 5:
                        sample_fn_matcher.append({"s1_id": sid, "true_cand_id": cid})
                else:
                    tp_matches += 1

    fn_total = fn_blocking + fn_matcher
    blocker_fn_rate = fn_blocking / max(1, total_true_matches)
    matcher_fn_rate = fn_matcher / max(1, total_true_matches)
    total_fn_rate = fn_total / max(1, total_true_matches)
    singleton_fp_rate = fp_singletons_s1 / max(1, singleton_count)

    diagnostics = {
        "label": label,
        "total_true_matches": total_true_matches,
        "true_positives": tp_matches,
        "total_false_negatives": fn_total,
        "fn_blocking_count": fn_blocking,
        "fn_blocking_rate": blocker_fn_rate,
        "fn_matcher_count": fn_matcher,
        "fn_matcher_rate": matcher_fn_rate,
        "total_singletons": singleton_count,
        "fp_singletons_s1_count": fp_singletons_s1,
        "fp_singletons_s1_rate": singleton_fp_rate,
        "fp_singletons_edges": fp_singletons_edges,
        "sample_fn_blocking": sample_fn_blocking,
        "sample_fn_matcher": sample_fn_matcher,
        "sample_fp_singletons": sample_fp_singletons,
    }

    logger.info(
        f"\n{'='*70}\n"
        f"EXPLICIT FALSE-NEGATIVE & SINGLETON ERROR DIAGNOSTICS ({label})\n"
        f"{'='*70}\n"
        f"1. TRUE MATCHES MISSED BY BLOCKING (Blocker Bottleneck):\n"
        f"   - Missed Count: {fn_blocking} / {total_true_matches} true matches ({blocker_fn_rate * 100:.2f}%)\n"
        f"   - Blocker Recall Ceiling: {(1.0 - blocker_fn_rate) * 100:.2f}%\n"
        f"\n"
        f"2. TRUE MATCHES REJECTED BY MATCHER (Classifier/Threshold Bottleneck):\n"
        f"   - Rejected Count: {fn_matcher} / {total_true_matches} true matches ({matcher_fn_rate * 100:.2f}%)\n"
        f"   - Candidates retrieved by blocker but scored below threshold\n"
        f"\n"
        f"   TOTAL FALSE NEGATIVES: {fn_total} ({total_fn_rate * 100:.2f}% of all true matches)\n"
        f"   - Blocker share of False Negatives: {fn_blocking / max(1, fn_total) * 100:.1f}%\n"
        f"   - Matcher share of False Negatives: {fn_matcher / max(1, fn_total) * 100:.1f}%\n"
        f"\n"
        f"3. FALSE POSITIVES ON SINGLETON S1 RECORDS:\n"
        f"   - Singletons with False Matches: {fp_singletons_s1} / {singleton_count} ({singleton_fp_rate * 100:.2f}%)\n"
        f"   - Total False Positive Edges on Singletons: {fp_singletons_edges}\n"
        f"   - Singleton Precision: {(1.0 - singleton_fp_rate) * 100:.2f}%\n"
        f"{'='*70}"
    )

    return diagnostics

