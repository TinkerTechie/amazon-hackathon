"""
model.py — LightGBM pairwise match classifier.

Features:
  - GroupKFold cross-validation (groups = S1 entity_id)
  - Hard-negative aware training
  - OOF prediction generation
  - Feature importance reporting
"""
import logging
import pickle
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from evaluation import (
    optimize_threshold,
    evaluate_predictions,
    apply_threshold_with_margin,
)

logger = logging.getLogger(__name__)


# ─── Default LightGBM hyperparameters ────────────────────────────────────────
DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def train_with_oof(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    group_col: str = "s1_id",
    n_folds: int = 5,
    lgb_params: Optional[Dict] = None,
    gt_dict: Optional[Dict] = None,
    all_s1_ids: Optional[List[str]] = None,
    cand_df: Optional[pd.DataFrame] = None,
    s1_tfidf_texts: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[np.ndarray, List[lgb.LGBMClassifier], float]:
    """
    Train LightGBM with GroupKFold cross-validation grouped strictly by source1_entity_id.
    
    Data-leakage prevention:
    - GroupKFold splits strictly on group_col ('s1_id'), ensuring all candidate pairs for any S1
      belong entirely to either train or validation fold.
    - When cand_df and s1_tfidf_texts are provided, TF-IDF vectorizers are fitted strictly
      on each fold's training candidates only. Validation candidates are transformed using the
      training fold's vectorizer (never fitted).
    
    Returns:
      (oof_probs, list of fold models, best_threshold)
    """
    if lgb_params is None:
        lgb_params = DEFAULT_LGB_PARAMS.copy()

    X = train_df[feature_cols].values
    y = train_df[label_col].values
    groups = train_df[group_col].values

    oof_probs = np.zeros(len(train_df))
    fold_models = []

    gkf = GroupKFold(n_splits=n_folds)

    # Check for fold-isolated TF-IDF feature computation
    tfidf_feature_names = [
        "tfidf_name_char", "tfidf_name_word", "tfidf_addr_char",
        "tfidf_addr_word", "tfidf_fulltext"
    ]
    has_tfidf = any(c in feature_cols for c in tfidf_feature_names)
    do_fold_tfidf = (
        has_tfidf
        and cand_df is not None
        and s1_tfidf_texts is not None
    )

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info(f"Training fold {fold_idx + 1}/{n_folds}...")

        if do_fold_tfidf:
            # Strictly fit TF-IDF vectorizers ONLY on training fold candidates
            from features import TfidfCosineScorer
            train_cands = set(train_df.iloc[train_idx]["cand_id"])
            fold_cand_train = cand_df[cand_df["entity_id"].isin(train_cands)]

            fold_scorer = TfidfCosineScorer()
            fold_scorer.fit(fold_cand_train)

            # Score training pairs using fold vectorizer
            train_pairs = list(zip(train_df.iloc[train_idx]["s1_id"], train_df.iloc[train_idx]["cand_id"]))
            train_tfidf_scores = fold_scorer.score_pair_batch(s1_tfidf_texts, train_pairs)

            # Index validation candidates with fold vectorizer (transform ONLY, no re-fitting)
            val_cands = set(train_df.iloc[val_idx]["cand_id"])
            fold_cand_val = cand_df[cand_df["entity_id"].isin(val_cands)]
            fold_scorer.index_candidates(fold_cand_val)

            val_pairs = list(zip(train_df.iloc[val_idx]["s1_id"], train_df.iloc[val_idx]["cand_id"]))
            val_tfidf_scores = fold_scorer.score_pair_batch(s1_tfidf_texts, val_pairs)

            # Build fold-specific feature matrices
            X_tr = X[train_idx].copy()
            X_val = X[val_idx].copy()

            # Update TF-IDF columns in X_tr and X_val
            for f_idx, col in enumerate(feature_cols):
                if col in tfidf_feature_names:
                    score_key = col.replace("tfidf_", "")
                    for i, p in enumerate(train_pairs):
                        X_tr[i, f_idx] = train_tfidf_scores.get(p, {}).get(score_key, 0.0)
                    for i, p in enumerate(val_pairs):
                        X_val[i, f_idx] = val_tfidf_scores.get(p, {}).get(score_key, 0.0)

            del fold_cand_train, fold_cand_val, fold_scorer, train_tfidf_scores, val_tfidf_scores
            import gc; gc.collect()
        else:
            X_tr, X_val = X[train_idx], X[val_idx]

        y_tr, y_val = y[train_idx], y[val_idx]

        pos_weight = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
        params = lgb_params.copy()
        params["scale_pos_weight"] = pos_weight

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs

        fold_pos = int(y_val.sum())
        fold_neg = int((y_val == 0).sum())
        logger.info(
            f"  Fold {fold_idx + 1}: val_pos={fold_pos}, val_neg={fold_neg}, "
            f"best_iter={model.best_iteration_}"
        )
        fold_models.append(model)

    # Build OOF DataFrame for threshold optimization
    oof_df = train_df[["s1_id", "cand_id"]].copy()
    oof_df["prob"] = oof_probs

    best_thresh = 0.5
    if gt_dict is not None and all_s1_ids is not None:
        best_thresh, best_f05 = optimize_threshold(oof_df, gt_dict, all_s1_ids)
        logger.info(f"OOF Best threshold: {best_thresh:.2f} | F0.5: {best_f05:.5f}")

    return oof_probs, fold_models, best_thresh



def train_final_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    lgb_params: Optional[Dict] = None,
    n_estimators: Optional[int] = None,
) -> lgb.LGBMClassifier:
    """
    Train a single model on ALL training data (for final test predictions).
    n_estimators should come from OOF fold average.
    """
    if lgb_params is None:
        lgb_params = DEFAULT_LGB_PARAMS.copy()

    X = train_df[feature_cols].values
    y = train_df[label_col].values

    pos_weight = max(1.0, (y == 0).sum() / max(1, (y == 1).sum()))
    params = lgb_params.copy()
    params["scale_pos_weight"] = pos_weight

    if n_estimators is not None:
        params["n_estimators"] = n_estimators

    # Remove early stopping for final model — use fixed n_estimators
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def predict_proba(
    model_or_models,
    X: np.ndarray,
) -> np.ndarray:
    """Predict using single model or ensemble of models."""
    if isinstance(model_or_models, list):
        probs = np.stack([m.predict_proba(X)[:, 1] for m in model_or_models])
        return probs.mean(axis=0)
    return model_or_models.predict_proba(X)[:, 1]


def report_feature_importance(
    models: List[lgb.LGBMClassifier],
    feature_cols: List[str],
    top_n: int = 30,
) -> pd.DataFrame:
    """Report average feature importance across folds."""
    importances = np.zeros(len(feature_cols))
    for model in models:
        importances += model.feature_importances_
    importances /= len(models)

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info(f"\nTop {top_n} feature importances:")
    logger.info(df.head(top_n).to_string(index=False))
    return df


def build_hard_negatives(
    all_candidates_df: pd.DataFrame,
    gt_dict: Dict[str, set],
    feature_cols: List[str],
    neg_to_pos_ratio: float = 5.0,
    hard_neg_score_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    From the candidate pairs, select hard negatives: negative pairs with
    high similarity scores (these confuse the model the most).
    
    Returns dataframe with positives + hard negatives.
    """
    positives = all_candidates_df[all_candidates_df["label"] == 1].copy()
    negatives = all_candidates_df[all_candidates_df["label"] == 0].copy()

    n_pos = len(positives)
    n_neg_target = int(n_pos * neg_to_pos_ratio)

    # Compute a rough similarity score to identify hard negatives
    similarity_cols = [c for c in feature_cols if any(
        k in c for k in ["jaro", "fuzz", "overlap", "tfidf", "exact", "match"]
    )]

    if similarity_cols and len(negatives) > n_neg_target:
        # Score negatives by average similarity
        sim_scores = negatives[similarity_cols].mean(axis=1)
        negatives = negatives.copy()
        negatives["_sim_score"] = sim_scores.values
        # Mix: 60% hard (high sim) + 40% random
        n_hard = int(n_neg_target * 0.6)
        n_rand = n_neg_target - n_hard
        hard_negs = negatives.nlargest(min(n_hard, len(negatives)), "_sim_score")
        rand_mask = ~negatives.index.isin(hard_negs.index)
        rand_pool = negatives[rand_mask]
        if len(rand_pool) > n_rand:
            rand_negs = rand_pool.sample(n=n_rand, random_state=42)
        else:
            rand_negs = rand_pool
        sampled_negatives = pd.concat([hard_negs, rand_negs]).drop(columns=["_sim_score"])
    elif len(negatives) > n_neg_target:
        sampled_negatives = negatives.sample(n=n_neg_target, random_state=42)
    else:
        sampled_negatives = negatives

    result = pd.concat([positives, sampled_negatives]).sample(frac=1, random_state=42)
    logger.info(
        f"Training set: {len(positives)} positives, "
        f"{len(sampled_negatives)} negatives (hard+random)"
    )
    return result.reset_index(drop=True)
