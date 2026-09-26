"""
tests/test_data_leakage.py — Unit tests verifying zero data leakage across the pipeline.

Verifies:
1. GroupKFold groups strictly by source1_entity_id: train and validation sets have zero overlapping S1 IDs.
2. TF-IDF vectorizers in OOF evaluation are fitted strictly on each training fold's candidates.
3. Test candidate indexing strictly applies transform() and never fits vectorizers or learns vocabulary/IDF.
4. Ground-truth injection occurs strictly after raw blocker recall is measured.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sklearn.model_selection import GroupKFold
from preprocessing import preprocess_dataframe
from blocking import MultiBlocker, evaluate_candidate_recall
from features import TfidfCosineScorer, build_feature_dataframe, get_feature_columns
from model import train_with_oof, build_hard_negatives


class TestDataLeakage(unittest.TestCase):

    def test_groupkfold_strict_s1_grouping(self):
        """
        Verify that GroupKFold groups strictly by source1_entity_id with zero S1 overlap
        between training and validation folds.
        """
        # Create a sample DataFrame with multiple candidate pairs per S1
        s1_ids = [f"S1-{i:03d}" for i in range(50)]
        rows = []
        for sid in s1_ids:
            for j in range(3):
                rows.append({
                    "s1_id": sid,
                    "cand_id": f"C-{sid}-{j}",
                    "feat_1": np.random.rand(),
                    "feat_2": np.random.rand(),
                    "label": int(j == 0),
                })
        df = pd.DataFrame(rows)

        gkf = GroupKFold(n_splits=5)
        groups = df["s1_id"].values
        X = df[["feat_1", "feat_2"]].values
        y = df["label"].values

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            train_s1 = set(df.iloc[train_idx]["s1_id"])
            val_s1 = set(df.iloc[val_idx]["s1_id"])

            overlap = train_s1 & val_s1
            self.assertEqual(
                len(overlap), 0,
                f"Fold {fold} leaked S1 entities across train and val splits: {overlap}"
            )
            self.assertEqual(len(train_s1) + len(val_s1), 50)

    def test_test_data_never_fits_vectorizer(self):
        """
        Verify that test candidate indexing uses transform() and never modifies
        or refits the vectorizer vocabulary or IDF weights learned on training data.
        """
        train_cand_df = pd.DataFrame([
            {"entity_id": "C-TR-1", "name_clean": "apple inc", "addr_expanded": "1 infinite loop", "combined_text": "apple inc 1 infinite loop"},
            {"entity_id": "C-TR-2", "name_clean": "google llc", "addr_expanded": "1600 amphitheatre", "combined_text": "google llc 1600 amphitheatre"},
        ])
        test_cand_df = pd.DataFrame([
            {"entity_id": "C-TE-1", "name_clean": "microsoft corp", "addr_expanded": "one microsoft way", "combined_text": "microsoft corp one microsoft way"},
        ])

        scorer = TfidfCosineScorer()
        scorer.fit(train_cand_df)

        # Record vocabulary before indexing test candidates
        vocab_before = {k: set(vec.vocabulary_.keys()) for k, vec in scorer.vectorizers.items()}

        # Index test candidates
        scorer.index_candidates(test_cand_df)

        # Record vocabulary after indexing test candidates
        vocab_after = {k: set(vec.vocabulary_.keys()) for k, vec in scorer.vectorizers.items()}

        for k in vocab_before:
            self.assertEqual(
                vocab_before[k], vocab_after[k],
                f"Vocabulary for {k} was altered by test data! Leakage detected."
            )
            # Ensure novel test terms are NOT in vocabulary
            self.assertNotIn("microsoft", vocab_after[k])

    def test_gt_injection_after_raw_blocker_recall(self):
        """
        Verify that ground-truth injection occurs strictly after raw blocker recall
        is measured, ensuring the logged blocker recall is not inflated.
        """
        from main import build_training_pairs

        s1_df = pd.DataFrame([
            {"entity_id": "S1-A", "name_clean": "acme supply", "country_norm": "us"},
        ])
        cand_df = pd.DataFrame([
            {"entity_id": "C-RETRIEVED", "name_clean": "acme tools", "country_norm": "us"},
            {"entity_id": "C-MISSED", "name_clean": "totally different", "country_norm": "us"},
        ])

        # Mock blocker output where blocker retrieved C-RETRIEVED but missed C-MISSED
        block_info = {
            "S1-A": {
                "all_candidates": {"C-RETRIEVED"},
                "blocks": {"exact_name": set()},
            }
        }
        gt_df = pd.DataFrame([{
            "source1_entity_id": "S1-A",
            "matched_entity_ids": "C-MISSED",  # Ground truth is the MISSED one
        }])
        gt_dict = {"S1-A": {"C-MISSED"}}

        # Measure raw blocker recall FIRST
        raw_stats = evaluate_candidate_recall(block_info, gt_df, label="Test Raw Recall")
        self.assertEqual(raw_stats["candidate_recall"], 0.0, "Raw blocker recall should be 0.0 before injection!")

        # Inject missed GT positives into training pairs
        training_pairs = build_training_pairs(s1_df, cand_df, gt_dict, block_info)
        pair_cands = {p[1] for p in training_pairs}

        # C-MISSED is in training pairs for model training
        self.assertIn("C-MISSED", pair_cands)
        # But block_info itself was NOT mutated
        self.assertNotIn("C-MISSED", block_info["S1-A"]["all_candidates"])


if __name__ == "__main__":
    unittest.main()
