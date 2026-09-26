"""
tests/test_diagnostics.py — Unit tests for false-negative and singleton diagnostics.

Verifies:
1. Category 1: True matches missed by blocking (blocker bottleneck) are accurately identified.
2. Category 2: True matches retrieved as candidates but rejected by the matcher (classifier bottleneck) are accurately identified.
3. Category 3: False positives on singleton S1 records are accurately identified.
4. Accounting identities: FN_total == FN_blocking + FN_matcher.
"""
import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from evaluation import diagnose_oof_errors


class TestDiagnostics(unittest.TestCase):

    def test_partitioned_false_negatives_and_singletons(self):
        """
        Create a known scenario with:
        - S1-A: has true match C-1 (missed by blocking)
        - S1-B: has true match C-2 (retrieved by blocking, but rejected by matcher)
        - S1-C: has true match C-3 (retrieved by blocking AND matched by matcher -> TP)
        - S1-D: singleton S1 (no true match), but matcher predicted C-4 (FP on singleton)
        - S1-E: singleton S1 (no true match), matcher predicted nothing (correct singleton)
        """
        all_s1_ids = ["S1-A", "S1-B", "S1-C", "S1-D", "S1-E"]

        # Ground truth
        gt_dict = {
            "S1-A": {"C-1"},
            "S1-B": {"C-2"},
            "S1-C": {"C-3"},
            "S1-D": set(),  # singleton
            "S1-E": set(),  # singleton
        }

        # Blocker candidate pool
        block_info = {
            "S1-A": {"all_candidates": {"C-OTHER"}},      # C-1 missed by blocking!
            "S1-B": {"all_candidates": {"C-2", "C-ALT"}}, # C-2 retrieved by blocker
            "S1-C": {"all_candidates": {"C-3"}},          # C-3 retrieved by blocker
            "S1-D": {"all_candidates": {"C-4"}},          # candidate retrieved for singleton
            "S1-E": {"all_candidates": set()},            # no candidates
        }

        # Matcher predictions (after thresholding)
        oof_matches = {
            "S1-A": [],         # cannot match C-1 because blocker missed it
            "S1-B": ["C-ALT"],  # matcher picked C-ALT, rejected true match C-2!
            "S1-C": ["C-3"],    # correct match (TP)
            "S1-D": ["C-4"],    # false positive match on singleton!
            "S1-E": [],         # correct singleton
        }

        diag = diagnose_oof_errors(block_info, oof_matches, gt_dict, all_s1_ids, label="Test Diagnostics")

        # 1. True matches missed by blocking
        self.assertEqual(diag["fn_blocking_count"], 1)
        self.assertEqual(diag["sample_fn_blocking"][0]["s1_id"], "S1-A")
        self.assertEqual(diag["sample_fn_blocking"][0]["true_cand_id"], "C-1")

        # 2. True matches retrieved but rejected by matcher
        self.assertEqual(diag["fn_matcher_count"], 1)
        self.assertEqual(diag["sample_fn_matcher"][0]["s1_id"], "S1-B")
        self.assertEqual(diag["sample_fn_matcher"][0]["true_cand_id"], "C-2")

        # True positives
        self.assertEqual(diag["true_positives"], 1)

        # Total false negatives = FN_blocking + FN_matcher
        self.assertEqual(diag["total_false_negatives"], 2)
        self.assertEqual(diag["total_false_negatives"], diag["fn_blocking_count"] + diag["fn_matcher_count"])

        # 3. False positives on singleton S1 records
        self.assertEqual(diag["total_singletons"], 2)
        self.assertEqual(diag["fp_singletons_s1_count"], 1)
        self.assertEqual(diag["fp_singletons_edges"], 1)
        self.assertEqual(diag["sample_fp_singletons"][0]["s1_id"], "S1-D")
        self.assertEqual(diag["sample_fp_singletons"][0]["false_matches"], ["C-4"])

        # Rates
        self.assertAlmostEqual(diag["fn_blocking_rate"], 1 / 3)
        self.assertAlmostEqual(diag["fn_matcher_rate"], 1 / 3)
        self.assertAlmostEqual(diag["fp_singletons_s1_rate"], 1 / 2)


if __name__ == "__main__":
    unittest.main()
