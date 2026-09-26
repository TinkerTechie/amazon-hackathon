"""
tests/test_blocking_chunks.py — Unit tests for chunked blocking implementation.

Verifies:
1. Candidates retrieved from every reference chunk are merged before applying per-S1 budget.
2. Chunk boundaries do not cause true matches located in later chunks to be discarded.
3. Deterministic exact-match candidates are preserved even when candidates exceed max_candidates_per_s1.
"""
import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocessing import preprocess_dataframe
from blocking import MultiBlocker, evaluate_candidate_recall


class TestChunkedBlocking(unittest.TestCase):

    def setUp(self):
        # S1 record to resolve
        self.s1_df = pd.DataFrame([{
            "entity_id": "S1-TARGET",
            "business_name": "Apex Global Logistics Solutions",
            "business_address": "742 Evergreen Terrace Suite 100",
            "country": "us",
        }])
        self.s1_prep = preprocess_dataframe(self.s1_df)

    def test_true_match_in_later_chunk(self):
        """
        Verify that a true match located in a later chunk (chunk 3 of 4)
        is successfully retrieved by chunked blocking.
        """
        # Create 8 candidate records partitioned into chunks of size 2
        # True match is placed in the 4th chunk (index 6, 7)
        cands_data = [
            # Chunk 0
            {"entity_id": "C-01", "business_name": "Omega Health Clinic", "business_address": "100 Main St", "country": "us"},
            {"entity_id": "C-02", "business_name": "Blue Horizon Bakery", "business_address": "200 Market St", "country": "us"},
            # Chunk 1
            {"entity_id": "C-03", "business_name": "Zenith Cloud Consulting", "business_address": "300 Tech Park", "country": "us"},
            {"entity_id": "C-04", "business_name": "Pioneer Metal Works", "business_address": "400 Industrial Way", "country": "us"},
            # Chunk 2
            {"entity_id": "C-05", "business_name": "Summit Financial Partners", "business_address": "500 Wall St", "country": "us"},
            {"entity_id": "C-06", "business_name": "Cascade Software Labs", "business_address": "600 Silicon Blvd", "country": "us"},
            # Chunk 3 (LATER CHUNK containing TRUE MATCH)
            {"entity_id": "C-07", "business_name": "Metro Transit Authority", "business_address": "700 Transit Ave", "country": "us"},
            {"entity_id": "C-TRUE", "business_name": "Apex Global Logistics", "business_address": "742 Evergreen Terrace", "country": "us"},
        ]
        cand_df = pd.DataFrame(cands_data)
        cand_prep = preprocess_dataframe(cand_df)

        # Use cand_chunk_size=2 to force 4 distinct chunks
        blocker = MultiBlocker(top_k=5, max_candidates_per_s1=20, cand_chunk_size=2)
        block_info = blocker.generate_candidates(self.s1_prep, cand_prep)

        retrieved = block_info["S1-TARGET"]["all_candidates"]
        self.assertIn(
            "C-TRUE", retrieved,
            f"True match C-TRUE in chunk 3 was not retrieved! Retrieved: {retrieved}"
        )

        # Ground truth recall check
        gt = pd.DataFrame([{
            "source1_entity_id": "S1-TARGET",
            "matched_entity_ids": "C-TRUE",
        }])
        stats = evaluate_candidate_recall(block_info, gt, label="Later Chunk Recall Test")
        self.assertEqual(stats["candidate_recall"], 1.0)
        self.assertEqual(stats["recovered"], 1)

    def test_exact_matches_preserved_under_budget_constraint(self):
        """
        Verify that deterministic exact matches are preserved even when the total
        candidate pool exceeds max_candidates_per_s1, across multiple chunks.
        """
        # Create 12 candidate records, including 2 exact matches in different chunks
        cands_data = []
        # Chunks 0-2: filler approximate candidates that share tokens
        for i in range(10):
            cands_data.append({
                "entity_id": f"C-APPROX-{i:02d}",
                "business_name": f"Apex Solutions Logistics Branch {i}",
                "business_address": f"{700 + i} Evergreen Terrace",
                "country": "us",
            })
        # Later chunk exact match (exact clean / stripped name)
        cands_data.append({
            "entity_id": "C-EXACT-CLEAN",
            "business_name": "Apex Global Logistics Solutions",
            "business_address": "Different Address 999",
            "country": "us",
        })
        cands_data.append({
            "entity_id": "C-EXACT-STRIPPED",
            "business_name": "Apex Global Logistics Solutions Inc",
            "business_address": "Another Address 888",
            "country": "us",
        })

        cand_df = pd.DataFrame(cands_data)
        cand_prep = preprocess_dataframe(cand_df)

        # Small chunk size = 3 (4 chunks total), tight budget of 4 candidates
        blocker = MultiBlocker(top_k=5, max_candidates_per_s1=4, cand_chunk_size=3)
        block_info = blocker.generate_candidates(self.s1_prep, cand_prep)

        retrieved = block_info["S1-TARGET"]["all_candidates"]
        self.assertLessEqual(len(retrieved), 4)

        # Both exact matches MUST be preserved
        self.assertIn("C-EXACT-CLEAN", retrieved, "Exact clean match was discarded by candidate budget!")
        self.assertIn("C-EXACT-STRIPPED", retrieved, "Exact stripped match was discarded by candidate budget!")

    def test_deterministic_ranking_across_runs(self):
        """
        Verify that candidate selection and ranking are completely deterministic
        across multiple runs with chunking enabled.
        """
        cands_data = [
            {"entity_id": f"C-{i:02d}", "business_name": f"Apex {i} Logistics", "business_address": f"{i} Terrace", "country": "us"}
            for i in range(20)
        ]
        cand_df = pd.DataFrame(cands_data)
        cand_prep = preprocess_dataframe(cand_df)

        blocker = MultiBlocker(top_k=3, max_candidates_per_s1=6, cand_chunk_size=4)
        run1 = blocker.generate_candidates(self.s1_prep, cand_prep)["S1-TARGET"]["all_candidates"]
        run2 = blocker.generate_candidates(self.s1_prep, cand_prep)["S1-TARGET"]["all_candidates"]

        self.assertEqual(run1, run2, "Chunked blocking candidate generation is not deterministic!")

    def test_bounded_memory_across_many_chunks(self):
        """
        Verify that candidate stores remain bounded and do not grow unboundedly
        across many candidate chunks, while preserving recall on true matches.
        """
        # 5 S1 records
        s1_records = [
            {
                "entity_id": f"S1-{i}",
                "business_name": f"Enterprise Solutions Hub {i}",
                "business_address": f"{100 + i} Main St Suite {i}",
                "country": "us",
            }
            for i in range(5)
        ]
        s1_prep = preprocess_dataframe(pd.DataFrame(s1_records))

        # 50 candidates spread across 10 chunks of size 5
        # Embed true matches in chunk 8 (later chunk)
        cands_records = []
        for i in range(50):
            if i == 42:
                # True match for S1-2 in chunk 8
                cands_records.append({
                    "entity_id": "C-TRUE-42",
                    "business_name": "Enterprise Solutions Hub 2",
                    "business_address": "102 Main St Suite 2",
                    "country": "us",
                })
            else:
                cands_records.append({
                    "entity_id": f"C-{i:03d}",
                    "business_name": f"Enterprise Solutions Branch {i % 10}",
                    "business_address": f"{100 + (i % 5)} Main St Suite 100",
                    "country": "us",
                })

        cand_prep = preprocess_dataframe(pd.DataFrame(cands_records))
        blocker = MultiBlocker(top_k=5, max_candidates_per_s1=10, cand_chunk_size=5)
        block_info = blocker.generate_candidates(s1_prep, cand_prep)

        # All S1 candidate sets must be strictly capped by max_candidates_per_s1
        for sid, r in block_info.items():
            self.assertLessEqual(len(r["all_candidates"]), 10)

        # True match in later chunk must be present
        self.assertIn("C-TRUE-42", block_info["S1-2"]["all_candidates"])


if __name__ == "__main__":
    unittest.main()

