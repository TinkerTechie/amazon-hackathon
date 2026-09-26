"""
blocking.py — Multi-strategy candidate generation for entity resolution.

Implements 9 blocking strategies and unions their outputs:
  A. Exact normalized name
  B. Exact stripped name
  C. Name character TF-IDF (3-5 char n-grams)
  D. Name word TF-IDF (1-2 word n-grams)
  E. Address character TF-IDF
  F. Address word TF-IDF
  G. Full-text (name + address) TF-IDF
  H. Numeric/address token blocking (house number, PIN)
  I. Rare-token inverted index

All strategies are country-restricted by default, with a fallback pass
that relaxes the country constraint if recall drops too low.

Scalability notes:
- top_k_cosine() never materialises a full (n_queries × n_reference) dense
  matrix.  It operates on sparse rows only, extracting nonzero entries before
  argpartition, so peak memory is O(batch_size × avg_nnz) instead of
  O(n_queries × n_reference).
- Candidate truncation uses cumulative similarity scores (ranked, descending)
  so the strongest candidates survive, not an arbitrary prefix of a set.
"""
import logging
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def get_process_rss_mb() -> float:
    """Return resident set size (RSS) of current process in megabytes."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        try:
            import resource
            import sys
            rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return rusage / (1024.0 * 1024.0)
            else:
                return rusage / 1024.0
        except Exception:
            return 0.0


# ─── TF-IDF helpers ──────────────────────────────────────────────────────────

def build_tfidf_index(
    corpus: List[str],
    analyzer: str,
    ngram_range: Tuple[int, int],
    max_features: int = 200_000,
    sublinear_tf: bool = True,
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """Fit a TF-IDF vectorizer on corpus and return (vectorizer, matrix)."""
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=1,
        dtype=np.float32,
    )
    mat = vec.fit_transform(corpus)
    mat = normalize(mat, norm="l2", copy=False)
    return vec, mat


def top_k_cosine(
    query_mat: csr_matrix,
    index_mat: csr_matrix,
    top_k: int,
    batch_size: int = 500,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    For each query row, retrieve top-K candidates from index_mat by cosine
    similarity.  Returns {query_idx: [(index_idx, score), ...]}.

    Memory safety: we never materialise the full (n_queries × n_index) dense
    matrix.  Each sparse row's nonzero entries are extracted directly via CSR
    pointers (indices, data) without allocating temporary csr_matrix wrappers.
    Temporary batch matrices and transposed index matrices are explicitly freed.
    """
    results: Dict[int, List[Tuple[int, float]]] = {}
    n_queries = query_mat.shape[0]
    index_mat_t = index_mat.T.tocsr()   # pre-transpose once; reused across batches

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_mat[start:end]

        # Sparse matmul → sparse result; NO dense materialisation
        sim_sparse = (batch @ index_mat_t).tocsr()
        indptr = sim_sparse.indptr
        indices = sim_sparse.indices
        data = sim_sparse.data

        for i in range(sim_sparse.shape[0]):
            q_idx = start + i
            r_start = indptr[i]
            r_end = indptr[i + 1]
            if r_start == r_end:
                results[q_idx] = []
                continue

            cols = indices[r_start:r_end]      # nonzero column positions
            vals = data[r_start:r_end]         # corresponding cosine scores

            k = min(top_k, len(vals))
            if k < len(vals):
                part_idx = np.argpartition(vals, -k)[-k:]
                part_idx = part_idx[np.argsort(vals[part_idx])[::-1]]
            else:
                part_idx = np.argsort(vals)[::-1]

            results[q_idx] = [
                (int(cols[j]), float(vals[j]))
                for j in part_idx
                if vals[j] > 0.0
            ]

        del sim_sparse, batch

    del index_mat_t
    return results


# ─── Exact-match blocking ─────────────────────────────────────────────────────

def exact_block(
    s1_col: pd.Series,
    candidates_col: pd.Series,
    s1_ids: pd.Series,
    cand_ids: pd.Series,
) -> Dict[str, Set[str]]:
    """
    Build an inverted index: value → set of candidate IDs.
    Returns {s1_id: {cand_id, ...}} for matches.
    """
    # Build value → cand_ids map
    val_to_cands = defaultdict(set)
    for cid, val in zip(cand_ids, candidates_col):
        if val and isinstance(val, str) and val.strip():
            val_to_cands[val.strip()].add(cid)

    result = defaultdict(set)
    for sid, val in zip(s1_ids, s1_col):
        if val and isinstance(val, str) and val.strip():
            v = val.strip()
            if v in val_to_cands:
                result[sid].update(val_to_cands[v])
    return result


# ─── Numeric token blocking ───────────────────────────────────────────────────

def numeric_block(
    s1_numerics: pd.Series,
    cand_numerics: pd.Series,
    s1_ids: pd.Series,
    cand_ids: pd.Series,
    min_token_len: int = 3,
    max_token_len: int = 10,
) -> Dict[str, Set[str]]:
    """
    Block on shared numeric tokens (house numbers, PIN codes).
    Only use numeric tokens of length >= min_token_len to avoid
    very common tokens like '1', '2', etc.
    """
    # Build numeric_token → cand_ids
    num_to_cands = defaultdict(set)
    for cid, nums in zip(cand_ids, cand_numerics):
        for n in (nums or []):
            if min_token_len <= len(n) <= max_token_len:
                num_to_cands[n].add(cid)

    result = defaultdict(set)
    for sid, nums in zip(s1_ids, s1_numerics):
        for n in (nums or []):
            if min_token_len <= len(n) <= max_token_len:
                # Disregard ubiquitous numeric tokens (e.g. 500+ hits in a single chunk) to bound memory
                if n in num_to_cands and len(num_to_cands[n]) <= 500:
                    result[sid].update(num_to_cands[n])
    return result


# ─── Rare-token inverted index blocking ───────────────────────────────────────

def build_token_frequencies(
    *text_series_list,
) -> Dict[str, int]:
    """Compute global term frequency across all texts without concatenating."""
    freq = defaultdict(int)
    for item in text_series_list:
        if isinstance(item, (pd.Series, list)):
            for text in item:
                if isinstance(text, str):
                    for tok in text.split():
                        if len(tok) >= 2:
                            freq[tok] += 1
        elif isinstance(item, str):
            for tok in item.split():
                if len(tok) >= 2:
                    freq[tok] += 1
    return dict(freq)


def rare_token_block(
    s1_texts: pd.Series,
    cand_texts: pd.Series,
    s1_ids: pd.Series,
    cand_ids: pd.Series,
    global_freq: Dict[str, int],
    max_freq_pct: float = 0.005,
    total_docs: Optional[int] = None,
) -> Dict[str, Set[str]]:
    """
    Block using tokens that appear in <= max_freq_pct fraction of documents.
    These rare tokens are highly discriminative.
    """
    if total_docs is None:
        total_docs = len(s1_texts) + len(cand_texts)

    max_freq = int(max_freq_pct * total_docs)
    max_freq = max(5, min(max_freq, 2000))

    # Index candidate rare tokens
    tok_to_cands = defaultdict(set)
    for cid, text in zip(cand_ids, cand_texts):
        if not isinstance(text, str):
            continue
        for tok in text.split():
            if len(tok) >= 3 and global_freq.get(tok, 0) <= max_freq:
                tok_to_cands[tok].add(cid)

    result = defaultdict(set)
    for sid, text in zip(s1_ids, s1_texts):
        if not isinstance(text, str):
            continue
        for tok in text.split():
            if len(tok) >= 3 and global_freq.get(tok, 0) <= max_freq:
                if tok in tok_to_cands:
                    result[sid].update(tok_to_cands[tok])
    return result


# ─── Multi-strategy blocker (country-aware) ───────────────────────────────────

# ─── Compact Candidate Tracker ───────────────────────────────────────────────

class CandidateTracker:
    """
    Compact per-S1 candidate store that bounds memory while preserving candidate recall.

    For each S1:
      - exact_cands: set of candidate IDs matching exact name / stripped name.
        (Always preserved with priority 0, never pruned unless exact matches alone exceed budget).
      - scores: dict of candidate_id -> cumulative similarity score.
      - blocks: dict of block_name -> set of candidate IDs.

    Compaction:
      When candidate count for an S1 exceeds buffer_limit, prune to top candidates:
        1. Exact match priority (rank 0)
        2. Score descending
        3. Candidate ID ascending (deterministic tie-break)
      Pruned candidates are also removed from blocks to keep block sets compact.
    """

    def __init__(
        self,
        s1_ids: List[str],
        max_candidates_per_s1: int = 200,
        buffer_multiplier: int = 2,
    ):
        self.s1_ids = s1_ids
        self.max_candidates = max_candidates_per_s1
        self.buffer_limit = max(max_candidates_per_s1 * buffer_multiplier, 500)
        self.keep_limit = max(max_candidates_per_s1 + 50, max_candidates_per_s1)

        self.scores: Dict[str, Dict[str, float]] = {sid: {} for sid in s1_ids}
        self.exact_cands: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}
        self.blocks: Dict[str, Dict[str, Set[str]]] = {
            sid: defaultdict(set) for sid in s1_ids
        }

    def add_exact_matches(
        self,
        block_name: str,
        matches: Dict[str, Set[str]],
        default_score: float = 3.0,
    ):
        for sid, cand_set in matches.items():
            if sid not in self.scores:
                continue
            self.exact_cands[sid].update(cand_set)
            self.blocks[sid][block_name].update(cand_set)
            s_dict = self.scores[sid]
            for cid in cand_set:
                s_dict[cid] = s_dict.get(cid, 0.0) + default_score

    def add_token_matches(
        self,
        block_name: str,
        matches: Dict[str, Set[str]],
        default_score: float = 1.5,
    ):
        for sid, cand_set in matches.items():
            if sid not in self.scores:
                continue
            self.blocks[sid][block_name].update(cand_set)
            s_dict = self.scores[sid]
            for cid in cand_set:
                s_dict[cid] = s_dict.get(cid, 0.0) + default_score

    def add_tfidf_hits(
        self,
        block_name: str,
        top_hits: Dict[int, List[Tuple[int, float]]],
        chunk_cand_ids: List[str],
        s1_ids: List[str],
    ):
        for qi, hits in top_hits.items():
            sid = s1_ids[qi]
            if sid not in self.scores:
                continue
            s_dict = self.scores[sid]
            b_set = self.blocks[sid][block_name]
            for ci, score in hits:
                cid = chunk_cand_ids[ci]
                b_set.add(cid)
                s_dict[cid] = s_dict.get(cid, 0.0) + float(score)

    def compact(self, s1_subset: Optional[List[str]] = None):
        """
        Prune candidate pools exceeding buffer_limit.
        Always preserves exact-match candidates, keeping top candidates by cumulative score.
        """
        target_sids = s1_subset if s1_subset is not None else self.s1_ids
        for sid in target_sids:
            s_dict = self.scores.get(sid)
            if not s_dict or len(s_dict) <= self.buffer_limit:
                continue
            exact_set = self.exact_cands[sid]
            ranked = sorted(
                s_dict.keys(),
                key=lambda c: (
                    0 if c in exact_set else 1,
                    -s_dict[c],
                    c,
                ),
            )
            kept = set(ranked[: self.keep_limit])
            dropped = set(s_dict.keys()) - kept
            for c in dropped:
                del s_dict[c]
                exact_set.discard(c)
            for b_set in self.blocks[sid].values():
                b_set.difference_update(dropped)

    def to_final_results(self) -> Dict[str, Dict]:
        """
        Produce the final dictionary format expected by downstream evaluation and inference:
        {
          sid: {
            "all_candidates": set of candidate entity_ids,
            "blocks": {block_name: set of candidate entity_ids},
            "scores": {candidate_id: cumulative_score},
          }
        }
        """
        final_results = {}
        total_pairs = 0
        for sid in self.s1_ids:
            s_dict = self.scores.get(sid, {})
            exact_set = self.exact_cands.get(sid, set())
            cands = list(s_dict.keys())
            if len(cands) > self.max_candidates:
                ranked = sorted(
                    cands,
                    key=lambda c: (
                        0 if c in exact_set else 1,
                        -s_dict[c],
                        c,
                    ),
                )
                kept = set(ranked[: self.max_candidates])
                dropped = set(cands) - kept
                for b_set in self.blocks[sid].values():
                    b_set.difference_update(dropped)
            else:
                kept = set(cands)

            final_blocks = {
                b_name: b_set
                for b_name, b_set in self.blocks[sid].items()
                if b_set
            }
            final_results[sid] = {
                "all_candidates": kept,
                "blocks": final_blocks,
                "scores": {c: s_dict[c] for c in kept},
            }
            total_pairs += len(kept)

        logger.info(
            f"Total candidate pairs: {total_pairs} | "
            f"Avg per S1: {total_pairs / max(1, len(self.s1_ids)):.1f}"
        )
        return final_results


# ─── Multi-strategy blocker (country-aware) ───────────────────────────────────

class MultiBlocker:
    """
    Combines all blocking strategies into a single candidate generator.
    Works country-by-country for efficiency, with a global fallback.
    Processes reference candidates in chunks to guarantee bounded memory.
    """

    def __init__(
        self,
        top_k: int = 50,
        max_candidates_per_s1: int = 200,
        fallback_fraction: float = 0.10,
        cand_chunk_size: int = 250_000,
    ):
        self.top_k = top_k
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.fallback_fraction = fallback_fraction
        self.cand_chunk_size = cand_chunk_size

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        cand_df: pd.DataFrame,  # S2 + S3 combined
    ) -> Dict[str, Dict[str, Set[str]]]:
        """
        Returns:
          {
            s1_id: {
              "all_candidates": set of candidate entity_ids,
              "blocks": {block_name: set of candidate_ids}  # for feature creation
            }
          }
        """
        import gc
        tracker = CandidateTracker(
            s1_ids=s1_df["entity_id"].tolist(),
            max_candidates_per_s1=self.max_candidates_per_s1,
        )

        countries = s1_df["country_norm"].unique().tolist()
        logger.info(f"Blocking across {len(countries)} countries: {countries} (cand_chunk_size={self.cand_chunk_size})")

        # Compute global token frequencies without concatenating large Series
        global_freq = build_token_frequencies(s1_df["name_clean"], cand_df["name_clean"])
        total_docs = len(s1_df) + len(cand_df)
        logger.info(f"Built global token frequency dict with {len(global_freq)} terms")

        # Only extract columns needed for blocking to avoid copying unused fields
        cand_cols = [
            c for c in [
                "entity_id", "country_norm", "name_clean", "name_stripped",
                "addr_expanded", "addr_numerics", "combined_text"
            ]
            if c in cand_df.columns
        ]
        s1_cols = [
            c for c in [
                "entity_id", "country_norm", "name_clean", "name_stripped",
                "addr_expanded", "addr_numerics", "combined_text"
            ]
            if c in s1_df.columns
        ]

        # ── Country-restricted blocking ────────────────────────────────────
        for country in countries:
            s1_country = s1_df.loc[s1_df["country_norm"] == country, s1_cols]
            cand_country = cand_df.loc[cand_df["country_norm"] == country, cand_cols]

            if len(s1_country) == 0 or len(cand_country) == 0:
                logger.info(f"  Skipping country '{country}': no candidates or S1 records")
                del s1_country, cand_country
                continue

            logger.info(
                f"  Country '{country}': {len(s1_country)} S1, {len(cand_country)} candidates"
            )

            self._block_partition(
                s1_country, cand_country, tracker, global_freq, total_docs,
                partition_name=f"country_{country}"
            )
            del s1_country, cand_country
            gc.collect()

        # ── Fallback: global blocking for S1 entities with zero candidates ──
        s1_with_no_cands = [
            sid for sid in s1_df["entity_id"]
            if len(tracker.scores.get(sid, {})) == 0
        ]
        if s1_with_no_cands:
            logger.info(
                f"  Fallback blocking for {len(s1_with_no_cands)} S1 entities with 0 candidates"
            )
            s1_fallback = s1_df.loc[s1_df["entity_id"].isin(s1_with_no_cands), s1_cols]
            n_cand_sample = min(len(cand_df), max(50_000, int(len(cand_df) * self.fallback_fraction)))
            cand_sample = cand_df[cand_cols].sample(n=n_cand_sample, random_state=42)
            self._block_partition(
                s1_fallback, cand_sample, tracker, global_freq, total_docs,
                partition_name="global_fallback"
            )
            del s1_fallback, cand_sample
            gc.collect()

        return tracker.to_final_results()

    def _block_partition(
        self,
        s1: pd.DataFrame,
        cands: pd.DataFrame,
        tracker: CandidateTracker,
        global_freq: Dict,
        total_docs: int,
        partition_name: str,
    ):
        """
        Run all blocking strategies on a country/partition subset.
        Processes reference candidates in chunks to guarantee bounded peak RAM.
        Logs RSS before and after every chunk.
        """
        import gc
        s1_ids = s1["entity_id"].tolist()
        n_cands = len(cands)
        if n_cands == 0 or len(s1_ids) == 0:
            return

        chunk_size = self.cand_chunk_size
        n_chunks = max(1, (n_cands + chunk_size - 1) // chunk_size)
        logger.info(
            f"    [{partition_name}] Processing {n_cands:,} candidates across "
            f"{n_chunks} chunk(s) of <= {chunk_size:,} | Initial RSS: {get_process_rss_mb():.1f} MB"
        )

        # ── Pre-fit TF-IDF vectorizers and transform S1 queries ───────────
        # Sample evenly across reference candidates for vocabulary fitting
        if n_cands <= 250_000:
            sample_indices = np.arange(n_cands)
        else:
            sample_indices = np.linspace(0, n_cands - 1, 250_000, dtype=int)

        tfidf_configs = [
            ("name_char_tfidf", "name_clean", "char_wb", (3, 5)),
            ("name_word_tfidf", "name_clean", "word", (1, 2)),
            ("addr_char_tfidf", "addr_expanded", "char_wb", (3, 5)),
            ("addr_word_tfidf", "addr_expanded", "word", (1, 2)),
            ("fulltext_tfidf", "combined_text", "word", (1, 2)),
        ]

        tfidf_models = {}
        for block_name, col_name, analyzer, ngrams in tfidf_configs:
            if col_name not in s1.columns or col_name not in cands.columns:
                continue
            try:
                s1_texts = s1[col_name].fillna("").tolist()
                fit_sample = cands[col_name].iloc[sample_indices].fillna("").tolist()
                fit_corpus = s1_texts + fit_sample
                vec, _ = build_tfidf_index(fit_corpus, analyzer, ngrams)
                s1_mat = normalize(vec.transform(s1_texts), norm="l2")
                tfidf_models[block_name] = (vec, s1_mat, col_name)
                del s1_texts, fit_sample, fit_corpus
            except Exception as e:
                logger.warning(f"    [{partition_name}] Pre-fit TF-IDF {block_name} failed: {e}")
        gc.collect()

        # ── Stream through candidate chunks (single pass) ─────────────────
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_cands)
            chunk_len = end_idx - start_idx

            rss_before = get_process_rss_mb()
            logger.info(
                f"    [{partition_name}] >>> Chunk {chunk_idx + 1}/{n_chunks} "
                f"({chunk_len:,} candidates) START | RSS: {rss_before:.1f} MB"
            )

            # Slice single candidate chunk
            chunk = cands.iloc[start_idx:end_idx]
            chunk_ids = chunk["entity_id"].tolist()

            # 1. Exact normalized name
            if "name_clean" in s1.columns and "name_clean" in chunk.columns:
                try:
                    m_a = exact_block(s1["name_clean"], chunk["name_clean"], s1_ids, chunk_ids)
                    tracker.add_exact_matches("exact_name", m_a, default_score=3.0)
                    del m_a
                except Exception as e:
                    logger.warning(f"    [{partition_name}] Block exact_name failed in chunk {chunk_idx + 1}: {e}")

            # 2. Exact stripped name
            if "name_stripped" in s1.columns and "name_stripped" in chunk.columns:
                try:
                    m_b = exact_block(s1["name_stripped"], chunk["name_stripped"], s1_ids, chunk_ids)
                    tracker.add_exact_matches("exact_stripped", m_b, default_score=2.5)
                    del m_b
                except Exception as e:
                    logger.warning(f"    [{partition_name}] Block exact_stripped failed in chunk {chunk_idx + 1}: {e}")

            # 3. Numeric / address blocking
            if "addr_numerics" in s1.columns and "addr_numerics" in chunk.columns:
                try:
                    m_h = numeric_block(s1["addr_numerics"], chunk["addr_numerics"], s1_ids, chunk_ids)
                    tracker.add_token_matches("numeric_addr", m_h, default_score=1.5)
                    del m_h
                except Exception as e:
                    logger.warning(f"    [{partition_name}] Block numeric_addr failed in chunk {chunk_idx + 1}: {e}")

            # 4. Rare token blocking
            if "name_clean" in s1.columns and "name_clean" in chunk.columns:
                try:
                    m_i = rare_token_block(
                        s1["name_clean"], chunk["name_clean"],
                        s1_ids, chunk_ids,
                        global_freq=global_freq,
                        total_docs=total_docs,
                    )
                    tracker.add_token_matches("rare_token", m_i, default_score=2.0)
                    del m_i
                except Exception as e:
                    logger.warning(f"    [{partition_name}] Block rare_token failed in chunk {chunk_idx + 1}: {e}")

            # 5. TF-IDF blocks
            for block_name, (vec, s1_mat, col_name) in tfidf_models.items():
                try:
                    c_texts = chunk[col_name].fillna("").tolist()
                    c_mat = normalize(vec.transform(c_texts), norm="l2")
                    top_hits = top_k_cosine(s1_mat, c_mat, self.top_k)
                    tracker.add_tfidf_hits(block_name, top_hits, chunk_ids, s1_ids)
                    del c_texts, c_mat, top_hits
                except Exception as e:
                    logger.warning(f"    [{partition_name}] {block_name} failed in chunk {chunk_idx + 1}: {e}")

            # Compact per-S1 candidate structures to bound memory
            tracker.compact(s1_ids)

            # Free all chunk-local objects immediately
            del chunk, chunk_ids
            gc.collect()

            rss_after = get_process_rss_mb()
            delta = rss_after - rss_before
            logger.info(
                f"    [{partition_name}] <<< Chunk {chunk_idx + 1}/{n_chunks} "
                f"({chunk_len:,} candidates) END   | RSS: {rss_after:.1f} MB (Delta: {delta:+.1f} MB)"
            )

        # Free fitted TF-IDF models for this partition
        del tfidf_models
        gc.collect()



def evaluate_candidate_recall(
    candidates: Dict[str, Dict],
    gt: pd.DataFrame,
    # NOTE: This function measures raw blocker recall — before any training-pair
    # augmentation that injects missed positives.  Always call this *before*
    # build_training_pairs() so the reported recall reflects what the blocker
    # actually retrieved on its own.
    label: str = "",
) -> Dict:
    """
    Evaluate blocking recall against ground truth.
    gt must have columns: source1_entity_id, matched_entity_ids (comma-sep str)
    """
    gt_dict = {}
    for _, row in gt.iterrows():
        sid = row["source1_entity_id"]
        m = str(row.get("matched_entity_ids", "")).strip()
        gt_dict[sid] = set(x.strip() for x in m.split(",") if x.strip())

    total_true_matches = 0
    recovered = 0
    total_candidates = 0
    s1_count = 0

    for sid, r in candidates.items():
        if sid not in gt_dict:
            continue
        true_matches = gt_dict[sid]
        found_cands = r["all_candidates"]
        total_true_matches += len(true_matches)
        recovered += len(true_matches & found_cands)
        total_candidates += len(found_cands)
        s1_count += 1

    recall = recovered / max(1, total_true_matches)
    avg_cands = total_candidates / max(1, s1_count)
    reduction_ratio = 1.0 - (total_candidates / max(1, s1_count * (len(candidates) - 1)))

    result = {
        "label": label,
        "candidate_recall": recall,
        "total_true_matches": total_true_matches,
        "recovered": recovered,
        "total_candidates": total_candidates,
        "avg_candidates_per_s1": avg_cands,
        "s1_evaluated": s1_count,
    }
    logger.info(
        f"[Blocking Recall] {label}: recall={recall:.4f} | "
        f"avg_cands={avg_cands:.1f} | total_pairs={total_candidates}"
    )
    return result


def validate_blocking(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    gt: pd.DataFrame,
    blocker: "MultiBlocker",
    sample_size: int = 5_000,
    seed: int = 42,
) -> Dict:
    """
    Lightweight validation mode: run the complete blocking pipeline on a
    small, stratified sample of S1 records and report:
      - Blocker candidate recall  (unadjusted — no GT injection)
      - Average candidates per S1
      - Singleton fraction after blocking
      - Approximate precision (fraction of candidates that are true matches)
      - F0.5 score

    Call this BEFORE the full dataset run.  Raises RuntimeError if recall < 0.60.
    """
    rng = np.random.RandomState(seed)
    gt_dict: Dict[str, Set[str]] = {}
    for _, row in gt.iterrows():
        sid = row["source1_entity_id"]
        m = str(row.get("matched_entity_ids", "")).strip()
        gt_dict[sid] = set(x.strip() for x in m.split(",") if x.strip())

    # Stratified sample: 80% with GT matches, 20% singletons
    matched_ids   = [sid for sid in s1_df["entity_id"] if gt_dict.get(sid)]
    singleton_ids = [sid for sid in s1_df["entity_id"] if not gt_dict.get(sid)]

    n_matched   = min(int(sample_size * 0.8), len(matched_ids))
    n_singleton = min(sample_size - n_matched, len(singleton_ids))

    sampled_ids = set(
        rng.choice(matched_ids,   n_matched,   replace=False).tolist() +
        rng.choice(singleton_ids, n_singleton, replace=False).tolist()
    )
    s1_sample = s1_df[s1_df["entity_id"].isin(sampled_ids)].copy()
    gt_sample  = gt[gt["source1_entity_id"].isin(sampled_ids)].copy()

    logger.info(
        f"\n{'='*60}\n"
        f"VALIDATION MODE: blocking on {len(s1_sample)} S1 records "
        f"({n_matched} with GT, {n_singleton} singletons)\n"
        f"{'='*60}"
    )
    t0 = time.perf_counter()
    block_info = blocker.generate_candidates(s1_sample, cand_df)
    elapsed = time.perf_counter() - t0
    logger.info(f"Validation blocking took {elapsed:.1f}s")

    # Recall — raw blocker, no GT injection
    recall_stats = evaluate_candidate_recall(block_info, gt_sample, label="Validation")

    total_cands = recall_stats["total_candidates"]
    total_tp    = recall_stats["recovered"]
    precision   = total_tp / max(1, total_cands)
    recall      = recall_stats["candidate_recall"]

    beta = 0.5
    f05  = (1 + beta**2) * precision * recall / max(1e-9, beta**2 * precision + recall)

    n_singletons_after = sum(
        1 for sid, r in block_info.items() if len(r["all_candidates"]) == 0
    )
    singleton_rate = n_singletons_after / max(1, len(block_info))

    stats = {
        "n_sample":                    len(s1_sample),
        "blocker_recall":              recall,
        "blocker_precision":           precision,
        "f05":                         f05,
        "avg_candidates_per_s1":       recall_stats["avg_candidates_per_s1"],
        "singleton_rate_after_blocking": singleton_rate,
        "total_candidates":            total_cands,
        "recovered_gt":                total_tp,
        "total_gt_matches":            recall_stats["total_true_matches"],
        "elapsed_seconds":             elapsed,
    }

    logger.info(
        f"\n{'='*60}\n"
        f"VALIDATION RESULTS\n"
        f"  Blocker recall (unadjusted):  {recall:.4f}\n"
        f"  Blocker precision:            {precision:.4f}\n"
        f"  F0.5:                         {f05:.4f}\n"
        f"  Avg candidates per S1:        {recall_stats['avg_candidates_per_s1']:.1f}\n"
        f"  Singleton rate after block:   {singleton_rate:.3f}\n"
        f"  GT matches recovered:         {total_tp} / {recall_stats['total_true_matches']}\n"
        f"{'='*60}"
    )

    MIN_RECALL = 0.60
    if recall < MIN_RECALL:
        raise RuntimeError(
            f"Validation FAILED: blocker recall {recall:.4f} < {MIN_RECALL:.2f}. "
            "Tune top_k, max_candidates_per_s1, or blocking strategies before full run."
        )

    logger.info("Validation PASSED — proceeding to full dataset run.")
    return stats
