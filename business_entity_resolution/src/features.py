"""
features.py — Rich pairwise feature engineering for entity pair classification.

Feature groups:
  1. NAME features (Jaro-Winkler, Levenshtein, RapidFuzz ratios, token overlap)
  2. ADDRESS features (same similarity metrics)
  3. NUMERIC features (house number, PIN, numeric Jaccard)
  4. TF-IDF cosine features (name char/word, addr char/word, fulltext)
  5. GLOBAL/ENTITY features (country match, missing indicators, source, blocks)
"""
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


# ─── String similarity helpers ────────────────────────────────────────────────

def safe_ratio(a: str, b: str, func) -> float:
    try:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return func(a, b) / 100.0
    except Exception:
        return 0.0


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of token sets."""
    ta = set(a.split()) if a else set()
    tb = set(b.split()) if b else set()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def char_overlap(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity of character n-grams."""
    def ngrams(s, k):
        return set(s[i:i+k] for i in range(len(s) - k + 1))
    a_ng = ngrams(a, n) if len(a) >= n else set(a)
    b_ng = ngrams(b, n) if len(b) >= n else set(b)
    if not a_ng and not b_ng:
        return 1.0
    if not a_ng or not b_ng:
        return 0.0
    return len(a_ng & b_ng) / len(a_ng | b_ng)


def numeric_jaccard(nums_a: list, nums_b: list) -> float:
    sa = set(nums_a) if nums_a else set()
    sb = set(nums_b) if nums_b else set()
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def jaro_winkler(a: str, b: str) -> float:
    try:
        return distance.JaroWinkler.normalized_similarity(a, b)
    except Exception:
        return 0.0


def levenshtein_similarity(a: str, b: str) -> float:
    try:
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        dist = distance.Levenshtein.distance(a, b)
        return 1.0 - dist / max_len
    except Exception:
        return 0.0


# ─── TF-IDF cosine feature computation ───────────────────────────────────────

class TfidfCosineScorer:
    """
    Precomputes TF-IDF cosine similarity features for candidate pairs.
    Fitted once on the full corpus (train + test combined for blocking; 
    for classifer features: fit on train only to avoid leakage).
    """

    def __init__(self):
        self.vectorizers = {}
        self.cand_matrices = {}

    def fit(self, cand_df: pd.DataFrame):
        """Fit all TF-IDF vectorizers on candidates."""
        configs = {
            "name_char": ("name_clean", "char_wb", (3, 5)),
            "name_word": ("name_clean", "word", (1, 2)),
            "addr_char": ("addr_expanded", "char_wb", (3, 5)),
            "addr_word": ("addr_expanded", "word", (1, 2)),
            "fulltext": ("combined_text", "word", (1, 2)),
        }
        for key, (col, analyzer, ngrams) in configs.items():
            corpus = cand_df[col].fillna("").tolist()
            vec = TfidfVectorizer(
                analyzer=analyzer, ngram_range=ngrams,
                max_features=100_000, sublinear_tf=True, min_df=1, dtype=np.float32
            )
            mat = normalize(vec.fit_transform(corpus), norm="l2", copy=False)
            self.vectorizers[key] = vec
            self.cand_matrices[key] = mat

        # Save index (entity_id) mapping
        self._cand_ids = cand_df["entity_id"].tolist()
        self._cand_id_to_idx = {cid: i for i, cid in enumerate(self._cand_ids)}

    def index_candidates(self, cand_df: pd.DataFrame):
        """
        Index candidate texts using already-fitted vectorizers without re-fitting.
        Strictly applies .transform() — NEVER re-fits vocabulary or IDF weights.
        Guarantees zero data leakage from test or validation candidate sets.
        """
        if not self.vectorizers:
            raise RuntimeError("Cannot index candidates before vectorizers are fitted on training data!")

        configs = {
            "name_char": "name_clean",
            "name_word": "name_clean",
            "addr_char": "addr_expanded",
            "addr_word": "addr_expanded",
            "fulltext": "combined_text",
        }
        for key, col in configs.items():
            if key in self.vectorizers:
                vec = self.vectorizers[key]
                corpus = cand_df[col].fillna("").tolist()
                mat = normalize(vec.transform(corpus), norm="l2", copy=False)
                self.cand_matrices[key] = mat

        self._cand_ids = cand_df["entity_id"].tolist()
        self._cand_id_to_idx = {cid: i for i, cid in enumerate(self._cand_ids)}

    def score_pair_batch(
        self,
        s1_texts: Dict[str, str],
        pair_list: List[Tuple[str, str]],  # [(s1_id, cand_id), ...]
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Compute TF-IDF cosine similarities for a batch of pairs efficiently.
        Batched sparse matrix ops: transform all unique S1 texts at once,
        gather unique candidate rows, then batch element-wise multiply+sum.
        """
        if not pair_list:
            return {}

        # Unique ordered S1 and candidate IDs
        unique_s1_ids = list(dict.fromkeys(s1 for s1, _ in pair_list))
        unique_cand_ids = list(dict.fromkeys(cid for _, cid in pair_list))
        s1_local = {sid: i for i, sid in enumerate(unique_s1_ids)}
        cand_local = {cid: i for i, cid in enumerate(unique_cand_ids)}

        # Candidate → global matrix row index (-1 if not in index)
        cand_global_idx = np.array(
            [self._cand_id_to_idx.get(cid, -1) for cid in unique_cand_ids],
            dtype=np.int32,
        )

        # Pair index arrays
        s1_pair_idx = np.array([s1_local[s1] for s1, _ in pair_list], dtype=np.int32)
        cand_pair_local_idx = np.array([cand_local[cid] for _, cid in pair_list], dtype=np.int32)

        scores: Dict = {}
        for key, vec in self.vectorizers.items():
            pair_sims = np.zeros(len(pair_list), dtype=np.float32)
            try:
                # 1. Transform unique S1 texts once
                s1_corpus = [s1_texts.get(sid, {}).get(key, "") for sid in unique_s1_ids]
                s1_mat = normalize(vec.transform(s1_corpus), norm="l2", copy=False)

                # 2. Gather unique candidate rows that exist in index
                valid_mask = cand_global_idx >= 0
                valid_local = np.where(valid_mask)[0]  # local unique-cand indices that are valid
                if len(valid_local) == 0:
                    pass  # all zeros
                else:
                    valid_global = cand_global_idx[valid_local]
                    cand_sub = self.cand_matrices[key][valid_global]  # sparse sub-matrix

                    # Mapping: local unique-cand idx -> sub-matrix row
                    local_to_sub = np.full(len(unique_cand_ids), -1, dtype=np.int32)
                    local_to_sub[valid_local] = np.arange(len(valid_local), dtype=np.int32)

                    sub_pair_idx = local_to_sub[cand_pair_local_idx]  # -1 for invalid pairs
                    valid_pairs = sub_pair_idx >= 0

                    if valid_pairs.any():
                        # Gather matching S1 and candidate rows
                        vp_s1 = s1_mat[s1_pair_idx[valid_pairs]]
                        vp_cand = cand_sub[sub_pair_idx[valid_pairs]]
                        # Element-wise multiply + row sum = dot product (both L2-normalized)
                        dots = np.asarray(vp_s1.multiply(vp_cand).sum(axis=1)).ravel()
                        pair_sims[valid_pairs] = dots.astype(np.float32)

            except Exception as e:
                logger.warning(f"TF-IDF scoring failed for {key}: {e}")

            # Store into scores dict
            for i, (s1_id, cand_id) in enumerate(pair_list):
                k = (s1_id, cand_id)
                if k not in scores:
                    scores[k] = {}
                scores[k][key] = float(pair_sims[i])

        return scores


# ─── Main feature computation ─────────────────────────────────────────────────

BLOCK_FEATURE_NAMES = [
    "blk_exact_name",
    "blk_exact_stripped",
    "blk_name_char_tfidf",
    "blk_name_word_tfidf",
    "blk_addr_char_tfidf",
    "blk_addr_word_tfidf",
    "blk_fulltext_tfidf",
    "blk_numeric_addr",
    "blk_rare_token",
]

BLOCK_NAME_MAP = {
    "exact_name": "blk_exact_name",
    "exact_stripped": "blk_exact_stripped",
    "name_char_tfidf": "blk_name_char_tfidf",
    "name_word_tfidf": "blk_name_word_tfidf",
    "addr_char_tfidf": "blk_addr_char_tfidf",
    "addr_word_tfidf": "blk_addr_word_tfidf",
    "fulltext_tfidf": "blk_fulltext_tfidf",
    "numeric_addr": "blk_numeric_addr",
    "rare_token": "blk_rare_token",
}


def compute_pair_features(
    s1_row: pd.Series,
    cand_row: pd.Series,
    block_flags: Dict[str, bool],
    tfidf_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute all features for a single (S1, candidate) pair.
    """
    feats = {}

    s1_name_c = str(s1_row.get("name_clean", "") or "")
    s1_name_s = str(s1_row.get("name_stripped", "") or "")
    cand_name_c = str(cand_row.get("name_clean", "") or "")
    cand_name_s = str(cand_row.get("name_stripped", "") or "")

    s1_addr_e = str(s1_row.get("addr_expanded", "") or "")
    cand_addr_e = str(cand_row.get("addr_expanded", "") or "")

    # ── NAME features ──────────────────────────────────────────────────────
    feats["name_jaro_winkler"] = jaro_winkler(s1_name_c, cand_name_c)
    feats["name_levenshtein"] = levenshtein_similarity(s1_name_c, cand_name_c)
    feats["name_fuzz_ratio"] = safe_ratio(s1_name_c, cand_name_c, fuzz.ratio)
    feats["name_fuzz_partial"] = safe_ratio(s1_name_c, cand_name_c, fuzz.partial_ratio)
    feats["name_fuzz_token_sort"] = safe_ratio(s1_name_c, cand_name_c, fuzz.token_sort_ratio)
    feats["name_fuzz_token_set"] = safe_ratio(s1_name_c, cand_name_c, fuzz.token_set_ratio)

    feats["name_stripped_jw"] = jaro_winkler(s1_name_s, cand_name_s)
    feats["name_stripped_fuzz"] = safe_ratio(s1_name_s, cand_name_s, fuzz.token_set_ratio)
    feats["name_exact_clean"] = float(s1_name_c == cand_name_c and s1_name_c != "")
    feats["name_exact_stripped"] = float(s1_name_s == cand_name_s and s1_name_s != "")

    s1_ntok = len(s1_name_c.split())
    cand_ntok = len(cand_name_c.split())
    feats["name_len_diff"] = abs(s1_ntok - cand_ntok)
    feats["name_len_ratio"] = min(s1_ntok, cand_ntok) / max(1, max(s1_ntok, cand_ntok))
    feats["name_token_overlap"] = token_overlap(s1_name_c, cand_name_c)
    feats["name_char_overlap"] = char_overlap(s1_name_c, cand_name_c)

    s1_char_len = len(s1_name_c)
    cand_char_len = len(cand_name_c)
    feats["name_charlen_diff"] = abs(s1_char_len - cand_char_len)
    feats["name_charlen_ratio"] = min(s1_char_len, cand_char_len) / max(1, max(s1_char_len, cand_char_len))

    # ── ADDRESS features ───────────────────────────────────────────────────
    feats["addr_jaro_winkler"] = jaro_winkler(s1_addr_e, cand_addr_e)
    feats["addr_levenshtein"] = levenshtein_similarity(s1_addr_e, cand_addr_e)
    feats["addr_fuzz_ratio"] = safe_ratio(s1_addr_e, cand_addr_e, fuzz.ratio)
    feats["addr_fuzz_partial"] = safe_ratio(s1_addr_e, cand_addr_e, fuzz.partial_ratio)
    feats["addr_fuzz_token_sort"] = safe_ratio(s1_addr_e, cand_addr_e, fuzz.token_sort_ratio)
    feats["addr_fuzz_token_set"] = safe_ratio(s1_addr_e, cand_addr_e, fuzz.token_set_ratio)
    feats["addr_token_overlap"] = token_overlap(s1_addr_e, cand_addr_e)
    feats["addr_char_overlap"] = char_overlap(s1_addr_e, cand_addr_e)

    s1_atok = len(s1_addr_e.split())
    cand_atok = len(cand_addr_e.split())
    feats["addr_len_diff"] = abs(s1_atok - cand_atok)
    feats["addr_len_ratio"] = min(s1_atok, cand_atok) / max(1, max(s1_atok, cand_atok))

    # ── NUMERIC features ───────────────────────────────────────────────────
    s1_nums = list(s1_row.get("addr_numerics") or [])
    cand_nums = list(cand_row.get("addr_numerics") or [])

    feats["numeric_jaccard"] = numeric_jaccard(s1_nums, cand_nums)
    feats["n_common_numerics"] = len(set(s1_nums) & set(cand_nums))
    feats["s1_has_nums"] = float(len(s1_nums) > 0)
    feats["cand_has_nums"] = float(len(cand_nums) > 0)

    # House number: first numeric token (often 3-6 digits)
    def first_num(nums):
        for n in nums:
            if 2 <= len(n) <= 6:
                return n
        return None

    s1_hn = first_num(s1_nums)
    cand_hn = first_num(cand_nums)
    feats["house_num_match"] = float(s1_hn is not None and s1_hn == cand_hn)

    # PIN/postal code: numeric token of 4-6 digits
    def find_pin(nums):
        for n in nums:
            if 4 <= len(n) <= 6:
                return n
        return None

    s1_pin = find_pin(s1_nums)
    cand_pin = find_pin(cand_nums)
    feats["pin_match"] = float(s1_pin is not None and s1_pin == cand_pin)

    # ── TF-IDF cosine features ─────────────────────────────────────────────
    if tfidf_scores:
        feats["tfidf_name_char"] = tfidf_scores.get("name_char", 0.0)
        feats["tfidf_name_word"] = tfidf_scores.get("name_word", 0.0)
        feats["tfidf_addr_char"] = tfidf_scores.get("addr_char", 0.0)
        feats["tfidf_addr_word"] = tfidf_scores.get("addr_word", 0.0)
        feats["tfidf_fulltext"] = tfidf_scores.get("fulltext", 0.0)
    else:
        for k in ["tfidf_name_char", "tfidf_name_word", "tfidf_addr_char",
                  "tfidf_addr_word", "tfidf_fulltext"]:
            feats[k] = 0.0

    # ── GLOBAL/ENTITY features ─────────────────────────────────────────────
    s1_country = str(s1_row.get("country_norm", "") or "")
    cand_country = str(cand_row.get("country_norm", "") or "")
    feats["country_match"] = float(s1_country == cand_country)
    feats["s1_has_name"] = float(s1_row.get("has_name", 1))
    feats["s1_has_address"] = float(s1_row.get("has_address", 1))
    feats["cand_has_name"] = float(cand_row.get("has_name", 1))
    feats["cand_has_address"] = float(cand_row.get("has_address", 1))

    # Candidate source (S2 vs S3)
    cand_id = str(cand_row.get("entity_id", ""))
    feats["cand_is_s2"] = float(cand_id.startswith("S2-"))
    feats["cand_is_s3"] = float(cand_id.startswith("S3-"))

    # Blocking indicators
    for blk_feat in BLOCK_FEATURE_NAMES:
        feats[blk_feat] = float(block_flags.get(blk_feat, False))

    return feats


def build_feature_dataframe(
    pair_list: List[Tuple[str, str, int]],  # (s1_id, cand_id, label)
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    block_info: Dict[str, Dict],  # from MultiBlocker.generate_candidates()
    tfidf_scorer: Optional[TfidfCosineScorer] = None,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """
    Compute features for all (S1, candidate) pairs.
    Processes in chunks to avoid OOM.
    """
    needed_s1 = {p[0] for p in pair_list}
    needed_cand = {p[1] for p in pair_list}
    s1_lookup = s1_df[s1_df["entity_id"].isin(needed_s1)].set_index("entity_id").to_dict("index")
    cand_lookup = cand_df[cand_df["entity_id"].isin(needed_cand)].set_index("entity_id").to_dict("index")

    # Precompute TF-IDF text lookup for S1 entities
    tfidf_text_keys = {
        "name_char": "name_clean",
        "name_word": "name_clean",
        "addr_char": "addr_expanded",
        "addr_word": "addr_expanded",
        "fulltext": "combined_text",
    }
    s1_tfidf_texts = {}
    if tfidf_scorer:
        for sid in {p[0] for p in pair_list}:
            row = s1_lookup.get(sid, {})
            s1_tfidf_texts[sid] = {k: str(row.get(col, "") or "") for k, col in tfidf_text_keys.items()}

    all_rows = []
    total = len(pair_list)
    logger.info(f"Computing features for {total} pairs...")

    for chunk_start in range(0, total, chunk_size):
        chunk = pair_list[chunk_start: chunk_start + chunk_size]

        # TF-IDF scores for this chunk
        tfidf_chunk = {}
        if tfidf_scorer and chunk:
            pairs_only = [(s1, cand) for s1, cand, _ in chunk]
            try:
                tfidf_chunk = tfidf_scorer.score_pair_batch(s1_tfidf_texts, pairs_only)
            except Exception as e:
                logger.warning(f"TF-IDF batch scoring failed: {e}")

        for s1_id, cand_id, label in chunk:
            s1_row = s1_lookup.get(s1_id, {})
            cand_row = cand_lookup.get(cand_id, {})
            if not s1_row or not cand_row:
                continue

            # Block flags
            block_info_for_s1 = block_info.get(s1_id, {})
            blocks = block_info_for_s1.get("blocks", {})
            block_flags = {
                BLOCK_NAME_MAP.get(bname, bname): cand_id in bset
                for bname, bset in blocks.items()
            }

            tfidf_pair = tfidf_chunk.get((s1_id, cand_id), {})

            feats = compute_pair_features(
                pd.Series(s1_row),
                pd.Series(cand_row),
                block_flags,
                tfidf_pair,
            )
            feats["s1_id"] = s1_id
            feats["cand_id"] = cand_id
            feats["label"] = label
            all_rows.append(feats)

        if (chunk_start // chunk_size) % 5 == 0:
            logger.info(f"  Processed {min(chunk_start + chunk_size, total)}/{total} pairs")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info(f"Feature matrix shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of feature column names (excluding meta columns)."""
    meta_cols = {"s1_id", "cand_id", "label"}
    return [c for c in df.columns if c not in meta_cols]
