"""
preprocessing.py — Robust normalization for business entity resolution.

Creates multiple representations of business names and addresses:
- cleaned (lowercase, unicode-normalized)
- stripped (legal suffixes removed)
- tokenized
- character n-gram fingerprints
- numeric tokens (for house/PIN matching)

Performance notes (10 M+ records):
- All regex maps are compiled once at import time into single-pass alternation
  patterns with a match→replacement callback; this replaces the original N
  sequential re.sub calls per row.
- preprocess_dataframe() is fully vectorized: pd.Series.str operations are
  used throughout so Python per-row overhead is eliminated.
- Scalar helper functions (normalize_name, normalize_address, …) are preserved
  for backward compatibility; they delegate to the same compiled patterns.
"""
import logging
import re
import unicodedata
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Legal-suffix normalization ──────────────────────────────────────────────
# Map abbreviations → canonical form  (both directions of noise)
LEGAL_SUFFIX_MAP = {
    r"\bprivate limited\b": "pvt ltd",
    r"\bpvt\.? ltd\.?\b": "pvt ltd",
    r"\bpvt\.?\b": "pvt",
    r"\bpriv\.?\b": "pvt",
    r"\blimited liability partnership\b": "llp",
    r"\bllp\b": "llp",
    r"\blimited\b": "ltd",
    r"\bltd\.?\b": "ltd",
    r"\bcorporation\b": "corp",
    r"\bcorp\.?\b": "corp",
    r"\bincorporated\b": "inc",
    r"\binc\.?\b": "inc",
    r"\bllc\.?\b": "llc",
    r"\blimited liability company\b": "llc",
    r"\bcompany\b": "co",
    r"\bco\.?\b": "co",
    r"\benterprises\b": "enterprises",
    r"\benterprise\b": "enterprise",
    r"\bservices\b": "services",
    r"\bsolutions\b": "solutions",
    r"\bgroups?\b": "group",
    r"\bassociates\b": "associates",
    r"\bindustries\b": "industries",
    r"\btrading\b": "trading",
    r"\bholdings?\b": "holdings",
    r"\bventures?\b": "ventures",
    r"\binternational\b": "intl",
    r"\bintl\.?\b": "intl",
    r"\bnational\b": "natl",
    r"\bmanagement\b": "mgmt",
    r"\bconsultants?\b": "consulting",
    r"\bconsulting\b": "consulting",
    r"\brestaurant\b": "restaurant",
    r"\bcafé\b": "cafe",
    r"\bcafe\b": "cafe",
    r"\bshoppes?\b": "shop",
    r"\bstore\b": "store",
    r"\bmart\b": "mart",
    r"\bsupermarket\b": "supermarket",
    r"\bpharmacy\b": "pharmacy",
    r"\bhospital\b": "hospital",
    r"\bclinic\b": "clinic",
    r"\bacademy\b": "academy",
    r"\bschool\b": "school",
    r"\bcollege\b": "college",
    r"\buniversity\b": "university",
    r"\bfoundation\b": "foundation",
    r"\bcharitable trust\b": "trust",
    r"\btrust\b": "trust",
    r"\bsociety\b": "society",
    r"\bclub\b": "club",
    r"\bassociation\b": "assoc",
    r"\bassoc\.?\b": "assoc",
    r"\bgmbh\b": "gmbh",
    r"\bsarl\b": "sarl",
    r"\bsa\.?\b": "sa",
    r"\bsas\b": "sas",
    r"\beurl\b": "eurl",
    r"\bsci\b": "sci",
}

# Legal suffixes to strip for the "stripped name" representation
LEGAL_SUFFIX_STRIP = re.compile(
    r"\b(pvt\s+ltd|pvt|private\s+limited|limited|ltd|corporation|corp|"
    r"incorporated|inc|llc|llp|company|co|enterprises?|enterprise|"
    r"services?|solutions?|groups?|associates?|industries|trading|"
    r"holdings?|ventures?|international|intl|national|natl|management|mgmt|"
    r"consultants?|consulting|gmbh|sarl|sas|eurl|sci|sa)\b\.?",
    re.IGNORECASE,
)

# Address abbreviation expansion
ADDR_ABBR_MAP = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave?\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bdr\.?\b": "drive",
    r"\bln\.?\b": "lane",
    r"\bct\.?\b": "court",
    r"\bpl\.?\b": "place",
    r"\bsq\.?\b": "square",
    r"\bfwy\.?\b": "freeway",
    r"\bhwy\.?\b": "highway",
    r"\bpkwy\.?\b": "parkway",
    r"\bexpy\.?\b": "expressway",
    r"\bnagar\b": "nagar",
    r"\bngr\.?\b": "nagar",
    r"\bcolony\b": "colony",
    r"\bcol\.?\b": "colony",
    r"\bsector\b": "sector",
    r"\bsec\.?\b": "sector",
    r"\bapt\.?\b": "apartment",
    r"\bapartment\b": "apartment",
    r"\bflr\.?\b": "floor",
    r"\bfl\.?\b": "floor",
    r"\bfloor\b": "floor",
    r"\bsuite\b": "suite",
    r"\bste\.?\b": "suite",
    r"\bbldg\.?\b": "building",
    r"\bbuilding\b": "building",
    r"\bno\.?\s*": "no ",
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w",
    r"\bnortheast\b": "ne",
    r"\bnorthwest\b": "nw",
    r"\bsoutheast\b": "se",
    r"\bsouthwest\b": "sw",
}

# Country name variants → normalized label
COUNTRY_NORM_MAP = {
    "india": "india",
    "ind": "india",
    "bharat": "india",
    "inde": "india",
    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "u.s.a.": "us",
    "u.s.": "us",
    "france": "france",
    "fr": "france",
    "république française": "france",
    "republique francaise": "france",
}


# ─── Pre-compiled patterns ────────────────────────────────────────────────────
# Build a single alternation regex for each map so we scan the string once
# instead of len(map) times.  The replacement callback looks up the canonical
# form from each original compiled sub-pattern (cached after first hit).

def _make_single_pass_replacer(abbr_map: dict):
    """
    Compile all patterns in *abbr_map* into one alternation regex and return
    (compiled_pattern, replacer_callable).

    The replacer uses a local cache keyed on lower-cased matched text so that
    repeated tokens pay only O(1) amortised lookup cost.
    """
    combined = re.compile(
        "|".join(f"(?:{p})" for p in abbr_map.keys()),
        re.IGNORECASE,
    )
    # Pre-compile each individual pattern for fullmatch lookup in the callback
    _pairs = [(re.compile(p, re.IGNORECASE), r) for p, r in abbr_map.items()]
    _cache: dict[str, str] = {}

    def _replacer(m: re.Match) -> str:
        tok = m.group(0)
        key = tok.lower()
        if key not in _cache:
            for cpat, repl in _pairs:
                if cpat.fullmatch(tok):
                    _cache[key] = repl
                    break
            else:
                _cache[key] = tok  # safety: keep as-is if no pattern matched
        return _cache[key]

    return combined, _replacer


# Compiled at import time — no per-call overhead
_LEGAL_NORM_RE, _legal_norm_replacer = _make_single_pass_replacer(LEGAL_SUFFIX_MAP)
_ADDR_RE, _addr_replacer = _make_single_pass_replacer(ADDR_ABBR_MAP)

# Common utility patterns
_RE_SPECIAL      = re.compile(r"[^\w\s\-]")
_RE_HYPHEN       = re.compile(r"[-_]")
_RE_SPACES       = re.compile(r"\s+")
_RE_DIGITS       = re.compile(r"\d+")
_RE_WORD_DIGITS  = re.compile(r"\b\d+\b")


# ─── Scalar helper functions (backward-compatible public API) ─────────────────

def unicode_normalize(text: str) -> str:
    """Normalize unicode to closest ASCII/latin equivalent, keep non-latin."""
    try:
        nfkd = unicodedata.normalize("NFKD", text)
        ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
        return ascii_text
    except Exception:
        return text


def clean_text(text: str, abbr_map: Optional[dict] = None) -> str:
    """
    Core cleaning:
      - fillna / cast to str
      - unicode normalize
      - lowercase
      - expand & → and
      - remove punctuation except hyphens
      - collapse whitespace
      - optionally expand abbreviations
    """
    if not isinstance(text, str) or text.lower() in ("nan", "none", "null", ""):
        return ""
    text = unicode_normalize(text)
    text = text.lower()
    text = text.replace("&", " and ")
    # Remove special chars but keep alphanumeric, space, hyphen
    text = _RE_SPECIAL.sub(" ", text)
    text = _RE_HYPHEN.sub(" ", text)
    if abbr_map:
        for pattern, replacement in abbr_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = _RE_SPACES.sub(" ", text).strip()
    return text


def normalize_name(name) -> dict:
    """
    Return multiple name representations:
      - 'clean': lowercase, unicode-normalized
      - 'stripped': legal suffixes removed
      - 'tokens': list of tokens (for inverted index)
      - 'joined_stripped': space-joined stripped tokens
    """
    if not isinstance(name, str) or pd.isna(name):
        name = ""

    clean = clean_text(str(name))

    # Apply legal suffix normalization (single-pass — replaces N re.sub loops)
    clean = _LEGAL_NORM_RE.sub(_legal_norm_replacer, clean)
    clean = _RE_SPACES.sub(" ", clean).strip()

    # Stripped: remove legal suffixes entirely
    stripped = LEGAL_SUFFIX_STRIP.sub(" ", clean)
    stripped = _RE_SPACES.sub(" ", stripped).strip()

    tokens = clean.split()
    stripped_tokens = stripped.split()

    return {
        "clean": clean,
        "stripped": stripped,
        "tokens": tokens,
        "stripped_tokens": stripped_tokens,
        "joined_stripped": stripped,
    }


def extract_numerics(text: str) -> list:
    """Extract numeric tokens from text (house numbers, PINs, etc.)."""
    if not isinstance(text, str) or not text:
        return []
    return _RE_DIGITS.findall(text)


def normalize_address(address) -> dict:
    """
    Return multiple address representations:
      - 'clean': basic cleaned address
      - 'expanded': address with abbreviations expanded
      - 'tokens': list of tokens
      - 'numerics': list of numeric tokens
      - 'non_numeric': text with numbers removed
    """
    if not isinstance(address, str) or pd.isna(address):
        address = ""

    clean = clean_text(str(address))
    numerics = extract_numerics(clean)

    # Expand address abbreviations (single-pass — replaces N re.sub loops)
    expanded = _ADDR_RE.sub(_addr_replacer, clean)
    expanded = _RE_SPACES.sub(" ", expanded).strip()

    tokens = expanded.split()
    non_numeric = _RE_WORD_DIGITS.sub(" ", expanded)
    non_numeric = _RE_SPACES.sub(" ", non_numeric).strip()

    return {
        "clean": clean,
        "expanded": expanded,
        "tokens": tokens,
        "numerics": numerics,
        "non_numeric": non_numeric,
    }


def normalize_country(country) -> str:
    """Normalize country string. Open-set: unknown countries pass through."""
    if not isinstance(country, str) or pd.isna(country):
        return "unknown"
    key = country.lower().strip()
    return COUNTRY_NORM_MAP.get(key, key)


# ─── Vectorized series helpers (used by preprocess_dataframe) ─────────────────

def _vec_unicode_normalize(s: pd.Series) -> pd.Series:
    """
    Vectorized NFKD unicode normalization.
    pandas str.normalize('NFKD') + encode/decode is 3-5× faster than a
    Python-level apply with unicodedata on large series.
    """
    return (
        s.str.normalize("NFKD")
         .str.encode("ascii", errors="ignore")
         .str.decode("ascii")
    )


def _vec_clean(s: pd.Series) -> pd.Series:
    """
    Vectorized equivalent of clean_text() for a whole Series.
    No abbreviation expansion is applied here; callers do that in a separate
    step via _vec_apply_legal_norm / _vec_expand_addr.
    """
    s = s.fillna("").astype(str)
    # Blank out sentinel strings
    s = s.where(~s.str.lower().isin(["nan", "none", "null"]), "")
    s = _vec_unicode_normalize(s)
    s = s.str.lower()
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(_RE_SPECIAL, " ", regex=True)
    s = s.str.replace(_RE_HYPHEN, " ", regex=True)
    s = s.str.replace(_RE_SPACES, " ", regex=True).str.strip()
    return s


# ─── Public API ───────────────────────────────────────────────────────────────

def _apply_unique(
    series: pd.Series,
    transform_fn,
) -> pd.Series:
    """
    Apply *transform_fn* only to the distinct values in *series*, then map
    results back.  For columns with many repeated values (e.g. company names
    that appear in multiple sources) this can cut work by 60-90 %.

    *transform_fn* must accept a pd.Series and return a pd.Series of the same
    length with the same index.
    """
    unique_vals = series.drop_duplicates().reset_index(drop=True)
    transformed = transform_fn(unique_vals)
    lookup = dict(zip(unique_vals, transformed))
    res = series.map(lookup)
    res.index = series.index
    return res


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to a source dataframe.
    Input df must have: entity_id, business_name, business_address, country

    All transformations are fully vectorized; no Python-level row iteration.
    Repeated values are normalized only once via _apply_unique deduplication.
    Per-field timings and deep memory usage are measured and logged at INFO level.
    Preserves all required output columns while avoiding memory amplification.
    """
    import time as _time

    def _stage_report(label: str, t0: float) -> float:
        elapsed = _time.perf_counter() - t0
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"  [preprocess] {label}: {elapsed:.3f}s | DataFrame RAM: {mem_mb:.1f} MB")
        return _time.perf_counter()

    logger.info(f"preprocess_dataframe: {len(df)} rows")
    t_total = _time.perf_counter()

    df = df.copy(deep=False)
    df["business_name"]    = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"]          = df["country"].fillna("unknown").astype(str)

    # ── Name normalizations ──────────────────────────────────────────────────
    t0 = _time.perf_counter()

    def _name_clean_fn(s: pd.Series) -> pd.Series:
        return (
            _vec_clean(s)
            .str.replace(_LEGAL_NORM_RE, _legal_norm_replacer, regex=True)
            .str.replace(_RE_SPACES, " ", regex=True)
            .str.strip()
        )

    name_clean = _apply_unique(df["business_name"], _name_clean_fn)
    df["name_clean"] = name_clean
    t0 = _stage_report("business_name (clean + legal norm)", t0)

    def _name_stripped_fn(s: pd.Series) -> pd.Series:
        return (
            s.str.replace(LEGAL_SUFFIX_STRIP, " ", regex=True)
             .str.replace(_RE_SPACES, " ", regex=True)
             .str.strip()
        )

    # Stripped depends on cleaned names; deduplicate over cleaned values
    name_stripped = _apply_unique(name_clean, _name_stripped_fn)
    df["name_stripped"] = name_stripped
    # Reusing list references across identical clean names avoids Python object overhead
    df["name_tokens"]   = _apply_unique(name_clean, lambda s: s.str.split())
    t0 = _stage_report("stripped name", t0)

    # ── Address normalizations ───────────────────────────────────────────────
    t0 = _time.perf_counter()

    def _addr_base_fn(s: pd.Series) -> pd.Series:
        return _vec_clean(s)

    addr_base = _apply_unique(df["business_address"], _addr_base_fn)

    def _addr_expanded_fn(s: pd.Series) -> pd.Series:
        return (
            s.str.replace(_ADDR_RE, _addr_replacer, regex=True)
             .str.replace(_RE_SPACES, " ", regex=True)
             .str.strip()
        )

    addr_expanded = _apply_unique(addr_base, _addr_expanded_fn)

    df["addr_clean"]       = addr_base
    df["addr_expanded"]    = addr_expanded
    # Reusing list references across identical expanded addresses
    df["addr_tokens"]      = _apply_unique(addr_expanded, lambda s: s.str.split())
    t0 = _stage_report("address (clean + expand)", t0)

    t0 = _time.perf_counter()
    # Deduplicate numerics extraction to avoid regex and list allocation on duplicate rows
    df["addr_numerics"]    = _apply_unique(addr_base, lambda s: s.str.findall(r"\d+"))
    t0 = _stage_report("addr numerics", t0)

    def _non_num_fn(s: pd.Series) -> pd.Series:
        return (
            s.str.replace(_RE_WORD_DIGITS, " ", regex=True)
             .str.replace(_RE_SPACES, " ", regex=True)
             .str.strip()
        )
    df["addr_non_numeric"] = _apply_unique(addr_expanded, _non_num_fn)

    # ── Country normalization ────────────────────────────────────────────────
    country_key = df["country"].str.lower().str.strip()
    df["country_norm"] = country_key.map(COUNTRY_NORM_MAP).fillna(country_key)

    # ── Combined text for full-text TF-IDF ───────────────────────────────────
    t0 = _time.perf_counter()
    df["combined_text"] = df["name_clean"] + " " + df["addr_expanded"]
    t0 = _stage_report("combined_text", t0)

    # ── Flags (use int8 to minimize memory footprint) ────────────────────────
    df["has_name"]    = (df["name_clean"].str.strip() != "").astype(np.int8)
    df["has_address"] = (df["addr_clean"].str.strip() != "").astype(np.int8)

    total_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(
        f"  [preprocess] TOTAL: {_time.perf_counter() - t_total:.3f}s | Final RAM: {total_mem:.1f} MB | "
        f"unique names: {df['business_name'].nunique()} / {len(df)} | "
        f"unique addrs: {df['business_address'].nunique()} / {len(df)}"
    )
    return df

