# Business Entity Resolution — Pipeline README

## Overview

Multi-strategy blocking + LightGBM pairwise classifier for large-scale business entity resolution.

## Directory Structure

```
business_entity_resolution/
├── src/
│   ├── main.py           — End-to-end pipeline entry point
│   ├── preprocessing.py  — Robust normalization (name, address, country)
│   ├── blocking.py       — 9-strategy multi-blocker + recall evaluation
│   ├── features.py       — Rich pairwise feature engineering (~40 features)
│   ├── model.py          — LightGBM training (GroupKFold, hard negatives)
│   ├── evaluation.py     — F0.5 computation, threshold optimization
│   └── inference.py      — Test inference + TSV output generation
├── output/               — Submission files (generated)
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── model_artifacts/      — Saved model (generated)
│   └── pipeline_artifacts.pkl
├── requirements.txt
└── README.md
```

## Setup

```bash
# From the student_resource/ directory:
python3 -m venv .venv
.venv/bin/pip install -r business_entity_resolution/requirements.txt
```

## Running the Pipeline

### Full pipeline (recommended — runs train + test inference):
```bash
cd /path/to/student_resource
.venv/bin/python business_entity_resolution/src/main.py
```

### With custom settings:
```bash
.venv/bin/python business_entity_resolution/src/main.py \
    --n-train-sample 300000 \
    --top-k 50 \
    --n-folds 5 \
    --output-dir output \
    --data-dir dataset
```

### Training only (saves artifacts):
```bash
.venv/bin/python business_entity_resolution/src/main.py --train-only
```

### Test inference only (requires saved artifacts):
```bash
.venv/bin/python business_entity_resolution/src/main.py --test-only
```

## Pipeline Architecture

```
RAW DATA
  ↓
ROBUST NORMALIZATION (preprocessing.py)
  - name_clean, name_stripped, addr_expanded, addr_numerics, country_norm
  ↓
MULTI-BLOCKING (blocking.py) — 9 strategies unioned per S1 entity
  A. Exact normalized name
  B. Exact stripped name
  C. Name char TF-IDF (3-5 n-grams)
  D. Name word TF-IDF (1-2 grams)
  E. Address char TF-IDF
  F. Address word TF-IDF
  G. Full-text TF-IDF (name + address)
  H. Numeric/address token blocking (house numbers, PINs)
  I. Rare-token inverted index
  — Country-restricted with global fallback
  ↓
CANDIDATE RECALL EVALUATION (>= target recall)
  ↓
PAIRWISE FEATURES (features.py) — ~40 features
  - Name: Jaro-Winkler, Levenshtein, 4x RapidFuzz ratios, token/char overlap
  - Address: same 8 metrics
  - Numeric: Jaccard, house-number match, PIN match
  - TF-IDF cosine: 5 variants (name/addr × char/word + fulltext)
  - Global: country match, missing fields, source (S2/S3), 9 blocking indicators
  ↓
HARD-NEGATIVE SAMPLING (model.py)
  - 60% high-similarity negatives + 40% random (ratio 5:1 neg:pos)
  ↓
LIGHTGBM CLASSIFIER (model.py)
  - GroupKFold (groups = S1 entity_id, no leakage)
  - Early stopping per fold
  - Ensemble fold predictions
  ↓
F0.5 THRESHOLD OPTIMIZATION on OOF predictions
  - Sweep 0.30→0.99, step 0.01
  - Entity-level F0.5 including singletons
  ↓
TEST INFERENCE → output/matching_results.tsv + candidate_pairs.tsv
```

## Validate Output

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Key Design Decisions

- **Precision-heavy metric**: threshold optimized for F0.5 (β=0.5), not accuracy
- **Scale**: 2.2M S1 × 10M S2+S3. Handled via country-batched TF-IDF blocking + sampling
- **Hard negatives**: near-miss pairs (similar name/address but different business) improve precision
- **Open-set countries**: no hard-coded country list; France and other unseen countries work identically
- **Singletons**: correctly predicting "no match" scores 1.0; false merges score 0.0

## Hardware Requirements

- RAM: ~32GB recommended (64GB for `--n-train-sample 0`)
- CPU: 8+ cores recommended
- Disk: ~5GB for model artifacts
- Runtime: ~2-4 hours for default settings on 200K sample
