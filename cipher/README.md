# CIPHER

**Cognitive Information Pattern Hierarchy for Entropy-based Recognition**

AI-generated text detector that works without calling an LLM at inference time.

Target AUC: **≥ 0.95 ROC-AUC** on held-out test sets.

## Quick start

```bash
pip install -e ".[dev]"
cipher detect "Your text goes here..."
```

## How it works

CIPHER detects the *generative process* (human cognition vs. machine generation) rather than matching output against LLM distributions. It assembles **33 features** across 6 independent layers and feeds them into a gradient-boosted tree ensemble.

| Layer | Features | Method | LLM-free? |
|-------|----------|--------|-----------|
| 1 | l1–l7 (7) | Compression-theoretic (LZ76, NCD) | ✅ |
| 2 | c1–c8 (8) | Cognitive load fingerprint (Hurst exponent, epistemic markers) | ✅ |
| 3 | d1–d7 (7) | Discourse topology (co-reference graph, topic boundary divergence) | ✅ |
| 4 | m1 (1) | Contrastive manifold (DeBERTa-v3-base, optional) | ❌ |
| 5 | — | Gradient-boosted meta-ensemble + conformal calibration | ✅ |
| 6 | s1–s10 (10) | Lexical richness & stylometric features | ✅ |

### Layer 6 — Lexical Richness Features

| Feature | Name | Signal |
|---------|------|--------|
| s1 | Type-Token Ratio | AI text reuses vocabulary less |
| s2 | Hapax Legomena ratio | Rare word diversity |
| s3 | Yule's K | Higher K → less diverse (AI signal) |
| s4 | Bigram repetition | AI reuses phrase patterns |
| s5 | Passive voice ratio | AI overuses passive constructions |
| s6 | Punctuation entropy | AI punctuation is more uniform |
| s7 | Sentence-initial word entropy | AI sentences start with fewer unique words |
| s8 | Paragraph length CV | AI paragraphs have more uniform length |
| s9 | Lexical density | Content-word fraction |
| s10 | Mean word log-rank | AI uses more common words |

### Meta-learner (Layer 5)

A soft-voting ensemble combining:
- `HistGradientBoostingClassifier(max_iter=400, max_depth=6, learning_rate=0.05)` (weight 2)
- `LogisticRegression(C=1.0)` (weight 1)

Predictions are isotonically calibrated and wrapped in a conformal predictor for uncertainty estimation.

## Text chunking strategy

Training chunks are created with a **paragraph-boundary-aware** algorithm:

1. Split document on blank lines.
2. Accumulate paragraphs until word count ≥ 100.
3. Close the chunk at the next paragraph boundary.
4. Short trailing paragraphs are merged into the last chunk.

Each chunk carries a `doc_id` referencing its source document. During train/val splits, entire documents are assigned to one split — preventing cross-document data leakage.

## CLI commands

```bash
# Single text
cipher detect "Some text..."

# From file
cipher detect-file report.md

# Batch CSV (columns: id,text)
cipher batch input.csv output.jsonl

# Evaluate on labelled dataset (columns: text,label)
cipher evaluate test.csv

# Train meta-learner on labelled data
cipher train meta --data train.csv --output-dir models/
```

## Installation extras

```bash
pip install "cipher-detect[nlp]"      # + spaCy + sentence-transformers
pip install "cipher-detect[neural]"   # + torch + DeBERTa (Layer 4)
pip install "cipher-detect[explain]"  # + SHAP + crepes
pip install "cipher-detect[all]"      # everything
```
