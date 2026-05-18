# User Guide

> Cross-references: [Architecture](architecture.md) · [API Reference](api_reference.md) · [Dataset Documentation](datasets.md) · [Index](index.md)

---

## Table of Contents

1. [Installation](#installation)
2. [Command-Line Interface](#command-line-interface)
   - [detect](#detect)
   - [detect-file](#detect-file)
   - [batch](#batch)
   - [evaluate](#evaluate)
   - [train meta](#train-meta)
   - [train contrastive](#train-contrastive)
3. [Python API](#python-api)
4. [Interpreting Results](#interpreting-results)
5. [Training Your Own Model](#training-your-own-model)
6. [Security](#security)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Standard (CPU-only)

```bash
pip install cipher-detect
python -m spacy download en_core_web_sm
```

### With Neural Layer 4 (GPU recommended)

```bash
pip install "cipher-detect[neural]"
# Installs torch and transformers; downloads DeBERTa-v3-base (~450 MB)
```

### From Source

```bash
git clone https://github.com/your-org/AITextDetection
cd AITextDetection/cipher
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

**Python version:** 3.10 or later.  
**Required packages:** `numpy`, `scipy`, `scikit-learn`, `spacy`, `typer`, `rich`, `networkx`, `zstandard`, `requests`.

---

## Command-Line Interface

All commands are available under the `cipher` entrypoint.

```
cipher --help
```

### detect

Analyse a text string passed as an argument.

```bash
cipher detect "The model generated highly coherent text..."
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--model-dir`, `-m` | None | Path to a trained model directory. Omit for heuristic mode. |
| `--spacy-model` | `en_core_web_sm` | spaCy model name for discourse features. |
| `--encoder/--no-encoder` | `--encoder` | Enable/disable sentence-transformers for coherence features. |
| `--coverage` | `0.95` | Conformal prediction coverage level (0.80–0.99). |
| `--sentences/--no-sentences` | `--sentences` | Show per-sentence scores. |
| `--json` | False | Output raw JSON instead of formatted display. |

**Read from stdin:**

```bash
cat article.txt | cipher detect -
```

**JSON output:**

```bash
cipher detect --json "Some text..." | jq .document_score
```

---

### detect-file

Analyse a `.txt` or `.md` file.

```bash
cipher detect-file report.txt
cipher detect-file essay.md --model-dir models/ --json
```

Returns non-zero exit code if the file is not found.

---

### batch

Analyse multiple documents from a CSV file. The CSV must have a `text` column; an optional `label` column (1=AI, 0=Human) enables accuracy reporting.

```bash
cipher batch documents.csv --output results.csv
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--output` | None | Path to write results CSV. Prints to stdout if omitted. |
| `--model-dir`, `-m` | None | Trained model directory. |
| `--threshold` | `0.5` | Score threshold for AI/Human classification. |

---

### evaluate

Evaluate a trained model on a labelled CSV file. Reports AUROC, accuracy, and per-class precision/recall.

```bash
cipher evaluate data/eval.csv --model-dir models/
```

---

### train meta

Train the Layer 5 meta-learner from a labelled CSV. The CSV must contain columns `text` (document text) and `label` (1=AI, 0=Human).

```bash
cipher train meta \
  --data data/train.csv \
  --output-dir models/ \
  --spacy-model en_core_web_sm \
  --calibration-split 0.2 \
  --coverage 0.95
```

This command:
1. Extracts all 22 features (Layers 1–3) for each document.
2. Trains a logistic regression meta-learner on 80% of the data.
3. Fits an isotonic calibration curve on 20% held-out calibration data.
4. Fits conformal prediction thresholds at the requested coverage.
5. Saves `meta_learner.pkl`, `isotonic.pkl`, and `conformal.npy` to `--output-dir`.

Training 10 000 documents takes approximately 30–90 minutes depending on hardware (discourse feature extraction is the bottleneck).

See [Training Your Own Model](#training-your-own-model) for a full workflow.

---

### train contrastive

Train the optional DeBERTa-v3-base contrastive encoder (Layer 4). **Requires `cipher-detect[neural]`.**

```bash
cipher train contrastive \
  --data data/train.csv \
  --output-dir models/ \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-5
```

This is compute-intensive (GPU recommended). It saves `contrastive_encoder.pt` and `manifold_scorer.pkl` which are automatically loaded by `CIPHERDetector` when present in `--model-dir`.

---

## Python API

### Basic Usage

```python
from cipher import CIPHERDetector

# Heuristic mode (no trained model required)
detector = CIPHERDetector()
result = detector.detect("Your text here.")

print(result.label)           # "AI", "Human", or "Uncertain"
print(result.document_score)  # float in [0, 1]
print(result.confidence_interval)  # (lo, hi) at 95% coverage
```

### With a Trained Model

```python
from pathlib import Path
from cipher import CIPHERDetector

detector = CIPHERDetector(model_dir=Path("models/"))
result = detector.detect(text, include_sentence_scores=True)
```

### With spaCy and Sentence-Transformers

```python
import spacy
from sentence_transformers import SentenceTransformer
from cipher import CIPHERDetector

nlp = spacy.load("en_core_web_sm")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

detector = CIPHERDetector(nlp=nlp, encoder=encoder)
result = detector.detect(text)
```

### Reading the Result

```python
result = detector.detect(text)

# Overall
result.document_score         # float [0,1]: P(AI-generated)
result.label                  # "AI" | "Human" | "Uncertain"
result.prediction_set         # e.g. ["AI"] or ["AI", "Human"]
result.confidence_interval    # (lo, hi) tuple
result.coverage               # e.g. 0.95

# Per-layer signals
bd = result.signal_breakdown
bd.layer1_compression         # float | None
bd.layer2_cognitive           # float | None
bd.layer3_discourse           # float | None
bd.layer4_manifold            # float | None (None if no encoder)

# Per-sentence attribution
for ss in result.sentence_scores:
    print(f"{ss.score:.0%} [{ss.is_highlighted}]  {ss.text[:60]}")

# Highlighted spans (character offsets into original text)
for span in result.highlighted_spans:
    print(f"  chars {span.start}–{span.end}: {span.score:.0%}")

# Metadata
result.word_count
result.processing_ms
result.model_version
```

---

## Interpreting Results

### Score Thresholds

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.75 | **AI** | Strongly AI-generated |
| 0.50–0.75 | Uncertain / AI | Likely AI-generated |
| 0.35–0.50 | Uncertain / Human | Likely human-written |
| < 0.35 | **Human** | Strongly human-written |

The exact cut-off depends on the conformal prediction quantile, not a hard threshold.

### Confidence Interval

The interval `[lo, hi]` at `coverage=0.95` contains the true P(AI) 95% of the time for documents drawn from the same distribution as the calibration set. Narrow intervals indicate high confidence; wide intervals mean the text is on the boundary.

### Prediction Set

- `["AI"]` — only AI label is consistent with the evidence
- `["Human"]` — only Human label is consistent
- `["AI", "Human"]` — either label is plausible (uncertain)

An empty prediction set (rare) is reported as `["AI", "Human"]` as a safe fallback.

---

## Training Your Own Model

### Step 1 — Prepare Data

Your CSV must have two columns:

```
text,label
"The model generated...",1
"I think the results were...",0
```

`label` values: `1` = AI-generated, `0` = Human-written.

Recommended minimum: 5 000 documents per class. At least 500 examples per class should be held out for calibration (controlled by `--calibration-split`).

Pre-built public datasets are documented in [Dataset Documentation](datasets.md).

### Step 2 — Download/Prepare a Dataset

```bash
# Download 6 public AI-detection datasets (~100k rows total)
python download_datasets.py --output-dir data/full/

# Or sample a smaller balanced subset for fast iteration
python sample_dataset.py --n-each 5000 --output-dir data/sampled/
```

### Step 3 — Train

```bash
cipher train meta \
  --data data/full/train.csv \
  --output-dir models/ \
  --spacy-model en_core_web_sm \
  --no-use-encoder   # omit encoder for faster training
```

### Step 4 — Evaluate

```bash
cipher evaluate data/full/test.csv --model-dir models/
```

### Step 5 — Use the Model

```bash
cipher detect --model-dir models/ "Your text here."
```

---

## Security

### Input Validation

- CIPHER requires at least 20 words. Shorter texts raise `ValueError`.
- Long texts (> 50 000 characters) are silently truncated in the discourse feature layer to prevent quadratic slowdowns in the NER co-occurrence graph.
- No user input is passed to a shell or executed.

### Path Handling

- `detect-file` reads only files that exist on disk. Non-existent paths return exit code 1.
- Model directories are read-only at inference time.
- No path traversal is possible: all paths are validated by `pathlib.Path`.

### Pickle Security

Model files (`meta_learner.pkl`, `isotonic.pkl`, `manifold_scorer.pkl`) are loaded with Python's `pickle`. **Only load model files you trust.** Loading a crafted `.pkl` file from an untrusted source can execute arbitrary code.

### Denial of Service

Very long texts (megabytes) will slow down feature extraction (particularly the spaCy NLP pipeline). Consider truncating inputs at the application boundary.

---

## Troubleshooting

### "Text too short: N words. Minimum is 20."

The input text has fewer than 20 words. Provide more text.

### All texts score ~100% AI

This indicates a calibration issue with a trained model. Symptoms:
- The `d4_topic_var` feature fallback was set to 0.1 instead of 0.0 in the training data, inflating the z-score.
- The isotonic calibrator is being applied to a trained LogisticRegression model.

Fix: ensure you are using `model_dir` pointing to models trained with CIPHER v1.0.0+, which contains the calibration bypass for trained models. See [Architecture § Calibration](architecture.md#calibration).

### spaCy model not found

```bash
python -m spacy download en_core_web_sm
```

### sentence-transformers not available

```bash
pip install sentence-transformers
```

This is optional. Omit `--encoder` to run without it.

### ImportError: No module named 'torch'

Install the neural extra:

```bash
pip install "cipher-detect[neural]"
```
