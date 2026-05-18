# API Reference

> Cross-references: [User Guide](user_guide.md) · [Architecture](architecture.md) · [Dataset Documentation](datasets.md) · [Index](index.md)

---

## Table of Contents

1. [Top-level Package (`cipher`)](#top-level-package-cipher)
2. [CIPHERDetector](#cipherdetector)
3. [DetectionResult](#detectionresult)
4. [SignalBreakdown](#signalbreakdown)
5. [Feature Layers](#feature-layers)
   - [CompressionFeatures](#compressionfeatures)
   - [CognitiveFeatures](#cognitivefeatures)
   - [DiscourseFeatures](#discoursefeatures)
   - [FeatureVector](#featurevector)
6. [Models](#models)
   - [MetaLearner](#metalearner)
   - [ManifoldScorer](#manifoldscorer)
   - [ManifoldEncoder](#manifoldencoder)
7. [Calibration](#calibration)
   - [IsotonicCalibrator](#isotoniccalibrator)
   - [ConformalPredictor](#conformalpredictor)
   - [ConformalResult](#conformalresult)
8. [Output](#output)
   - [SentenceScore](#sentencescore)
   - [HighlightedSpan](#highlightedspan)
9. [Training](#training)
   - [train_meta](#train_meta)
   - [train_contrastive](#train_contrastive)
10. [Exceptions](#exceptions)

---

## Top-level Package (`cipher`)

```python
from cipher import CIPHERDetector, DetectionResult
```

The `cipher` package exposes two names directly:

| Name | Type | Description |
|---|---|---|
| `CIPHERDetector` | class | Main detector. See [CIPHERDetector](#cipherdetector). |
| `DetectionResult` | dataclass | Returned by `CIPHERDetector.detect()`. See [DetectionResult](#detectionresult). |

---

## CIPHERDetector

```python
from cipher import CIPHERDetector
# or
from cipher.detect import CIPHERDetector
```

### Constructor

```python
CIPHERDetector(
    model_dir: Path | None = None,
    nlp: Any | None = None,
    encoder: Any | None = None,
    ai_corpus: str | None = None,
    human_corpus: str | None = None,
    coverage: float = 0.95,
    highlight_threshold: float = 0.60,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_dir` | `Path \| None` | `None` | Directory containing `meta_learner.pkl`, `isotonic.pkl`, `conformal.npy`. When `None`, heuristic mode is used. |
| `nlp` | spaCy `Language \| None` | `None` | Pre-loaded spaCy model. When `None`, discourse topology features use fallback values. |
| `encoder` | `SentenceTransformer \| None` | `None` | Pre-loaded sentence-transformers model for semantic coherence in Layer 2 and sentence attribution. |
| `ai_corpus` | `str \| None` | `None` | Reference text corpus for NCD AI distance (Layer 1). |
| `human_corpus` | `str \| None` | `None` | Reference text corpus for NCD human distance (Layer 1). |
| `coverage` | `float` | `0.95` | Conformal prediction coverage level. Must be in [0.80, 0.99]. |
| `highlight_threshold` | `float` | `0.60` | Minimum sentence AI score to mark as `is_highlighted=True`. |

### Class Attributes

| Attribute | Value | Description |
|---|---|---|
| `MIN_WORDS` | `20` | Minimum number of words required for detection. |

### Methods

#### `detect(text, include_sentence_scores=True) → DetectionResult`

Analyse a document and return a `DetectionResult`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | required | Document text to analyse. |
| `include_sentence_scores` | `bool` | `True` | Whether to compute and return per-sentence scores. Disable for batch performance. |

**Raises:**

| Exception | Condition |
|---|---|
| `ValueError` | `text` contains fewer than `MIN_WORDS` (20) words. |

**Example:**

```python
from cipher import CIPHERDetector
detector = CIPHERDetector()
result = detector.detect("The model generated coherent output...")
print(result.document_score)  # e.g. 0.82
```

---

## DetectionResult

```python
from cipher.detect import DetectionResult
```

A frozen dataclass containing all outputs of a single detection call.

### Fields

| Field | Type | Description |
|---|---|---|
| `document_score` | `float` | Calibrated P(AI-generated) ∈ [0, 1]. |
| `prediction_set` | `list[str]` | Conformal prediction set: `["AI"]`, `["Human"]`, or `["AI", "Human"]`. |
| `confidence_interval` | `tuple[float, float]` | `(lo, hi)` conformal interval at `coverage` level. |
| `coverage` | `float` | Coverage level, e.g. `0.95`. |
| `signal_breakdown` | `SignalBreakdown` | Per-layer partial scores. |
| `sentence_scores` | `list[SentenceScore]` | Per-sentence attribution. Empty if `include_sentence_scores=False`. |
| `highlighted_spans` | `list[HighlightedSpan]` | Character-offset spans for highlighted sentences. |
| `feature_vector` | `FeatureVector` | Full 22/23-dimensional feature vector used for this document. |
| `word_count` | `int` | Number of whitespace-delimited tokens in the input. |
| `processing_ms` | `float` | Wall-clock milliseconds from call to return. |
| `model_version` | `str` | `"heuristic"` or `"trained"`. |

### Properties

#### `label → str`

Human-readable classification: `"AI"`, `"Human"`, or `"Uncertain"`.

| `prediction_set` | `label` |
|---|---|
| `["AI"]` | `"AI"` |
| `["Human"]` | `"Human"` |
| anything else | `"Uncertain"` |

---

## SignalBreakdown

```python
from cipher.detect import SignalBreakdown
```

A dataclass with per-layer partial AI probability scores. Scores are `None` when the corresponding features are unavailable.

### Fields

| Field | Type | Description |
|---|---|---|
| `layer1_compression` | `float \| None` | Compression layer signal (l1–l7). |
| `layer2_cognitive` | `float \| None` | Cognitive load signal (c1–c8). |
| `layer3_discourse` | `float \| None` | Discourse topology signal (d1–d7). |
| `layer4_manifold` | `float \| None` | Contrastive manifold signal (m1). `None` when Layer 4 is not loaded. |

### Methods

#### `as_dict() → dict[str, float | None]`

Return all four fields as a dictionary.

---

## Feature Layers

### CompressionFeatures

```python
from cipher.layers.compression import compute_compression_features, CompressionFeatures
```

#### `compute_compression_features(text, ai_corpus=None, human_corpus=None) → CompressionFeatures`

Compute 7 compression-theoretic features from `text`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Input document. |
| `ai_corpus` | `str \| None` | Reference corpus for NCD (AI). |
| `human_corpus` | `str \| None` | Reference corpus for NCD (Human). |

**Returns:** `CompressionFeatures` dataclass with fields `l1_lz76` through `l7_ncd_human` and an `available` list of feature names.

---

### CognitiveFeatures

```python
from cipher.layers.cognitive import compute_cognitive_features, CognitiveFeatures
```

#### `compute_cognitive_features(text, sentences=None, encoder=None) → CognitiveFeatures`

Compute 8 cognitive load fingerprint features.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Input document. |
| `sentences` | `list[str] \| None` | Pre-split sentence list. Will be split internally if `None`. |
| `encoder` | `Any \| None` | sentence-transformers encoder for semantic coherence. |

**Returns:** `CognitiveFeatures` dataclass with fields `c1_sl_kl` through `c8_avg_dep_dist`.

---

### DiscourseFeatures

```python
from cipher.layers.discourse import compute_discourse_features, DiscourseFeatures
```

#### `compute_discourse_features(text, nlp=None) → DiscourseFeatures`

Compute 7 discourse topology features.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Input document. |
| `nlp` | `spacy.Language \| None` | spaCy model for co-reference and dependency parsing. Falls back to reduced feature set if `None`. |

**Returns:** `DiscourseFeatures` frozen dataclass with fields `d1_coref_density` through `d7_depth_skew`.

---

### FeatureVector

```python
from cipher.layers.assembler import FeatureVector
```

Assembles the outputs of Layers 1–4 into a single array for the meta-learner.

#### Constructor

```python
FeatureVector(
    compression: CompressionFeatures,
    cognitive: CognitiveFeatures,
    discourse: DiscourseFeatures,
    manifold_score: float | None = None,
)
```

#### Class Attributes

| Attribute | Type | Description |
|---|---|---|
| `FEATURE_NAMES` | `tuple[str, ...]` | 23-element tuple of feature names (l1…l7, c1…c8, d1…d7, m1). |

#### Methods

##### `to_array(include_manifold=False) → np.ndarray`

Return a 1D float64 array of shape `(22,)` or `(23,)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `include_manifold` | `bool` | `False` | Whether to append `m1_manifold` as the 23rd feature. |

##### `available_features(include_manifold=False) → list[str]`

Return a list of feature names for which non-fallback values were computed.

##### `as_dict() → dict[str, float]`

Return all 22 (or 23) features as a `{name: value}` dictionary.

---

## Models

### MetaLearner

```python
from cipher.models.meta_learner import MetaLearner
```

Layer 5 classifier. Wraps a scikit-learn `LogisticRegression` in trained mode, or uses a hand-tuned heuristic in untrained mode.

#### Constructor

```python
MetaLearner()  # always starts in heuristic mode
```

#### Properties

| Property | Type | Description |
|---|---|---|
| `mode` | `str` | `"heuristic"` before `fit()`, `"trained"` after. |

#### Methods

##### `fit(X, y) → MetaLearner`

Train the logistic regression meta-learner.

| Parameter | Type | Description |
|---|---|---|
| `X` | `np.ndarray` | Feature matrix of shape `(n_docs, 22)` or `(n_docs, 23)`. |
| `y` | `np.ndarray` | Binary labels of shape `(n_docs,)`. `1` = AI, `0` = Human. |

##### `predict_proba(x) → float`

Return P(AI) for a single feature vector `x` of shape `(22,)` or `(23,)`.

NaN and ±Inf values in `x` are imputed to 0. The z-scored vector is clipped to `[-5, 5]` before logistic regression to prevent extreme activations.

##### `feature_importances() → dict[str, float] | None`

Return a dictionary mapping feature names to their logistic regression coefficients. Returns `None` in heuristic mode.

##### `save(path: Path) → None`

Pickle the fitted model to `path`.

##### `classmethod load(path: Path) → MetaLearner`

Load a pickled model from `path`.

---

### ManifoldScorer

```python
from cipher.models.contrastive_encoder import ManifoldScorer
```

GMM-based density ratio scorer for Layer 4 embeddings.

#### Constructor

```python
ManifoldScorer(n_components: int = 16)
```

#### Methods

##### `fit(ai_embeddings, human_embeddings) → ManifoldScorer`

Fit two GMMs on AI and human embedding arrays of shape `(n, d)`.

**Raises:** `RuntimeError` if scikit-learn is not installed.

##### `score(embedding) → float`

Return $P(\text{AI} | z) = p_{AI}(z) / (p_{AI}(z) + p_{Human}(z))$.

Returns `0.5` before fitting.

##### `save(path: Path) → None` / `classmethod load(path: Path) → ManifoldScorer`

Pickle-based serialisation.

---

### ManifoldEncoder

```python
from cipher.models.contrastive_encoder import ManifoldEncoder
```

High-level wrapper that encodes text to embeddings and returns a manifold score. When `torch` is not installed, silently returns `None` for all operations.

#### Constructor

```python
ManifoldEncoder(
    model_path: Path | None = None,
    tokenizer_name: str = "microsoft/deberta-v3-base",
    device: str | None = None,
)
```

#### Properties

| Property | Returns | Description |
|---|---|---|
| `available` | `bool` | `True` only when torch is installed and a model is loaded. |

#### Methods

##### `encode(text, max_length=512) → np.ndarray | None`

Returns a 256-dim L2-normalised embedding, or `None` if unavailable.

##### `manifold_score(text) → float | None`

Returns `ManifoldScorer.score(encode(text))`, or `None` if unavailable.

---

## Calibration

### IsotonicCalibrator

```python
from cipher.calibration.isotonic import IsotonicCalibrator
```

Post-hoc monotone calibration via isotonic regression.

#### Constructor

```python
IsotonicCalibrator()
```

#### Methods

##### `fit(raw_scores, labels) → IsotonicCalibrator`

Fit on calibration set.

| Parameter | Type | Description |
|---|---|---|
| `raw_scores` | `np.ndarray` | Array of raw meta-learner scores. |
| `labels` | `np.ndarray` | Binary labels (1=AI, 0=Human). |

##### `transform(score: float) → float`

Map a single raw score to a calibrated probability. Returns `clip(score, 0, 1)` if not fitted.

##### `transform_array(scores: np.ndarray) → np.ndarray`

Vectorised version of `transform`.

##### `save(path: Path) → None` / `classmethod load(path: Path) → IsotonicCalibrator`

Pickle-based serialisation.

---

### ConformalPredictor

```python
from cipher.calibration.conformal import ConformalPredictor
```

Split-conformal predictor. See [Architecture § Conformal Prediction](architecture.md#conformal-prediction).

#### Constructor

```python
ConformalPredictor(alpha: float = 0.05)
```

`alpha` is the miscoverage rate; coverage = `1 - alpha`. Default `alpha=0.05` gives 95% coverage.

#### Methods

##### `calibrate(scores, labels) → ConformalPredictor`

Fit the nonconformity quantile on calibration data.

##### `predict(score: float) → ConformalResult`

Return a `ConformalResult` for a single calibrated score.

##### `save(path: Path) → None` / `classmethod load(path: Path) → ConformalPredictor`

NumPy array serialisation to `.npy` format.

---

### ConformalResult

A dataclass returned by `ConformalPredictor.predict()`.

| Field | Type | Description |
|---|---|---|
| `score` | `float` | Input calibrated score. |
| `prediction_set` | `list[str]` | `["AI"]`, `["Human"]`, or `["AI", "Human"]`. |
| `interval` | `tuple[float, float]` | `(lo, hi)` confidence interval. |
| `alpha` | `float` | Miscoverage rate used. |

**Properties:**

| Property | Returns | Description |
|---|---|---|
| `coverage` | `float` | `1 - alpha` |
| `label` | `str` | `"AI"`, `"Human"`, or `"Uncertain"` |

---

## Output

### SentenceScore

```python
from cipher.output.highlighter import SentenceScore
```

Per-sentence detection result.

| Field | Type | Description |
|---|---|---|
| `index` | `int` | 0-based sentence index. |
| `text` | `str` | Sentence text. |
| `score` | `float` | AI probability ∈ [0, 1]. |
| `is_highlighted` | `bool` | `True` if `score >= highlight_threshold`. |

#### Methods

##### `char_range(full_text: str) → tuple[int, int]`

Return `(start, end)` character offsets of this sentence in `full_text`. Returns `(0, 0)` if the sentence is not found.

---

### HighlightedSpan

```python
from cipher.output.highlighter import HighlightedSpan
```

Character-level span for a highlighted sentence.

| Field | Type | Description |
|---|---|---|
| `start` | `int` | Start character offset in original text. |
| `end` | `int` | End character offset (exclusive). |
| `score` | `float` | AI probability for this span. |

---

## Training

The training API is exposed via the `cipher train` CLI subcommands. The underlying Python functions are importable for programmatic use.

### train_meta

```python
from cipher.training.train_meta import train_meta, _load_labeled_csv, _extract_features
```

#### `_load_labeled_csv(path: Path) → tuple[list[str], list[int]]`

Read a `text,label` CSV. Returns `(texts, labels)`.

#### `_extract_features(texts, nlp, encoder, ai_corpus, human_corpus) → np.ndarray`

Run the full 22-feature pipeline on a list of texts. Returns array of shape `(n, 22)`.

#### `train_meta` (Typer command)

See [User Guide § train meta](user_guide.md#train-meta).

---

### train_contrastive

```python
from cipher.training.train_meta import train_contrastive
```

Requires `torch` and `transformers`. Trains the DeBERTa-v3-base encoder with triplet loss.

See [User Guide § train contrastive](user_guide.md#train-contrastive).

---

## Exceptions

| Exception | Raised by | Condition |
|---|---|---|
| `ValueError` | `CIPHERDetector.detect()` | Input has fewer than 20 words. |
| `RuntimeError` | `ManifoldScorer.fit()` | scikit-learn is not installed. |
| `typer.Exit(1)` | `train_contrastive` | torch/transformers not installed. |
| `typer.Exit(1)` | `detect-file` CLI | Specified file does not exist. |
