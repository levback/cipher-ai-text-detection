# AEGIS v2 — Adaptive Ensemble for AI-Generated text Identification System
## Implementation Plan

**Version**: 2.0  
**Date**: May 2026  
**Basis**: Full research review of 2019–2026 literature (42 papers)

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Requirements](#2-requirements)
3. [Architecture Overview](#3-architecture-overview)
4. [Technology Stack](#4-technology-stack)
5. [Implementation Phases](#5-implementation-phases)
   - [Phase 1 — Scaffold & Data](#phase-1--scaffold--data)
   - [Phase 2 — Structural Signals](#phase-2--structural-signals)
   - [Phase 3 — Statistical Signals](#phase-3--statistical-signals)
   - [Phase 4 — Neural Classifier (MELD-style)](#phase-4--neural-classifier-meld-style)
   - [Phase 5 — Watermark Detection](#phase-5--watermark-detection)
   - [Phase 6 — Ensemble & Calibration](#phase-6--ensemble--calibration)
   - [Phase 7 — Sentence-Level Attribution](#phase-7--sentence-level-attribution)
   - [Phase 8 — Retrieval-Augmented Detection](#phase-8--retrieval-augmented-detection)
   - [Phase 9 — API & Explainability](#phase-9--api--explainability)
   - [Phase 10 — Evaluation Framework](#phase-10--evaluation-framework)
   - [Phase 11 — Continuous Learning](#phase-11--continuous-learning)
   - [Phase 12 — Tests](#phase-12--tests)
6. [Directory Structure](#6-directory-structure)
7. [API Contract](#7-api-contract)
8. [Testing Strategy](#8-testing-strategy)
9. [Risk Register](#9-risk-register)
10. [Weakness → Phase Mapping](#10-weakness--phase-mapping)
11. [Success Criteria](#11-success-criteria)
12. [Research Grounding](#12-research-grounding)

---

## 1. Project Vision

AEGIS is a **research-grade, fully open-source** AI text detector that combines every independently validated signal family identified in the 2019–2026 literature into a single calibrated pipeline. It is designed to:

- Remain useful as LLMs improve (multi-signal resilience)
- Be honest about uncertainty (conformal prediction, not binary labels)
- Be fair across languages and writing backgrounds (bias-aware calibration)
- Stay current as new models emerge (continuous learning loop)
- Be fully reproducible and auditable (open weights, open data, MLflow tracking)

The system is named for the mythological shield that deflects every kind of attack — the system's layered design means no single attack vector can defeat it.

---

## 2. Requirements

| ID | Requirement | Source Motivation |
|----|-------------|-------------------|
| **R1** | Robust to paraphrasing attacks | DIPPER attacks reduce single-signal detectors to ~50% AUROC |
| **R2** | Low false positive rate for non-native English writers | Liang et al. (2023): systematic bias problem |
| **R3** | Graceful degradation on short texts (≥ 50 words) | All statistical methods fail below ~150 tokens |
| **R4** | Mixed human-AI attribution at sentence level | Turnitin does this; research tools generally don't |
| **R5** | Multilingual (EN, ES, ZH, AR, DE, FR, JA minimum) | BLUFF (2026): gap is critical |
| **R6** | Adapt to new LLMs within 72 h of public release | MELD-eval shows sharp degradation on new models |
| **R7** | Output calibrated probabilities with prediction sets | Avoid wrongful accusations; conformal prediction |
| **R8** | Detect AI-generated code and structured text | Code/structured text gap in current literature |
| **R9** | Watermark detection as a high-confidence override | KGW + ECC watermarks are open standards |
| **R10** | Reproducible evaluation against RAID / M4 / BLUFF | Pudasaini (2026): benchmark artifact warning |
| **R11** | Per-token explainability | NOTAI.AI (2026): deployment trust requirement |
| **R12** | Bias audit against 3 speaker groups | Disparity < 2× FPR across groups |

---

## 3. Architecture Overview

### 3.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: text (1 to N documents)                                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                     ┌────────▼────────┐
                     │  Layer 1        │
                     │  Confidence     │◄── returns INSUFFICIENT_TEXT
                     │  Gate           │    if < 100 tokens
                     └────────┬────────┘
                              │
          ┌───────────────────┼───────────────────────┐
          ▼                   ▼                        ▼
  ┌───────────────┐  ┌───────────────────┐   ┌──────────────────┐
  │  Layer 2      │  │  Layer 3          │   │  Layer 4         │
  │  Watermark    │  │  Zero-Shot        │   │  Structural /    │
  │  Detection    │  │  Statistical      │   │  Linguistic      │
  │  (KGW + ECC)  │  │  (Binoculars,     │   │  (DependencyAI,  │
  │               │  │   Fast-DetectGPT, │   │   PHD, stylom.)  │
  │               │  │   LAPD)           │   │                  │
  └───────┬───────┘  └─────────┬─────────┘   └────────┬─────────┘
          │                    │                       │
          │          ┌─────────▼────────────────────┐  │
          │          │  Layer 5                     │  │
          │          │  Multi-Task Neural Classifier │  │
          │          │  (MELD architecture:          │  │
          │          │   binary + generator +        │  │
          │          │   attack + domain heads)      │  │
          │          └─────────┬────────────────────┘  │
          │                    │                        │
          └──────────┬─────────┴────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Layer 6            │
          │  Retrieval-Augmented│
          │  (pgvector HNSW)    │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Layer 7            │
          │  CatBoost Meta-     │
          │  Learner Fusion     │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Layer 8            │
          │  Conformal          │
          │  Calibration        │◄── returns {AI}, {HUMAN}, or {?}
          │  + Bias Correction  │    with coverage guarantee
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Layer 9            │
          │  Sentence-Level     │
          │  Attribution        │◄── optional; slower path
          └──────────┬──────────┘
                     │
              DetectionResponse
```

### 3.2 Dynamic Signal Routing

The meta-learner receives not just signal scores but also metadata that triggers routing:

| Condition detected | Routing adjustment |
|--------------------|-------------------|
| Short text (50–100 tokens) | Disable statistical; up-weight structural + neural |
| Suspected paraphrase (low stat score, high neural) | Up-weight PHD + syntactic |
| Code / structured text | Switch to code-specific stylometric features |
| Non-English input | Up-weight multilingual neural; down-weight English-centric stats |
| Watermark detected (p < 0.01) | Override with WATERMARKED label; bypass fusion |

---

## 4. Technology Stack

| Layer | Technology | License | Rationale |
|-------|-----------|---------|-----------|
| Neural classifier backbone | `xlm-roberta-large` | MIT | 560M params; multilingual; proven |
| Training framework | PyTorch + HuggingFace Trainer | Apache 2.0 | Standard; well-supported |
| Efficient fine-tuning | `peft` (LoRA) | Apache 2.0 | Catastrophic forgetting prevention |
| Topological signal | `ripser` + `persim` | MIT | Persistent homology |
| Dependency parsing | `spacy` (multi-model) | MIT | 25+ language pipelines |
| Meta-learner | `CatBoost` | Apache 2.0 | Handles missing features; fast |
| Conformal calibration | `MAPIE` | BSD 3-Clause | Coverage guarantees; simple API |
| Vector store | `pgvector` (PostgreSQL) | MIT | Production-grade; HNSW support |
| API | `FastAPI` | MIT | Async; schema validation |
| Task queue | `Celery` + `Redis` | MIT / BSD | Async batch inference |
| Experiment tracking | `MLflow` | Apache 2.0 | Reproducibility |
| Data format | Apache Arrow (`pyarrow`) | Apache 2.0 | Efficient columnar storage |
| Explainability | `shap` | MIT | Token-level attribution |

---

## 5. Implementation Phases

---

### Phase 1 — Scaffold & Data

> **Goal**: Repository structure, environment, and data pipelines for all 7 languages and 5 domains.

#### 1.1 Repository and environment

- Create directory tree (see §6)
- Write `pyproject.toml` with pinned dependencies
- Set up `pre-commit` hooks: `ruff`, `mypy`, `pytest --co` (no-run check)
- **Files**: `pyproject.toml`, `src/__init__.py`, `.pre-commit-config.yaml`
- **Risk**: Low

#### 1.2 Human text corpus

- Fetch from HuggingFace datasets: `wikitext-103`, `cc_news`, `scientific_papers` (arXiv), `bookcorpus`, `multi_news`
- Stratify by: domain (academic, news, creative, legal, code), language (7 targets)
- Save as Apache Arrow with schema: `(id, text, language, domain, source, word_count)`
- **Files**: `data/collectors/human_corpus.py`, `data/schema.py`
- **Risk**: Low

#### 1.3 AI-generated text corpus

- Call APIs: OpenAI (GPT-4o, GPT-4o-mini), Anthropic (Claude 3.5 Sonnet), Google (Gemini 1.5 Pro), Mistral (Large), Meta (Llama-3.1-70B via Replicate or Together), Cohere (Command R+)
- Generation parameters: temperature ∈ {0.3, 0.7, 1.0}, top-p ∈ {0.9, 1.0}, greedy decoding
- Label with: model_name, temperature, top_p, domain, language, prompt_id
- Budget cap: 500 samples per (model × domain × language) combination
- **Files**: `data/collectors/ai_corpus.py`
- **Risk**: Medium — API cost; mitigate with caching and sampling budgets

#### 1.4 Attack corpus (adversarial augmentation)

Generate attacked variants of AI corpus:

| Attack | Tool | Notes |
|--------|------|-------|
| Discourse paraphrase | DIPPER (`jxhe/dipper-paraphrases`) | Levels 20, 40, 60 |
| Back-translation EN↔FR, EN↔ZH, EN↔DE | NLLB-200 (HuggingFace) | Two-hop |
| GPT-4 rewrite | OpenAI API with "humanize this text" prompt | High quality attack |
| Synonym substitution | `nlpaug` library | Token-level |
| Character substitution | Custom script | Low-effort attack |

- **Files**: `data/collectors/attack_corpus.py`
- **Risk**: Medium — DIPPER needs GPU; cache all outputs

#### 1.5 Dataset registry

- Train/val/test splits stratified by domain × language × model × attack
- Record version hash in MLflow
- Enforce strict test-set isolation
- **Files**: `data/registry.py`
- **Risk**: Low — contamination check is critical

---

### Phase 2 — Structural Signals

> **Goal**: Paraphrase-robust signals that work without logit access. Basis: DependencyAI (2026), PHD (2024), stylometric literature.

#### 2.1 Syntactic dependency signal

Implement `src/signals/dependency.py`:

```python
def dependency_signal(text: str, lang: str = "en") -> SignalResult:
    """
    Computes distribution over 37 Universal Dependency relation types.
    Returns normalized histogram as feature vector + AI-probability estimate.
    
    Based on: DependencyAI (Ahmed & Hammond, 2026, arXiv:2602.15514)
    """
```

- Use spacy `xx_ent_wiki_sm` (multilingual) or language-specific model
- Extract: arc type histogram, arc length distribution, tree depth stats
- Train a per-language logistic regression on these features against labeled corpus (Phase 1)
- **Files**: `src/signals/dependency.py`
- **Risk**: Low — spacy is mature; logistic regression is stable

#### 2.2 Stylometric signal

Implement `src/signals/stylometric.py` with features:

| Feature group | Features | Notes |
|---------------|----------|-------|
| Lexical diversity | TTR, MTTR (moving window TTR), Hapax ratio | Language-agnostic |
| Sentence structure | Mean length, variance, Gini coefficient | Language-agnostic |
| Burstiness | Variance of per-token surprisal (needs LM) | Requires logit access |
| Punctuation | Density, pattern entropy | Language-specific calibration |
| Readability | Flesch-Kincaid, Gunning Fog (EN only) | EN only |
| Code-specific | Identifier entropy, comment ratio, cyclomatic proxy | When domain = code |
| POS distribution | Normalized histogram over 17 Universal POS tags | Language-agnostic via spacy |

- **Files**: `src/signals/stylometric.py`
- **Risk**: Low

#### 2.3 Topological signal (PHD)

Implement `src/signals/topological.py`:

```python
def phd_score(text: str, embedding_model: str = "intfloat/multilingual-e5-large") -> SignalResult:
    """
    Persistent Homology Dimension estimate.
    AI text has systematically lower intrinsic dimension than human text.
    
    Based on: Tulchinskii et al. (2024, NeurIPS), arXiv:2306.04723
    
    Implementation:
    1. Split text into sentences
    2. Embed each sentence with multilingual-e5-large
    3. Compute Vietoris-Rips persistent homology via ripser
    4. Extract PHD estimate as 1/(1 + H1_lifetime_sum)
    5. Cache embeddings keyed by sha256(text)
    """
```

- Chunking for long docs: 512-token windows, average PHD
- **Files**: `src/signals/topological.py`
- **Risk**: Medium — `ripser` is slow for long texts; requires GPU for embedding

#### 2.4 Confidence gate

Implement `src/signals/confidence_gate.py`:

```python
@dataclass
class SignalResult:
    score: float           # [0, 1] — probability of AI
    confidence: float      # [0, 1] — how reliable this signal is
    is_reliable: bool      # False when text too short for this signal
    signal_name: str
    feature_values: dict   # raw features for explainability

MINIMUM_TOKENS = {
    "statistical": 150,
    "topological": 100,   # needs ≥5 sentences
    "stylometric": 75,
    "dependency": 50,
    "neural": 50,
}
```

- **Files**: `src/signals/confidence_gate.py`
- **Risk**: Low

---

### Phase 3 — Statistical Signals

> **Goal**: Zero-shot cross-check signals. Basis: Binoculars (Hans, 2024), Fast-DetectGPT (Bao, 2023), LAPD (Wu, 2026).

#### 3.1 Fast-DetectGPT scorer

```python
def fast_detectgpt_score(text: str, scoring_model: str) -> SignalResult:
    """
    Conditional probability curvature — no perturbations needed.
    
    d(x) ≈ (1/N) Σ_t [ log p(x_t|x<t) - H(p(·|x<t)) ]
    
    where H is per-position entropy.
    Positive values → AI text (lower entropy residual).
    
    Based on: Bao et al. (2023), arXiv:2310.05130
    """
```

- Fallback: if source model logits unavailable, use `falcon-7b` or `mistral-7b` as surrogate
- **Files**: `src/signals/statistical.py`

#### 3.2 Binoculars scorer

```python
def binoculars_score(text: str, model_a: str, model_b: str) -> SignalResult:
    """
    Cross-entropy / perplexity ratio.
    score = CE(x; M1) / PPL(x; M2)
    
    Human text: high ratio (both models similarly "surprised")
    AI text: low ratio (generating model M2 is not surprised)
    
    Based on: Hans et al. (2024), arXiv:2401.12070
    """
```

- Default: `model_a = falcon-7b-instruct`, `model_b = falcon-7b`
- **Files**: `src/signals/statistical.py`

#### 3.3 LAPD / Alignment Imprint scorer

```python
def lapd_score(text: str, base_model: str, aligned_model: str) -> SignalResult:
    """
    Log-likelihood Alignment Preference Discrepancy.
    
    Alignment Imprint = log p_aligned(x) - log p_base(x)
    LAPD standardizes this by weighting by position entropy.
    
    Provably dominates Fast-DetectGPT (Wu et al., 2026, arXiv:2604.16923).
    Requires: base model + RLHF-aligned variant.
    """
```

- Default pair: `mistralai/Mistral-7B-v0.3` (base) + `mistralai/Mistral-7B-Instruct-v0.3`
- **Files**: `src/signals/statistical.py`

---

### Phase 4 — Neural Classifier (MELD-style)

> **Goal**: Main detection signal. Open-source SOTA (arXiv:2605.06903).

#### 4.1 Model architecture

Implement `src/models/classifier.py`:

```
Encoder: xlm-roberta-large (frozen first 12 layers, fine-tune last 12)
    │
    ├── Head 1: Binary AI/Human           (cross-entropy loss, weight λ1)
    ├── Head 2: Generator family          (GPT / Claude / Gemini / Llama / Other)
    ├── Head 3: Attack type detected      (clean / paraphrase / backtrans / synonym)
    └── Head 4: Domain                   (academic / news / creative / legal / code)

Loss: L = Σ_k (1 / (2σ_k²)) * L_k + log σ_k   [homoscedastic uncertainty weights]
```

- All heads share the encoder; auxiliary heads are discarded at inference
- Hard-negative pairwise ranking loss on top of binary head to enlarge AI/human margin
- **Files**: `src/models/classifier.py`

#### 4.2 EMA teacher-student training

Implement `src/models/trainer.py`:

- **Teacher**: exponential moving average of student weights; receives clean inputs
- **Student**: receives attack-augmented inputs; distilled toward teacher predictions
- Distillation loss: KL divergence between teacher and student binary head outputs
- EMA decay: $\tau = 0.999$

```
Clean text  ──► Teacher (EMA) ──► soft labels ──┐
                                                  ▼ KL distillation
Attacked text ──► Student ──────────────────► Main loss
```

- **Files**: `src/models/trainer.py`

#### 4.3 Adversarial augmentation loop

Implement `src/models/adversarial_trainer.py` (RADAR-style):

1. Train LoRA paraphraser (Mistral-7B) to maximize student detector loss
2. Add its outputs to attack corpus
3. Retrain student on augmented corpus
4. Freeze paraphraser; repeat for 3 rounds with different paraphrasers (DIPPER, NLLB, GPT-4)

- KL regularization on paraphraser to prevent meaning destruction: $D_{\text{KL}}(p_{\text{para}} \| p_{\text{orig}}) < \epsilon$
- **Files**: `src/models/adversarial_trainer.py`

#### 4.4 Bias mitigation

Implement `src/models/bias_mitigation.py`:

- Proxy for non-native English: CEFR-level classifier trained on WriteHome / ICLE datasets
- Gradient reversal layer on a protected-attribute head (native speaker probability)
- Counterfactual data augmentation: pair human non-native texts with AI texts in "simple English"
- **Files**: `src/models/bias_mitigation.py`

#### 4.5 Training configuration

Implement `src/models/train.py`:

```python
# Key hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-5          # encoder
HEAD_LR = 1e-4                 # heads
EMA_DECAY = 0.999
MAX_SEQ_LEN = 512
PRECISION = "bf16"             # A100 / H100
GRAD_CLIP = 1.0
SCHEDULER = "cosine_with_warmup"
WARMUP_STEPS = 500
LORA_R = 16
LORA_ALPHA = 32
REPLAY_FRACTION = 0.20         # old data replayed during incremental fine-tuning
```

- Early stopping on validation AUROC (patience=3)
- MLflow logging: all hyperparameters, metrics per epoch, model artifacts
- **Files**: `src/models/train.py`

---

### Phase 5 — Watermark Detection

> **Goal**: Implement open published watermark algorithms. Basis: Kirchenbauer (2023), Chao (2024), Block (2025).

#### 5.1 KGW detector

Implement `src/watermark/kgw_detector.py`:

```python
def detect_kgw(text: str, seed_range: int = 1000) -> WatermarkResult:
    """
    For each candidate seed s in [0, seed_range]:
      - Reconstruct green list G_t for each position using hash(s, t, x_{t-1})
      - Count green-list tokens: |G|
      - Compute z = (|G| - N/2) / sqrt(N/4)
      - p-value from N(0,1)
    Report minimum p-value and corresponding seed.
    
    Blind detection: no generator model needed.
    """
```

- Decision threshold: $p < 0.01$
- **Files**: `src/watermark/kgw_detector.py`

#### 5.2 ECC watermark detector

Implement `src/watermark/ecc_detector.py`:

```python
def detect_ecc(text: str, code_params: ECCParams) -> WatermarkResult:
    """
    Error-correcting code watermark detection.
    
    1. For each position, determine token's bit assignment under code
    2. Recover noisy codeword from observed bits
    3. Apply Reed-Solomon decoding to correct up to t errors
    4. Check if decoded message has valid parity
    
    Based on: Chao et al. (2024), arXiv:2406.10281
    """
```

- **Files**: `src/watermark/ecc_detector.py`

#### 5.3 Watermark aggregator

Implement `src/watermark/aggregator.py`:

```python
@dataclass
class WatermarkResult:
    is_watermarked: bool
    watermark_type: str | None       # "KGW", "ECC", None
    confidence: float
    p_value: float | None
    seed: int | None                 # KGW only
```

- If any watermark detected at $p < 0.01$: override entire pipeline with `WATERMARKED` label
- **Files**: `src/watermark/aggregator.py`

---

### Phase 6 — Ensemble & Calibration

> **Goal**: Fuse all signals into a calibrated probability with prediction sets.

#### 6.1 CatBoost meta-learner

Implement `src/ensemble/meta_learner.py`:

**Input feature vector** (per document):

```
statistical signals:   fast_detectgpt_score, binoculars_score, lapd_score + confidences
structural signals:    phd_score, dependency_score, stylometric_score + confidences  
neural signal:         binary_prob + confidence
retrieval signal:      max_cosine_similarity
metadata:              token_count, language_id, domain_id, is_code
```

- Train on held-out calibration partition (not training set — prevents leakage)
- CatBoost handles missing features (unreliable signals) natively
- **Files**: `src/ensemble/meta_learner.py`

#### 6.2 Conformal prediction calibrator

Implement `src/ensemble/calibrator.py` using MAPIE:

```python
from mapie.classification import MapieClassifier

# Split-conformal prediction
# For coverage level 1-α (e.g., 0.90):
#   C_α(x) ⊆ {"AI", "HUMAN", "UNCERTAIN"}
# Guarantee: P(true_label ∈ C_α(x)) ≥ 1-α on calibration distribution

# Confidence interval: [p̂ - δ_α, p̂ + δ_α]
# Width of interval → uncertainty signal
```

- Default coverage: 90% (`alpha = 0.10`)
- When $C_\alpha(x) = \{\text{AI}, \text{HUMAN}\}$: return `UNCERTAIN`
- **Files**: `src/ensemble/calibrator.py`

#### 6.3 Bias correction

Implement `src/ensemble/bias_correction.py`:

- Estimate non-native English probability using proficiency proxy (from §4.4)
- For each language group: measure FPR on calibration set
- Apply learned additive offset: $\hat{p}_{\text{corrected}} = \hat{p} - \Delta_{\text{lang}}$
- Bound: $|\Delta_{\text{lang}}| \leq 0.15$ to prevent over-correction
- Iterate until FPR ≤ 5% for all language groups
- **Files**: `src/ensemble/bias_correction.py`

#### 6.4 Dynamic signal weights

Implement `src/ensemble/dynamic_weights.py`:

```python
def route_signals(meta_features: MetaFeatures) -> WeightVector:
    """
    Adjusts CatBoost feature importances before prediction based on conditions:
    
    SHORT_TEXT (<100 tokens):
        → disable statistical; weight structural × 2.0; weight neural × 1.5
    
    SUSPECTED_PARAPHRASE (stat_score < 0.4 AND neural_score > 0.7):
        → weight PHD × 2.0; weight syntactic × 2.0; suppress statistical
    
    CODE_TEXT (domain = code):
        → disable prose perplexity; use code stylometric features
    
    NON_ENGLISH:
        → weight multilingual neural × 1.5; suppress English-centric stats
    """
```

- **Files**: `src/ensemble/dynamic_weights.py`

---

### Phase 7 — Sentence-Level Attribution

> **Goal**: Identify which sentences are AI-generated in mixed documents.

#### 7.1 Sequence labeler (BIO tagging)

Implement `src/attribution/sequence_labeler.py`:

- Model: `xlm-roberta-base` + token classification head
- Tags: `B-AI`, `I-AI`, `O` (human)
- Loss: CRF loss on top of token logits (sequential consistency)
- Training data: synthetic mixed documents — concatenate human and AI paragraphs with known boundaries
- **Files**: `src/attribution/sequence_labeler.py`

#### 7.2 Change-point detection

Implement `src/attribution/changepoint.py`:

```python
from ruptures import Pelt

def detect_ai_boundaries(sentence_scores: list[float]) -> list[int]:
    """
    Apply PELT (Pruned Exact Linear Time) change-point detection
    to the series of per-sentence AI scores.
    Change-points = candidate human↔AI boundaries.
    
    Fuse with sequence labeler via majority vote.
    """
```

- **Files**: `src/attribution/changepoint.py`

#### 7.3 Attribution report

Implement `src/attribution/report.py`:

```python
@dataclass
class AttributionResult:
    overall_score: float
    sentence_scores: list[SentenceScore]
    boundary_positions: list[int]      # sentence indices of human-AI transitions
    confidence_map: list[float]
    html_report: str                   # color-highlighted HTML
```

- **Files**: `src/attribution/report.py`

---

### Phase 8 — Retrieval-Augmented Detection

> **Goal**: Catch copy-paste reuse of known AI outputs.

#### 8.1 pgvector index

Implement `src/retrieval/index.py`:

```sql
-- PostgreSQL schema
CREATE TABLE ai_text_index (
    id          BIGSERIAL PRIMARY KEY,
    text_hash   TEXT UNIQUE NOT NULL,
    embedding   VECTOR(1024),          -- multilingual-e5-large dim
    model_name  TEXT,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    domain      TEXT,
    language    TEXT
);

CREATE INDEX ON ai_text_index USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

- Populate from AI corpus (Phase 1.3)
- **Files**: `src/retrieval/index.py`

#### 8.2 Retrieval scorer

```python
def retrieval_score(text: str, top_k: int = 5) -> RetrievalResult:
    """
    score = 1 - min_i( cosine_distance(embed(text), embed(neighbor_i)) )
    
    Threshold: cosine_similarity > 0.97 → flag as AI (known output)
    """
```

- **Files**: `src/retrieval/scorer.py`

#### 8.3 Index updater

```python
# Runs as daily cron job
def update_index():
    """
    - Sample from major LLM APIs with diverse prompts
    - Embed with multilingual-e5-large
    - Deduplicate by text_hash
    - Insert into pgvector index
    """
```

- **Files**: `src/retrieval/updater.py`

---

### Phase 9 — API & Explainability

> **Goal**: Production REST API with explainable outputs.

#### 9.1 FastAPI application

Implement `api/main.py`:

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/v1/detect` | POST | Full detection, single document | API key |
| `/v1/detect/sentence` | POST | Sentence-level attribution | API key |
| `/v1/detect/batch` | POST | Async batch (up to 100 docs) | API key |
| `/v1/detect/batch/{task_id}` | GET | Poll batch result | API key |
| `/v1/explain/{text_id}` | GET | SHAP explanation for cached result | API key |
| `/v1/health` | GET | Liveness + readiness | None |
| `/v1/models` | GET | Loaded detector versions | API key |

- Rate limiting: 60 req/min per API key (Redis sliding window)
- Input validation: Pydantic v2 with max 10,000 token limit
- **Files**: `api/main.py`, `api/routes/`, `api/middleware.py`

#### 9.2 Response schema

```python
# api/schemas.py

class SignalResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    is_reliable: bool
    feature_values: dict[str, float] | None = None  # for explainability

class WatermarkResult(BaseModel):
    is_watermarked: bool
    watermark_type: str | None   # "KGW" | "ECC" | None
    confidence: float = Field(ge=0.0, le=1.0)
    p_value: float | None

class SentenceScore(BaseModel):
    index: int
    text: str
    score: float = Field(ge=0.0, le=1.0)
    label: Literal["AI", "HUMAN", "UNCERTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)

class DetectionResponse(BaseModel):
    text_id: str                                         # sha256 of input
    label: Literal["AI", "HUMAN", "UNCERTAIN", "WATERMARKED"]
    probability: float = Field(ge=0.0, le=1.0)          # P(AI)
    confidence_interval: tuple[float, float]             # conformal bounds
    prediction_set: list[str]                            # conformal set
    signals: dict[str, SignalResult]                     # per-signal breakdown
    watermark: WatermarkResult | None
    sentence_scores: list[SentenceScore] | None          # None unless requested
    language: str                                        # detected language
    domain: str | None
    token_count: int
    warnings: list[str]  # "text_too_short" | "non_native_english_possible" | ...
    model_version: str
    processing_time_ms: float
```

#### 9.3 Explainability module

Implement `api/explainability.py`:

- **Neural classifier**: SHAP `DeepExplainer` on top-K contributing token spans
- **Statistical signals**: Annotate which sentences triggered high Fast-DetectGPT curvature
- **Watermark**: Token-level green-list membership (KGW only)
- Run SHAP asynchronously; cache results by `text_id`
- Return HTML highlight + JSON `{token: str, shap_value: float}[]`
- **Files**: `api/explainability.py`

---

### Phase 10 — Evaluation Framework

> **Goal**: Reproducible evaluation against all major benchmarks.

#### 10.1 Benchmark harness

Implement `evaluation/harness.py`:

- Ingest: RAID, M4, BLUFF (58 languages), C-ReD (Chinese), custom held-out sets
- Run: every individual signal + full ensemble + individual commercial baselines (via API)
- Compute per-split:
  - AUROC, TPR@FPR=1%, F1 (threshold=0.5)
  - ECE (Expected Calibration Error)
  - Coverage @ α=0.10 (conformal validation)
  - Per-language, per-domain, per-generator breakdowns
- Export to MLflow as structured artifacts + CSV
- **Files**: `evaluation/harness.py`

#### 10.2 Attack suite

Implement `evaluation/attacks.py` — each attack implements `apply(text: str) → str`:

| Class | Method | Severity |
|-------|--------|---------|
| `DIPPERAttack` | Discourse-level paraphrase (level 40) | High |
| `BackTranslationAttack` | EN→FR→EN via NLLB-200 | Medium |
| `SynonymAttack` | `nlpaug` word substitution (p=0.2) | Low |
| `GPT4RewriteAttack` | "Rewrite this to sound more natural" | Very high |
| `ModelShiftAttack` | Train on GPT-3.5, test on GPT-4/Claude/Gemini | — |
| `DomainShiftAttack` | Train news, test academic/legal/creative | — |
| `TemperatureShiftAttack` | Generate at higher temperature | Low-medium |

- **Files**: `evaluation/attacks.py`

#### 10.3 Bias audit

Implement `evaluation/bias_audit.py`:

- Test groups: (1) native English (Reddit, news), (2) non-native English (ICLE, ESL essays), (3) translated-to-English text
- Compute FPR per group at threshold=0.5
- **Disparate impact check**: flag if any group FPR > 2× overall FPR
- Report: per-group ROC curves, FPR@threshold table
- **Files**: `evaluation/bias_audit.py`

#### 10.4 Artifact analysis (anti-overfitting check)

Implement `evaluation/artifact_analysis.py` — inspired by Pudasaini et al. (2026):

- Run SHAP on validation set; cluster feature importances by dataset
- Flag: if top-5 features differ significantly between RAID and M4, model may be fitting dataset artifacts
- **Files**: `evaluation/artifact_analysis.py`

---

### Phase 11 — Continuous Learning

> **Goal**: Detect and respond to model drift within 72 hours.

#### 11.1 Drift monitor

Implement `src/pipeline/drift_monitor.py`:

- Runs every 6 hours via cron
- Evaluates on rolling window: last 7 days of human text + freshly generated AI text
- Metrics: AUROC, TPR@FPR=1%, ECE
- **Alert threshold**: AUROC drops below 0.85 OR TPR@1% FPR drops below 0.80
- Alert: MLflow alert + webhook POST to configured endpoint
- **Files**: `src/pipeline/drift_monitor.py`

#### 11.2 New model detector

Implement `src/pipeline/new_model_collector.py`:

- Monitor: HuggingFace Hub RSS (filter by `text-generation`, downloads > 10k)
- Monitor: OpenAI / Anthropic / Google release pages via RSS
- When new major model detected: generate 1,000 labeled examples across domains and languages
- **Human approval gate**: staging directory + Slack/email notification before adding to training
- **Files**: `src/pipeline/new_model_collector.py`

#### 11.3 Incremental fine-tuning

Implement `src/pipeline/incremental_trainer.py`:

- LoRA fine-tuning on new model data ($r=16$, $\alpha=32$)
- Replay 20% of original training data to prevent catastrophic forgetting
- Accept new checkpoint only if: AUROC improves or holds on existing test sets (all generators)
- Log all runs to MLflow; tag checkpoint with `model_version = YYYYMMDD-{new_model_name}`
- **Files**: `src/pipeline/incremental_trainer.py`

---

### Phase 12 — Tests

> **Goal**: ≥ 85% coverage with meaningful tests at every layer.

#### 12.1 Unit tests — signals

`tests/signals/`:

```python
# For each signal extractor:
# - test_known_ai: AI text from GPT-4 should score > 0.7
# - test_known_human: Wikipedia text should score < 0.3
# - test_short_text: < 50 tokens must return is_reliable=False
# - test_multilingual: Spanish, Chinese, Arabic inputs must not crash
# - test_empty: empty string must return SignalResult with is_reliable=False
```

#### 12.2 Unit tests — ensemble

`tests/ensemble/test_calibration.py`:

```python
# test_conformal_coverage:
#   At alpha=0.10, ≥ 90% of test samples must have true label in prediction set
# test_bias_correction:
#   Non-native FPR must be ≤ 5% after correction
# test_uncertain_label:
#   When both AI and HUMAN are in prediction set, label must be UNCERTAIN
```

#### 12.3 Integration tests

`tests/integration/test_pipeline.py`:

```python
# test_default_performance:  AUROC > 0.90 on 50 AI + 50 human samples
# test_paraphrase_resistance: AUROC > 0.75 after DIPPER attack (level 40)
# test_nonnative_fpr:         FPR ≤ 5% on non-native English at threshold=0.5
# test_mixed_document:        Boundary F1 > 0.75 on 20 mixed docs
# test_watermark_override:    KGW-watermarked text always returns WATERMARKED
# test_short_text_gate:       60-token input returns INSUFFICIENT_TEXT
```

#### 12.4 API tests

`tests/api/test_endpoints.py`:

```python
# test_detect_valid_payload
# test_detect_invalid_payload (missing text, too long, wrong type)
# test_batch_100_documents
# test_rate_limiting (61st request in a minute → 429)
# test_auth (missing API key → 401)
# test_sentence_attribution_response_schema
```

---

## 6. Directory Structure

```
aegis/
├── pyproject.toml               # pinned deps; ruff, mypy, pytest config
├── .pre-commit-config.yaml
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── confidence_gate.py   # SignalResult dataclass; length thresholds
│   │   ├── dependency.py        # DependencyAI syntactic features (2026)
│   │   ├── stylometric.py       # TTR, burstiness, POS, readability
│   │   ├── topological.py       # PHD via ripser (NeurIPS 2024)
│   │   ├── statistical.py       # Fast-DetectGPT, Binoculars, LAPD
│   │   └── sentence_level.py    # Sliding-window per-sentence scores
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py        # xlm-roberta-large + 4 heads (MELD-style)
│   │   ├── trainer.py           # EMA teacher-student distillation
│   │   ├── adversarial_trainer.py  # RADAR-style adversarial loop
│   │   ├── bias_mitigation.py   # Gradient reversal + counterfactual augmentation
│   │   └── train.py             # Full training loop + MLflow logging
│   │
│   ├── watermark/
│   │   ├── __init__.py
│   │   ├── kgw_detector.py      # Kirchenbauer (2023) green/red list
│   │   ├── ecc_detector.py      # Error-correcting code detector (Chao, 2024)
│   │   └── aggregator.py        # WatermarkResult + override logic
│   │
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── meta_learner.py      # CatBoost stacking on all signal scores
│   │   ├── calibrator.py        # MAPIE split-conformal prediction
│   │   ├── bias_correction.py   # Per-language FPR equalization
│   │   └── dynamic_weights.py   # Short-text / attack / domain routing
│   │
│   ├── attribution/
│   │   ├── __init__.py
│   │   ├── sequence_labeler.py  # BIO tagger with CRF head
│   │   ├── changepoint.py       # PELT via ruptures library
│   │   └── report.py            # AttributionResult + HTML output
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── index.py             # pgvector HNSW — create / insert / search
│   │   ├── scorer.py            # cosine similarity → RetrievalResult
│   │   └── updater.py           # Daily cron: sample new AI text, embed, insert
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── drift_monitor.py     # Rolling AUROC + webhook alert
│       ├── new_model_collector.py  # HuggingFace Hub + API RSS monitoring
│       └── incremental_trainer.py  # LoRA fine-tuning with replay
│
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, lifespan, CORS, rate limit
│   ├── schemas.py               # Pydantic v2 request/response models
│   ├── middleware.py            # API key auth, rate limiting (Redis)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── detect.py            # /v1/detect, /v1/detect/sentence, /v1/detect/batch
│   │   └── admin.py             # /v1/health, /v1/models, /v1/explain
│   └── explainability.py        # SHAP + green-list annotations
│
├── evaluation/
│   ├── __init__.py
│   ├── harness.py               # RAID / M4 / BLUFF / C-ReD runner
│   ├── attacks.py               # Paraphrase, translation, shift attack classes
│   ├── bias_audit.py            # Disparate-impact FPR audit
│   └── artifact_analysis.py     # SHAP cross-dataset feature stability check
│
├── data/
│   ├── schema.py                # Arrow schema definitions
│   ├── registry.py              # Dataset versioning + split management
│   └── collectors/
│       ├── __init__.py
│       ├── human_corpus.py      # WikiText, CC-News, arXiv, BookCorpus
│       ├── ai_corpus.py         # Multi-API, multi-temperature, multi-language
│       └── attack_corpus.py     # DIPPER, NLLB back-translation, GPT-4 rewrite
│
└── tests/
    ├── conftest.py              # Fixtures: tiny labeled set, mock LLM clients
    ├── signals/
    │   ├── test_dependency.py
    │   ├── test_stylometric.py
    │   ├── test_topological.py
    │   └── test_statistical.py
    ├── ensemble/
    │   └── test_calibration.py
    ├── integration/
    │   └── test_pipeline.py
    └── api/
        └── test_endpoints.py
```

---

## 7. API Contract

### POST `/v1/detect`

**Request**:
```json
{
  "text": "string (50–10000 tokens)",
  "coverage": 0.90,
  "include_sentence_scores": false,
  "language_hint": "en"
}
```

**Response** (`DetectionResponse`):
```json
{
  "text_id": "sha256-hash",
  "label": "AI | HUMAN | UNCERTAIN | WATERMARKED",
  "probability": 0.84,
  "confidence_interval": [0.71, 0.94],
  "prediction_set": ["AI"],
  "signals": {
    "fast_detectgpt": {"score": 0.81, "confidence": 0.92, "is_reliable": true},
    "binoculars":     {"score": 0.78, "confidence": 0.88, "is_reliable": true},
    "lapd":           {"score": 0.85, "confidence": 0.91, "is_reliable": true},
    "phd":            {"score": 0.73, "confidence": 0.79, "is_reliable": true},
    "dependency":     {"score": 0.69, "confidence": 0.95, "is_reliable": true},
    "stylometric":    {"score": 0.77, "confidence": 0.90, "is_reliable": true},
    "neural_meld":    {"score": 0.89, "confidence": 0.94, "is_reliable": true},
    "retrieval":      {"score": 0.12, "confidence": 1.00, "is_reliable": true}
  },
  "watermark": null,
  "language": "en",
  "domain": "academic",
  "token_count": 312,
  "warnings": [],
  "model_version": "aegis-2.0.0-20260516",
  "processing_time_ms": 847
}
```

### Error responses

| Code | Condition |
|------|-----------|
| 400 | Text too long (> 10,000 tokens) or empty |
| 401 | Missing or invalid API key |
| 422 | Schema validation failure |
| 429 | Rate limit exceeded (60 req/min) |
| 503 | Model loading / GPU not available |

---

## 8. Testing Strategy

| Level | Coverage Target | What Is Tested |
|-------|:--------------:|----------------|
| Unit — signals | 90% | Each extractor: known AI/human, short text, multilingual, empty |
| Unit — ensemble | 90% | Conformal coverage guarantee, bias correction FPR bounds |
| Unit — attribution | 85% | BIO accuracy on synthetic mixed docs, change-point precision |
| Integration — pipeline | Full E2E | AUROC targets, paraphrase resistance, watermark override |
| Integration — API | All endpoints | Valid/invalid payloads, rate limiting, auth, batch |
| Evaluation — RAID | Benchmark | Per-signal and ensemble AUROC vs. published baselines |
| Evaluation — attacks | 7 attack types | Performance degradation curve |
| Evaluation — bias | 3 speaker groups | FPR disparate impact < 2× |
| Artifact check | Cross-dataset | SHAP feature stability across RAID vs. M4 |

---

## 9. Risk Register

| Risk | Severity | Likelihood | Mitigation |
|------|:--------:|:----------:|------------|
| Adversarial training collapse | High | Medium | KL regularization on paraphraser; early stopping |
| Multi-task gradient conflict | High | Medium | Homoscedastic uncertainty weights; separate optimizers |
| Conformal calibration distribution shift | High | Medium | Domain-stratified calibration; re-calibrate per domain |
| Dataset artifact overfitting | High | High | Phase 10.4 artifact analysis; cross-dataset SHAP monitoring |
| LAPD requires aligned+base model pair | Medium | Low | Maintain Mistral-7B base+instruct as default pair |
| PHD speed on long documents | Medium | Medium | 512-token chunking + average; GPU batch embedding |
| pgvector scale (millions of docs) | Medium | Low | HNSW (m=16, ef=64) + monthly pruning of low-quality entries |
| API cost for AI corpus generation | Medium | High | Sampling budgets; full result caching; throttle per model |
| LoRA catastrophic forgetting | Medium | Low | 20% old data replay; evaluate on full test suite before deploy |
| Non-native English bias | High | High | Phase 4.4 gradient reversal + Phase 6.3 FPR equalization |
| Watermark reverse-engineering | Low (ethical) | Low | Implement only published open algorithms |
| New models outpace detector | High | Certain | Phase 11 continuous learning pipeline with 72h target |

---

## 10. Weakness → Phase Mapping

| Weakness | Phase(s) That Address It |
|----------|--------------------------|
| Paraphrasing vulnerability | 4.2 (EMA distillation), 4.3 (adversarial loop), 2.1 (DependencyAI), 2.3 (PHD), 6.4 (reweight on suspected attack) |
| Bias against non-native English | 4.4 (gradient reversal), 6.3 (FPR correction), 10.3 (bias audit) |
| Short text unreliability | 2.4 (confidence gate), 6.4 (routing), 9.2 (warnings field) |
| Mixed human-AI documents | 7.1 (sequence labeler), 7.2 (change-point), 2.5 (sentence-level stats) |
| Model drift / new LLMs | 11.1 (drift monitor), 11.2 (new model collector), 11.3 (LoRA fine-tuning), 8 (retrieval — no retraining needed) |
| Domain generalization | 1.2 (domain-diverse training data), 4.1 (domain head), 6.4 (domain-aware routing), 10.4 (artifact analysis) |
| Theoretical impossibility | 5 (watermarking bypasses it), 6.2 (conformal UNCERTAIN label accepts uncertainty) |
| Multilingual gap | 4.1 (XLM-R backbone), 1.2–1.3 (7-language data), 6.3 (per-language calibration), 2.1 (multilingual dependency) |
| Dataset artifact overfitting | 10.4 (SHAP cross-dataset stability check) |
| Calibration and false positive harm | 6.2 (conformal prediction), 6.3 (bias correction), 9.2 (UNCERTAIN label) |
| Explainability gap | 4.6 approach (NOTAI.AI), 9.3 (SHAP per-token), watermark token annotations |
| Low-resource language gap | 2.1 (DependencyAI — 50+ languages), 10.1 (BLUFF evaluation) |

---

## 11. Success Criteria

- [ ] AUROC > **0.92** on RAID benchmark under default conditions
- [ ] AUROC > **0.78** on RAID benchmark after DIPPER paraphrase attack (level 40)
- [ ] AUROC > **0.85** on BLUFF benchmark for each of the 7 target languages
- [ ] False positive rate ≤ **5%** on non-native English human text at threshold = 0.5
- [ ] Sentence-level attribution **F1 > 0.78** on mixed-document test set
- [ ] **90% conformal coverage** verified on held-out calibration set
- [ ] Drift monitor detects > 10% AUROC drop within **72 hours** of new model release
- [ ] SHAP cross-dataset feature stability check passes (top features consistent across RAID and M4)
- [ ] All unit + integration tests pass at ≥ **85% coverage**
- [ ] API latency < **2 seconds** for documents up to 2,000 tokens (p95)
- [ ] API latency < **500 ms** for structural-only fast path (< 200 tokens, no GPU needed)
- [ ] Disparate impact FPR ratio < **2×** across all language groups

---

## 12. Research Grounding

Every design decision in AEGIS is grounded in published literature:

| AEGIS Component | Research Basis | Key Finding Used |
|-----------------|---------------|-----------------|
| MELD multi-task architecture | Li et al. 2026 (arXiv:2605.06903) | Multi-task supervision → better generalization; 99.9% TPR@1%FPR |
| Disentangled encoder (DRGD) | Pu et al. 2026 (arXiv:2604.13692) | Separating generator artifacts → +24.2% acc on unseen generators |
| LAPD / Alignment Imprint | Wu et al. 2026 (arXiv:2604.16923) | +45.82% relative improvement; RLHF imprint is a stable signal |
| DependencyAI syntactic signal | Ahmed & Hammond 2026 (arXiv:2602.15514) | Syntactic patterns alone are robust and multilingual |
| PHD topological signal | Tulchinskii et al. 2024 (NeurIPS) | Topology survives paraphrasing |
| Adversarial training loop | Hu et al. 2023 (RADAR) | Explicit paraphrase robustness |
| KGW watermark detection | Kirchenbauer et al. 2023 (ICML) | Provable false-positive bounds |
| ECC watermark detection | Chao et al. 2024 (arXiv:2406.10281) | Mathematical robustness to partial editing |
| Conformal prediction | Vovk et al.; MAPIE library | Coverage guarantees — honest uncertainty |
| Bias correction | Liang et al. 2023 | Non-native FPR problem is systematic and measurable |
| Artifact analysis | Pudasaini et al. 2026 (arXiv:2603.23146) | High benchmark scores ≠ robust detection |
| BLUFF multilingual evaluation | Lucas et al. 2026 (arXiv:2603.00634) | Low-resource languages are a critical gap |
| Retrieval defense | Krishna et al. 2024 (NeurIPS) | Retrieval is effective vs. paraphrase when text is known |
| Continuous learning | MELD-eval results | Models degrade sharply on newly released LLMs |
