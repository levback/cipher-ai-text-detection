# AEGIS — Adaptive Ensemble for AI-Generated Text Identification System
### Implementation Plan

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [System Architecture](#system-architecture)
4. [Implementation Steps](#implementation-steps)
   - [Phase 1 — Project Scaffold & Data Infrastructure](#phase-1--project-scaffold--data-infrastructure)
   - [Phase 2 — Signal Extraction Layer](#phase-2--signal-extraction-layer)
   - [Phase 3 — Neural Classifier (Adversarially Trained)](#phase-3--neural-classifier-adversarially-trained)
   - [Phase 4 — Watermark Detection Layer](#phase-4--watermark-detection-layer)
   - [Phase 5 — Calibrated Ensemble Fusion](#phase-5--calibrated-ensemble-fusion)
   - [Phase 6 — Token/Sentence-Level Attribution](#phase-6--tokensentence-level-attribution)
   - [Phase 7 — Retrieval-Augmented Detection (RAD)](#phase-7--retrieval-augmented-detection-rad)
   - [Phase 8 — Continuous Learning Pipeline](#phase-8--continuous-learning-pipeline)
   - [Phase 9 — Evaluation Framework](#phase-9--evaluation-framework)
   - [Phase 10 — REST API and Explainability](#phase-10--rest-api-and-explainability)
   - [Phase 11 — Testing Suite](#phase-11--testing-suite)
5. [Project Directory Structure](#project-directory-structure)
6. [Testing Strategy](#testing-strategy)
7. [Risks and Mitigations](#risks-and-mitigations)
8. [Weakness → Implementation Mapping](#weakness--implementation-mapping)
9. [Success Criteria](#success-criteria)

---

## Overview

Build a research-grade, production-ready AI text detection system that simultaneously addresses all seven identified weaknesses (paraphrasing vulnerability, bias, short-text failure, model drift, mixed-text, domain generalization, theoretical limits) and exploits all identified strengths (zero-shot generalization, topological robustness, provable watermark guarantees, calibrated uncertainty). The system is designed as a modular pipeline so each layer can be independently verified and replaced as the field advances.

---

## Requirements

| ID | Requirement |
|----|-------------|
| **R1** | Address paraphrasing vulnerability via adversarially-trained ensemble |
| **R2** | Eliminate bias against non-native English via bias-aware calibration |
| **R3** | Operate reliably on short texts (≥50 words) via signal-level confidence gating |
| **R4** | Handle mixed human-AI documents via sentence/token-level attribution |
| **R5** | Stay current as new LLMs emerge via continuous learning pipeline |
| **R6** | Support multilingual detection (English, Spanish, Chinese, Arabic, German, French, Japanese) |
| **R7** | Output calibrated probabilities with confidence intervals, not binary labels |
| **R8** | Detect code and structured text, not only prose |
| **R9** | Include watermark detection as a first-class signal |
| **R10** | Provide a reproducible, standardized evaluation framework against RAID/M4 benchmarks |

---

## System Architecture

### Architecture Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base encoder | `intfloat/multilingual-e5-large` + `xlm-roberta-large` | Multilingual from day one |
| Statistical scorer | Fast-DetectGPT + Binoculars | Best zero-shot AUROC, no retraining needed |
| Topological scorer | PHD via `ripser` | Robust to paraphrasing |
| Calibration | Conformal Prediction (split-conformal) | Coverage guarantees |
| Ensemble fusion | Learned stacking with CatBoost meta-learner | Handles feature correlations |
| Adversarial training | RADAR + DIPPER + back-translation | Multi-paraphraser coverage |
| Serving | FastAPI + Celery + Redis | Async inference with queuing |
| Storage | PostgreSQL + pgvector | Vector search for retrieval-augmented detection |
| Experiment tracking | MLflow | Reproducible training runs |

### High-Level Data Flow

```
Input Text
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Signal Extraction Layer (Phase 2)                              │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │  Statistical │  │  Topological  │  │     Stylometric      │ │
│  │  (PPL, Bino, │  │  (PHD/ripser) │  │  (TTR, POS, syntax) │ │
│  │  Fast-DGPT)  │  │               │  │                      │ │
│  └──────┬───────┘  └───────┬───────┘  └──────────┬───────────┘ │
│         └──────────────────┴────────────────────┘              │
│                   Confidence Gate                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴────────────────┐
          ▼                                ▼
┌─────────────────────┐       ┌────────────────────────┐
│  Neural Classifier  │       │  Watermark Detector    │
│  (xlm-roberta-large │       │  (KGW + SynthID-style) │
│   + adversarial)    │       │                        │
└──────────┬──────────┘       └────────────┬───────────┘
           └──────────┬───────────────────┘
                      ▼
          ┌───────────────────────┐
          │  Ensemble Fusion      │
          │  (CatBoost meta-learn │
          │   + conformal calib.) │
          └──────────┬────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────────┐
│  Sentence-Level  │  │  Retrieval-Augmented │
│  Attribution     │  │  Detection (pgvector)│
└────────┬─────────┘  └──────────┬───────────┘
         └─────────────┬─────────┘
                       ▼
              DetectionResponse
```

---

## Implementation Steps

### Phase 1 — Project Scaffold & Data Infrastructure

> **Enables**: all subsequent phases. No blocking dependencies.

#### 1.1 Repository structure and environment

- **Action**: Create `src/`, `api/`, `evaluation/`, `data/`, `tests/` directories with `__init__.py` files; write `pyproject.toml` with pinned deps (`torch`, `transformers`, `fastapi`, `scikit-learn`, `ripser`, `pgvector`, `mlflow`, `pytest`)
- **Files**: `pyproject.toml`, `src/__init__.py`, directory tree
- **Dependency**: None
- **Risk**: Low

#### 1.2 Data pipeline — human text corpora

- **Action**: Write `data/collectors/human_corpus.py` that fetches from open sources (WikiText-103, CC-News, arXiv abstracts, BookCorpus snippets) via HuggingFace datasets; stratify by domain (academic, news, creative, legal, code) and language (7 target languages); save as Arrow files
- **Files**: `data/collectors/human_corpus.py`, `data/schema.py`
- **Dependency**: 1.1
- **Risk**: Low

#### 1.3 Data pipeline — AI-generated text corpora

- **Action**: Write `data/collectors/ai_corpus.py`; call APIs (OpenAI, Anthropic, Mistral, Cohere) with a range of prompts, temperatures (0.3, 0.7, 1.0), and decoding strategies (greedy, nucleus, top-k); label with model name, temperature, domain; store in same Arrow schema
- **Files**: `data/collectors/ai_corpus.py`
- **Dependency**: 1.2
- **Risk**: Medium — API cost; mitigate by caching and sampling budgets

#### 1.4 Data pipeline — attack corpus

- **Action**: Write `data/collectors/attack_corpus.py`; apply DIPPER paraphrasing, back-translation (EN→FR→EN, EN→ZH→EN), GPT-4 rewriting at "humanize this" prompt, and simple character substitutions to AI corpus; label original vs. attacked
- **Files**: `data/collectors/attack_corpus.py`
- **Dependency**: 1.3
- **Risk**: Medium — DIPPER model is large; run on GPU

#### 1.5 Dataset registry and versioning

- **Action**: Write `data/registry.py` that reads all Arrow files, applies train/val/test splits stratified by domain+language+model, records dataset version hash in MLflow; ensure no test-set contamination
- **Files**: `data/registry.py`
- **Dependency**: 1.2–1.4
- **Risk**: Low

---

### Phase 2 — Signal Extraction Layer

> **Addresses**: R1 (partially), R3 (length-gating), R4 (sentence-level signals)

#### 2.1 Statistical signals module

- **Action**: Implement `src/signals/statistical.py` with three sub-detectors:
  - `perplexity_score(text, model_name)` — token-level log-probabilities using HuggingFace, returns mean and variance (burstiness)
  - `binoculars_score(text, model_a, model_b)` — CE/PPL ratio per Hans et al. 2024:
    $$\text{score} = \frac{H(x;\, M_1)}{\text{PPL}(x;\, M_2)}$$
  - `fast_detectgpt_score(text, model_name)` — conditional probability curvature per Bao et al. 2023:
    $$\hat{d}(x) \approx \frac{1}{N}\sum_{t=1}^{N}\left(\log p_\theta(x_t \mid x_{<t}) - \sum_{v \in V} p_\theta(v \mid x_{<t})\log p_\theta(v \mid x_{<t})\right)$$
- **Files**: `src/signals/statistical.py`
- **Dependency**: 1.1
- **Risk**: Medium — requires logit access; implement fallback for API-only models

#### 2.2 Topological signal module

- **Action**: Implement `src/signals/topological.py` with `phd_score(text, embedding_model)`:
  1. Embed sentences using `multilingual-e5-large`
  2. Compute persistent homology via `ripser.ripser(embeddings)`
  3. Extract intrinsic dimension estimate as detection score (AI text has lower PHD)
  4. Cache embeddings keyed by text hash
- **Files**: `src/signals/topological.py`
- **Dependency**: 1.1
- **Risk**: Medium — `ripser` is slow for long texts; implement chunk-and-aggregate strategy

#### 2.3 Stylometric signal module

- **Action**: Implement `src/signals/stylometric.py` with features:
  - Type-token ratio (TTR), Hapax legomena ratio
  - Mean sentence length, sentence length variance
  - Part-of-speech distribution (via `spacy`)
  - Punctuation density and patterns
  - Named entity density
  - Syntactic tree depth (via `spacy` dependency parser)
  - Readability scores (Flesch-Kincaid, Gunning Fog)
  - Code-specific features: cyclomatic complexity proxy, identifier naming entropy (when `domain=code`)
- **Files**: `src/signals/stylometric.py`
- **Dependency**: 1.1
- **Risk**: Low — all features are language-dependent; add per-language spacy model routing

#### 2.4 Length-gating and signal confidence module

- **Action**: Implement `src/signals/confidence_gate.py`:
  - Defines minimum token thresholds per signal (statistical: 100 tokens, topological: 5 sentences, stylometric: 50 tokens)
  - Returns `SignalResult(score, confidence, is_reliable)` dataclass per signal
  - When text is too short, marks signal as `unreliable` rather than returning a garbage score
- **Files**: `src/signals/confidence_gate.py`
- **Dependency**: 2.1–2.3
- **Risk**: Low

#### 2.5 Sentence-level signal computation

- **Action**: Extend `src/signals/sentence_level.py` to compute statistical signals for each sentence independently using a sliding window; return `List[SentenceScore]` with position, score, confidence; this is the foundation for mixed-document attribution
- **Files**: `src/signals/sentence_level.py`
- **Dependency**: 2.1, 2.4
- **Risk**: Medium — very slow for long documents; use batched inference

---

### Phase 3 — Neural Classifier (Adversarially Trained)

> **Addresses**: R1 (paraphrasing), R5 (model drift via retraining), R6 (multilingual)

#### 3.1 Base classifier architecture

- **Action**: Implement `src/models/classifier.py`:
  - **Backbone**: `xlm-roberta-large` (multilingual, 560 M params)
  - **Head**: multi-task — (a) binary AI/human classification, (b) source model family prediction (GPT, Claude, Gemini, Llama, Mistral), (c) domain classification
  - **Multi-task loss**:
    $$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{detect}} + \lambda_2 \mathcal{L}_{\text{source}} + \lambda_3 \mathcal{L}_{\text{domain}}$$
  - Shared encoder improves generalization
- **Files**: `src/models/classifier.py`
- **Dependency**: 1.5
- **Risk**: High — multi-task training is unstable; use separate learning rates per head and gradient surgery

#### 3.2 Adversarial training loop (RADAR-style)

- **Action**: Implement `src/models/adversarial_trainer.py`:
  1. Train paraphraser (LoRA-fine-tuned Mistral-7B) to maximize detector loss
  2. Train detector on paraphrased outputs to minimize detection loss
  3. Iterate for 3 rounds with different paraphrasers each round: DIPPER, back-translation (via NLLB-200), ChatGPT-4o rewrite
  4. Freeze paraphraser weights; only classifier continues updating after round 3
- **Files**: `src/models/adversarial_trainer.py`
- **Dependency**: 3.1, 1.4
- **Risk**: High — adversarial loop can collapse; add KL divergence regularization to prevent paraphraser from destroying meaning

#### 3.3 Bias mitigation for non-native English

- **Action**: Implement `src/models/bias_mitigation.py`:
  - Identify likely non-native English samples using a language proficiency proxy (CEFR-level classifier on writing)
  - Apply counterfactual data augmentation: pair human non-native texts with AI texts written in "simple English"
  - Add a protected attribute adversarial loss term: penalize the classifier for using native-language-correlated features
  - **Method**: gradient reversal on the native-speaker prediction head
- **Files**: `src/models/bias_mitigation.py`
- **Dependency**: 3.1
- **Risk**: High — gradient reversal can destabilize training; lower its learning rate and add warm-up

#### 3.4 Training infrastructure and MLflow logging

- **Action**: Write `src/models/train.py` with full training loop: data loading, optimizer (AdamW with cosine LR schedule), mixed-precision (bf16), gradient clipping, early stopping on validation AUROC, MLflow experiment logging (params, metrics, model artifact)
- **Files**: `src/models/train.py`
- **Dependency**: 3.1–3.3
- **Risk**: Low

---

### Phase 4 — Watermark Detection Layer

> **Addresses**: R9, provides provable guarantees when watermarks are present

#### 4.1 KGW watermark detector

- **Action**: Implement `src/watermark/kgw_detector.py`:
  - Hash-based green/red list reconstruction: for a given seed and vocabulary, recompute green list for each position
  - **Score** (z-statistic under null):
    $$z = \frac{|\{t : x_t \in G_t\}| - N/2}{\sqrt{N/4}}$$
  - P-value computation and decision threshold (default: $p < 0.01$)
  - Blind detection mode (no generator model needed; enumerate plausible hash seeds)
- **Files**: `src/watermark/kgw_detector.py`
- **Dependency**: 1.1
- **Risk**: Medium — blind detection requires brute-forcing seed space; limit to known common seeds

#### 4.2 SynthID-inspired tournament detector

- **Action**: Implement `src/watermark/tournament_detector.py`:
  - Reproduce the scoring function described in the SynthID-Text Nature paper (public algorithm)
  - Support multiple tournament configurations
  - Return watermark confidence score (0–1)
- **Files**: `src/watermark/tournament_detector.py`
- **Dependency**: 1.1
- **Risk**: Medium — SynthID's exact implementation is proprietary; implement the public algorithm, not the proprietary variant

#### 4.3 Watermark signal aggregation

- **Action**: Implement `src/watermark/aggregator.py`:
  - Run all watermark detectors
  - Return `WatermarkResult(is_watermarked, watermark_type, confidence, p_value)`
  - If any watermark is detected with high confidence, it overrides ensemble output with a `"WATERMARKED"` flag
- **Files**: `src/watermark/aggregator.py`
- **Dependency**: 4.1–4.2
- **Risk**: Low

---

### Phase 5 — Calibrated Ensemble Fusion

> **Addresses**: R2 (bias), R3 (short text), R7 (calibrated uncertainty)

#### 5.1 Meta-learner (ensemble fusion)

- **Action**: Implement `src/ensemble/meta_learner.py`:
  - **Input features**: all signal scores + confidences + text metadata (length, language, domain)
  - **Model**: CatBoost classifier (handles missing features gracefully when signals are unreliable)
  - Train on held-out calibration set (not training set — prevents leakage)
  - Output: raw ensemble score before calibration
- **Files**: `src/ensemble/meta_learner.py`
- **Dependency**: 2.4, 3.1, 4.3
- **Risk**: Medium — signal leakage during meta-learner training is a common mistake; enforce strict data split

#### 5.2 Conformal prediction calibrator

- **Action**: Implement `src/ensemble/calibrator.py`:
  - Apply split-conformal prediction: compute nonconformity scores on a calibration set
  - For a user-specified coverage level $1 - \alpha$ (e.g., 90%), return a prediction set $C(x) \in \bigl\{\{0\}, \{1\}, \{0, 1\}\bigr\}$, where $\{0,1\}$ means "uncertain"
  - Compute calibrated confidence intervals: $[\hat{p} - \delta_\alpha,\; \hat{p} + \delta_\alpha]$
  - The width of the interval serves as the uncertainty signal: wide interval = "don't know"
- **Files**: `src/ensemble/calibrator.py`
- **Dependency**: 5.1
- **Risk**: Medium — conformal prediction requires careful calibration set construction; ensure it is i.i.d. with test distribution

#### 5.3 Bias-aware score correction

- **Action**: Implement `src/ensemble/bias_correction.py`:
  - Detect likely non-native English text using the proficiency proxy from 3.3
  - Apply language-specific score adjustment: shift scores down by a learned offset for non-native text
  - Compute per-language false positive rate on calibration set; correct until FPR ≤ 5% per language group
- **Files**: `src/ensemble/bias_correction.py`
- **Dependency**: 5.2, 3.3
- **Risk**: Medium — correction may over-adjust; bound corrections with a maximum shift

#### 5.4 Dynamic signal weighting

- **Action**: Implement `src/ensemble/dynamic_weights.py` with routing rules:
  - **Short text** (< 100 tokens): up-weight stylometric and neural classifier, down-weight statistical
  - **Suspected paraphrase attack** (low statistical score + high neural score): up-weight PHD and stylometric
  - **Code/structured text**: up-weight code-specific stylometric features, disable prose perplexity
  - **Multilingual non-English**: up-weight multilingual neural classifier, down-weight English-centric statistical
- **Files**: `src/ensemble/dynamic_weights.py`
- **Dependency**: 5.1
- **Risk**: Medium — heuristics for attack detection may introduce feedback loops; log all weight decisions

---

### Phase 6 — Token/Sentence-Level Attribution

> **Addresses**: R4 (mixed human-AI documents)

#### 6.1 Sequence labeling model

- **Action**: Implement `src/attribution/sequence_labeler.py`:
  - Input: document split into sentences
  - Model: `xlm-roberta-base` with a token-classification head (BIO tagging: `B-AI`, `I-AI`, `O-human`)
  - Training data: mixed documents — concatenate human and AI paragraphs with known boundaries
  - Loss: CRF loss on top of token logits (enforces sequential consistency)
- **Files**: `src/attribution/sequence_labeler.py`
- **Dependency**: 1.5, 3.1
- **Risk**: High — mixed training data requires careful boundary annotation; create synthetic mixed documents from corpus by programmatic concatenation

#### 6.2 Change-point detection

- **Action**: Implement `src/attribution/changepoint.py`:
  - Compute per-sentence statistical scores (from 2.5)
  - Apply `ruptures` library PELT or BOCPD algorithm to detect abrupt changes in score series
  - Change-point positions are candidate human-AI boundaries
  - Fuse with sequence labeler predictions via majority vote
- **Files**: `src/attribution/changepoint.py`
- **Dependency**: 2.5
- **Risk**: Low

#### 6.3 Attribution report generator

- **Action**: Implement `src/attribution/report.py`:
  - Produce highlighted HTML output marking AI-attributed spans
  - Include per-sentence confidence scores
  - Return structured `AttributionResult` with: overall score, sentence-level scores, boundary positions, confidence map
- **Files**: `src/attribution/report.py`
- **Dependency**: 6.1, 6.2
- **Risk**: Low

---

### Phase 7 — Retrieval-Augmented Detection (RAD)

> **Addresses**: R5 (model drift — detects known outputs without retraining)

#### 7.1 Vector index of known AI-generated text

- **Action**: Implement `src/retrieval/index.py`:
  - Embed documents into `multilingual-e5-large` vector space
  - Store in PostgreSQL with `pgvector` extension
  - **Schema**: `(id, text_hash, embedding, model_name, timestamp, domain)`
  - **Index**: HNSW for approximate nearest-neighbor search (`m=16`, `ef_construction=64`)
  - Populate initially from AI corpus (Phase 1.3)
- **Files**: `src/retrieval/index.py`
- **Dependency**: 1.3, 1.1
- **Risk**: Medium — pgvector HNSW at scale requires tuning

#### 7.2 Retrieval-based scorer

- **Action**: Implement `src/retrieval/scorer.py`:
  - For a query text, retrieve top-5 nearest neighbors from index
  - $\text{score} = 1 - \min_i\bigl(\text{cosine\_distance}(q,\, n_i)\bigr)$
  - High similarity to known AI text → flag as AI regardless of classifier output
  - Return `RetrievalResult(max_similarity, top_matches, model_source)`
  - Threshold at cosine similarity > 0.97 to avoid false positives
- **Files**: `src/retrieval/scorer.py`
- **Dependency**: 7.1
- **Risk**: Low

#### 7.3 Index update pipeline

- **Action**: Implement `src/retrieval/updater.py`:
  - Cron job that periodically samples from major LLM APIs with diverse prompts
  - Embeds and inserts new samples into pgvector index
  - Deduplicates by text hash
- **Files**: `src/retrieval/updater.py`
- **Dependency**: 7.1
- **Risk**: Low

---

### Phase 8 — Continuous Learning Pipeline

> **Addresses**: R5 (model drift as new LLMs are released)

#### 8.1 Model drift monitor

- **Action**: Implement `src/pipeline/drift_monitor.py`:
  - Periodically run classifier on a held-out human corpus and a freshly generated AI corpus
  - Compute AUROC, precision, recall on this rolling window
  - Alert (log to MLflow + send webhook notification) when AUROC drops below 0.85
- **Files**: `src/pipeline/drift_monitor.py`
- **Dependency**: 3.1, 7.3
- **Risk**: Low

#### 8.2 Automated data generation for new models

- **Action**: Implement `src/pipeline/new_model_collector.py`:
  - Monitor HuggingFace model hub and major API changelogs via RSS
  - When a new major model is detected, auto-generate a labeled corpus (1,000 documents across domains and languages)
  - Stage data for human review before adding to training set
- **Files**: `src/pipeline/new_model_collector.py`
- **Dependency**: 1.3
- **Risk**: Medium — automated detection of "new major model" is heuristic; require human approval gate before retraining

#### 8.3 Incremental fine-tuning pipeline

- **Action**: Implement `src/pipeline/incremental_trainer.py`:
  - Use LoRA for parameter-efficient fine-tuning on new model data (avoids catastrophic forgetting); start with `r=16`, `alpha=32`
  - Evaluate on full validation set before and after
  - Accept new checkpoint only if AUROC improves or holds on existing models
  - Replay 20% of old training data during each incremental run to prevent forgetting
  - Log all runs to MLflow; tag with model version
- **Files**: `src/pipeline/incremental_trainer.py`
- **Dependency**: 3.4, 8.1, 8.2
- **Risk**: Medium — LoRA rank must be tuned

---

### Phase 9 — Evaluation Framework

> **Addresses**: the standardized evaluation gap; validates all other phases

#### 9.1 RAID-compatible evaluation harness

- **Action**: Implement `evaluation/harness.py`:
  - Ingest RAID and M4 benchmark datasets in their standard formats
  - Run all detectors (individual signals + full ensemble) on all splits
  - Compute: AUROC, TPR@FPR=1%, F1, calibration error (ECE), per-domain and per-language breakdowns
  - Save results to MLflow as structured artifacts
- **Files**: `evaluation/harness.py`
- **Dependency**: Phases 2–5
- **Risk**: Low

#### 9.2 Attack protocol suite

- **Action**: Implement `evaluation/attacks.py` — each attack class implements `apply(text: str) -> str`:

  | Attack class | Description |
  |---|---|
  | `ParaphraseAttack` | DIPPER, T5, GPT-4 rewrite |
  | `TranslationAttack` | Round-trip via NLLB-200 |
  | `ModelShiftAttack` | Train on GPT-3.5, test on GPT-4/Claude/Gemini |
  | `DomainShiftAttack` | Train on news, test on academic/legal/creative |
  | `TemperatureShiftAttack` | Generate with different temperatures |

- **Files**: `evaluation/attacks.py`
- **Dependency**: 1.4
- **Risk**: Low

#### 9.3 Bias audit

- **Action**: Implement `evaluation/bias_audit.py`:
  - Construct a test set of human text from: native English speakers (Reddit, news), non-native English (ESL essays, L2 academic writing), non-English (translated to English)
  - Compute FPR per group
  - Flag if any group's FPR exceeds 2× the overall FPR (disparate impact criterion)
- **Files**: `evaluation/bias_audit.py`
- **Dependency**: 9.1
- **Risk**: Low

---

### Phase 10 — REST API and Explainability

> Makes the system usable; exposes all capabilities

#### 10.1 FastAPI application

- **Action**: Implement `api/main.py` with endpoints:

  | Endpoint | Method | Description |
  |---|---|---|
  | `/detect` | POST | Full detection with ensemble |
  | `/detect/sentence` | POST | Sentence-level attribution |
  | `/detect/batch` | POST | Async batch processing via Celery |
  | `/health` | GET | Liveness + readiness probe |
  | `/models` | GET | List loaded detector versions |

- **Files**: `api/main.py`, `api/schemas.py`, `api/routes/`
- **Dependency**: Phases 2–7
- **Risk**: Low

#### 10.2 Response schema design

- **Action**: Define `api/schemas.py` with Pydantic v2 models:

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class SignalResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    is_reliable: bool

class SentenceScore(BaseModel):
    index: int
    text: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    label: Literal["AI", "HUMAN", "UNCERTAIN"]

class WatermarkResult(BaseModel):
    is_watermarked: bool
    watermark_type: str | None   # "KGW", "SynthID", None
    confidence: float = Field(ge=0.0, le=1.0)
    p_value: float | None

class DetectionResponse(BaseModel):
    text_id: str
    label: Literal["AI", "HUMAN", "UNCERTAIN", "WATERMARKED"]
    probability: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]   # conformal prediction bounds
    signals: dict[str, SignalResult]
    watermark: WatermarkResult | None
    sentence_scores: list[SentenceScore] | None
    language: str
    domain: str | None
    warnings: list[str]    # e.g. "text_too_short", "non_native_english_possible"
    model_version: str
    processing_time_ms: float

class AttributionResponse(BaseModel):
    text_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    sentence_scores: list[SentenceScore]
    boundary_positions: list[int]   # sentence indices of human-AI boundaries
    html_report: str                # highlighted HTML
    model_version: str

class BatchDetectRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)
    coverage: float = Field(default=0.9, ge=0.5, le=0.99)

class BatchStatusResponse(BaseModel):
    task_id: str
    status: Literal["queued", "processing", "done", "failed"]
    results: list[DetectionResponse] | None
    error: str | None
```

- **Files**: `api/schemas.py`
- **Dependency**: 5.2
- **Risk**: Low

#### 10.3 Explainability module

- **Action**: Implement `api/explainability.py`:
  - For neural classifier: SHAP values on top contributing tokens
  - For statistical signals: annotate which sentences triggered which signals
  - For watermark: indicate which tokens are in the green list
  - Produce HTML highlight report and JSON structured explanation
  - Cache SHAP computations and run asynchronously (SHAP is slow on large texts)
- **Files**: `api/explainability.py`
- **Dependency**: 6.3, 10.1
- **Risk**: Medium

---

### Phase 11 — Testing Suite

> Ensures correctness at every level

#### 11.1 Unit tests — signals

- **Action**: Write `tests/signals/` with pytest tests for every signal extractor:
  - Test on known AI text (should score high) and human text (should score low)
  - Test length-gating: short text must return `is_reliable=False`
  - Test multilingual input (5 languages minimum)
- **Files**: `tests/signals/test_statistical.py`, `test_topological.py`, `test_stylometric.py`
- **Dependency**: Phase 2
- **Risk**: Low

#### 11.2 Unit tests — calibration

- **Action**: Write `tests/ensemble/test_calibration.py`:
  - Verify coverage guarantee: at $1 - \alpha = 0.9$, ≥90% of test samples must have true label in prediction set
  - Verify bias correction lowers non-native FPR to ≤ 5%
- **Files**: `tests/ensemble/test_calibration.py`
- **Dependency**: Phase 5
- **Risk**: Low

#### 11.3 Integration tests — full pipeline

- **Action**: Write `tests/integration/test_pipeline.py`:
  - End-to-end test with 50 known AI texts and 50 known human texts
  - Assert AUROC > 0.90 under default conditions
  - Assert AUROC > 0.75 after paraphrase attack
  - Assert FPR for non-native English ≤ 5% at `threshold=0.5`
- **Files**: `tests/integration/test_pipeline.py`
- **Dependency**: Phases 2–5
- **Risk**: Low

#### 11.4 API tests

- **Action**: Write `tests/api/test_endpoints.py` using `httpx.AsyncClient`:
  - Test all endpoints with valid and invalid payloads
  - Test rate limiting, auth, and error responses
  - Test batch endpoint with 100-document payload
- **Files**: `tests/api/test_endpoints.py`
- **Dependency**: Phase 10
- **Risk**: Low

---

## Project Directory Structure

```
aegis/
├── pyproject.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── statistical.py       # Binoculars, Fast-DetectGPT, perplexity
│   │   ├── topological.py       # PHD via ripser
│   │   ├── stylometric.py       # TTR, POS, readability, code features
│   │   ├── confidence_gate.py   # Length-gating, SignalResult dataclass
│   │   └── sentence_level.py    # Sliding-window sentence scores
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py        # xlm-roberta-large multi-task head
│   │   ├── adversarial_trainer.py  # RADAR-style adversarial loop
│   │   ├── bias_mitigation.py   # Gradient reversal, counterfactual augmentation
│   │   └── train.py             # Full training loop + MLflow logging
│   ├── watermark/
│   │   ├── __init__.py
│   │   ├── kgw_detector.py      # Kirchenbauer et al. green/red list
│   │   ├── tournament_detector.py  # SynthID-style public algorithm
│   │   └── aggregator.py        # WatermarkResult + override logic
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── meta_learner.py      # CatBoost stacking
│   │   ├── calibrator.py        # Split-conformal prediction
│   │   ├── bias_correction.py   # Per-language FPR correction
│   │   └── dynamic_weights.py   # Short-text / attack / domain routing
│   ├── attribution/
│   │   ├── __init__.py
│   │   ├── sequence_labeler.py  # BIO tagger with CRF head
│   │   ├── changepoint.py       # PELT / BOCPD via ruptures
│   │   └── report.py            # AttributionResult + HTML output
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── index.py             # pgvector HNSW index
│   │   ├── scorer.py            # Nearest-neighbor similarity scorer
│   │   └── updater.py           # Cron-based index refresh
│   └── pipeline/
│       ├── __init__.py
│       ├── drift_monitor.py     # Rolling AUROC + webhook alert
│       ├── new_model_collector.py  # RSS-based new model detection
│       └── incremental_trainer.py  # LoRA fine-tuning with replay
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, startup, middleware
│   ├── schemas.py               # Pydantic v2 request/response models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── detect.py            # /detect, /detect/sentence, /detect/batch
│   │   └── admin.py             # /health, /models
│   └── explainability.py        # SHAP + green-list annotations
├── evaluation/
│   ├── __init__.py
│   ├── harness.py               # RAID/M4 evaluation runner
│   ├── attacks.py               # Paraphrase, translation, shift attacks
│   └── bias_audit.py            # Disparate-impact FPR audit
├── data/
│   ├── schema.py                # Arrow schema definitions
│   ├── registry.py              # Dataset versioning + split management
│   └── collectors/
│       ├── __init__.py
│       ├── human_corpus.py      # WikiText-103, CC-News, arXiv, BookCorpus
│       ├── ai_corpus.py         # Multi-API, multi-temperature generation
│       └── attack_corpus.py     # DIPPER, back-translation, GPT-4 rewrite
└── tests/
    ├── conftest.py              # Shared fixtures
    ├── signals/
    │   ├── test_statistical.py
    │   ├── test_topological.py
    │   └── test_stylometric.py
    ├── ensemble/
    │   └── test_calibration.py
    ├── integration/
    │   └── test_pipeline.py
    └── api/
        └── test_endpoints.py
```

---

## Testing Strategy

| Level | Coverage Target | What Is Tested |
|-------|:--------------:|----------------|
| Unit — signals | 90% | Each extractor, edge cases, empty/short text, multilingual input |
| Unit — ensemble | 90% | Calibration coverage guarantee, bias correction bounds |
| Unit — attribution | 85% | BIO labeling accuracy, change-point detection precision |
| Integration — pipeline | Full E2E | AUROC >0.90 default, >0.75 post-paraphrase, FPR bias <5% |
| Integration — API | All endpoints | Valid/invalid payloads, async batch, schema validation |
| Evaluation — RAID | Benchmark | Per-method, per-domain, per-language AUROC table |
| Evaluation — attacks | 6 attack types | Performance degradation curve per attack |
| Evaluation — bias | 3 speaker groups | Disparate impact < 2× |

---

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| Adversarial training instability | High | KL regularization + lower adversary learning rate + early stop |
| Multi-task gradient conflict | High | Gradient surgery / separate optimizers per head |
| Conformal calibration distribution shift | Medium | Domain-stratified calibration set; re-calibrate per domain |
| pgvector scale at millions of documents | Medium | HNSW index (`m=16`, `ef_construction=64`) + periodic pruning |
| API cost for AI corpus generation | Medium | Sampling budget per model per run; cache all results |
| PHD speed on long documents | Medium | Chunk into 512-token windows, average PHD scores |
| LoRA catastrophic forgetting | Medium | Replay 20% of old training data during incremental fine-tuning |
| Proprietary watermark reverse-engineering | Low (ethical) | Only implement published algorithms; do not attempt to clone SynthID's private key |

---

## Weakness → Implementation Mapping

| Weakness | Phase(s) That Address It |
|----------|--------------------------|
| Paraphrasing vulnerability | 3.2 (adversarial training), 2.2 (PHD is paraphrase-robust), 5.4 (reweight toward robust signals) |
| Bias against non-native English | 3.3 (gradient reversal), 5.3 (score correction), 9.3 (bias audit) |
| Short text unreliability | 2.4 (confidence gating), 5.1 (CatBoost handles missing features), 10.2 (`bias_warning` flag) |
| Mixed human-AI documents | 6.1 (sequence labeler), 6.2 (change-point detection), 2.5 (sentence-level signals) |
| Model drift | 8.1 (drift monitor), 8.2 (auto-collector), 8.3 (LoRA fine-tuning), 7.2 (retrieval doesn't need retraining) |
| Domain generalization | 1.2 (domain-diverse data), 3.1 (domain classification head), 5.4 (domain-aware signal weights) |
| Theoretical impossibility | 4.1–4.3 (watermarking bypasses it), 5.2 (conformal `UNCERTAIN` label accepts uncertainty) |
| Multilingual gap | 3.1 (XLM-R backbone), 1.2–1.3 (7-language data), 5.3 (per-language calibration) |
| Code/structured text gap | 2.3 (code-specific stylometric features), 5.4 (structured text routing) |
| Calibration and false positive harm | 5.2 (conformal prediction), 5.3 (bias correction), 10.2 (`UNCERTAIN` label) |

---

## Success Criteria

- [ ] AUROC > **0.92** on RAID benchmark under default conditions
- [ ] AUROC > **0.78** on RAID benchmark under paraphrase attack (DIPPER)
- [ ] False positive rate ≤ **5%** on non-native English human text at `threshold=0.5`
- [ ] Sentence-level attribution **F1 > 0.80** on mixed-document test set
- [ ] **90% conformal coverage** guarantee verified on held-out calibration set
- [ ] System handles **7 languages** with AUROC > 0.85 on each
- [ ] Drift monitor detects >10% AUROC drop **within 24 hours** of new model release
- [ ] All unit and integration tests passing (≥ **85% coverage**)
- [ ] API latency **< 2 seconds** for documents up to 2,000 words
