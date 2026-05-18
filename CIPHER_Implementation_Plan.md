# CIPHER — Implementation Plan
## Cognitive Information Pattern Hierarchy for Entropy-based Recognition

**Version**: 1.0  
**Date**: May 2026  
**Basis**: Competitive analysis of GPTZero (arXiv:2602.13042), Turnitin (3-model ensemble), iThenticate, and the AI text detection patent landscape (2022–2026)

---

## Table of Contents

1. [Vision and Positioning](#1-vision-and-positioning)
2. [Differentiation Matrix](#2-differentiation-matrix)
3. [Core Detection Hypothesis](#3-core-detection-hypothesis)
4. [System Architecture](#4-system-architecture)
5. [Mathematical Foundations](#5-mathematical-foundations)
6. [Technology Stack](#6-technology-stack)
7. [Implementation Phases](#7-implementation-phases)
8. [Directory Structure](#8-directory-structure)
9. [API Contract](#9-api-contract)
10. [Testing Strategy](#10-testing-strategy)
11. [Risk Register](#11-risk-register)
12. [Success Criteria](#12-success-criteria)

---

## 1. Vision and Positioning

CIPHER is a production-grade AI text detection system built on a **fundamentally different theoretical foundation** from every commercially deployed or patented system as of May 2026.

### The Novel Thesis

Where existing systems ask *"does this text look like what an LLM would generate?"* (a model-centric question requiring an LLM at inference time), CIPHER asks a different question:

> **"Does this text exhibit the structural and informational signatures of human cognition — bounded memory, strategic uncertainty, discourse improvisation — or the signatures of machine generation — uniform predictability, compression regularity, discourse smoothness?"**

This is a **process signature** approach. It detects the *generative process* rather than matching the output against a known model's distribution. This makes CIPHER:

- **Model-agnostic**: works on output from any LLM, present and future, without needing access to that LLM
- **LLM-free at inference time**: primary detection signals (Layers 1–3) require no neural language model calls
- **Intrinsically robust to paraphrasing**: structural signatures survive paraphrase attacks that defeat perplexity-based and n-gram-based detectors
- **Legally distinct**: does not use perplexity scoring (GPTZero), fine-tuned binary classification (Turnitin), or truncation+regeneration+n-gram overlap (NEC US12596869B2)

---

## 2. Differentiation Matrix

| Mechanism | GPTZero | Turnitin | NEC US12596869B2 | CIPHER |
|-----------|---------|----------|------------------|--------|
| Perplexity (PPL) scoring | ✅ Primary signal | Indirect | ✅ WScore uses log-prob | ❌ Not used |
| Burstiness (variance of sentence PPL) | ✅ Primary signal | ❌ | ❌ | ❌ Not used |
| Log-probability curvature (DetectGPT) | ✅ Layer 1 | ❌ | ❌ | ❌ Not used |
| Truncation → LLM regeneration | ❌ | ❌ | ✅ Core claim | ❌ Not used |
| N-gram overlap with regenerations | ❌ | ❌ | ✅ BScore (Claims 1, 5) | ❌ Not used |
| Fine-tuned binary classifier | ✅ Layer 2 | ✅ Primary model | ❌ | ❌ Not used |
| Three-model ensemble (writing/paraphrase/bypasser) | ❌ | ✅ | ❌ | ❌ Not used |
| **Lempel-Ziv / compression-theoretic analysis** | ❌ | ❌ | ❌ | ✅ **Layer 1 primary** |
| **Cognitive load fingerprinting** | ❌ | ❌ | ❌ | ✅ **Layer 2 primary** |
| **Discourse topology (co-reference graph)** | ❌ | ❌ | ❌ | ✅ **Layer 3 primary** |
| **Contrastive metric learning (manifold distance)** | ❌ | ❌ | ❌ | ✅ **Layer 4 primary** |
| **Isotonic regression + conformal calibration** | ❌ | ❌ | ❌ | ✅ **Layer 5** |

CIPHER avoids every patented mechanism while drawing equivalent or superior detection signal from an entirely different signal family.

---

## 3. Core Detection Hypothesis

### 3.1 The Human Writing Process Model

Human writing is the output of a **bounded cognitive process** with measurable properties:

1. **Working memory limits** constrain sentence complexity and introduce variance. Humans do not produce uniformly structured sentences.
2. **Semantic planning is non-linear**: humans revise intent mid-paragraph, introduce tangential thoughts, and self-correct — producing measurable semantic drift.
3. **Vocabulary deployment is bursty**: new words cluster around conceptual "arrivals" rather than diffusing uniformly.
4. **Discourse structure is improvised**: human texts have irregular co-reference patterns, abrupt topic transitions, and uneven paragraph structure.
5. **Epistemic marking is habitual**: humans signal uncertainty linguistically at rates incompatible with LLM defaults.

### 3.2 The Machine Writing Process Model

LLM-generated text exhibits a complementary signature:

1. **Token predictability is uniform across positions**: next-token probability remains high throughout the document.
2. **Semantic coherence is unnaturally smooth**: cosine similarity between consecutive sentences remains high with low variance — "too coherent."
3. **Compression regularity**: AI text is more locally compressible (repetitive sub-phrase patterns) than globally irregular, producing a characteristic compression profile distinct from human text.
4. **Co-reference is dense and regular**: LLMs re-reference entities more systematically and uniformly than humans.
5. **Topic transitions are smooth**: LLMs blend topics gradually where humans tend to jump.

### 3.3 Testable Separation

The key theoretical claim is that these two signature families are **linearly separable in the right feature space** and remain separable across paraphrase attacks because the attack does not change the underlying *process*. A paraphrase of AI text — even by another AI — preserves compression regularity, co-reference density, and coherence smoothness.

---

## 4. System Architecture

### 4.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: text document (min 150 words, max 100,000 words)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼──────────────────┐
         │               │                  │
         ▼               ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  LAYER 1     │  │  LAYER 2     │  │  LAYER 3     │
│  Compression │  │  Cognitive   │  │  Discourse   │
│  Entropy     │  │  Load        │  │  Topology    │
│  Analysis    │  │  Fingerprint │  │  Analysis    │
│  [LLM-free]  │  │  [LLM-free]  │  │  [NLP-only]  │
│  7 features  │  │  8 features  │  │  7 features  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┴──────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     LAYER 4      │
                │   Contrastive    │
                │  Manifold Embed  │
                │  DeBERTa-v3 +    │
                │  projection head │
                │  scored by GMM   │
                │   density ratio  │
                │   1 feature      │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     LAYER 5      │
                │  Bayesian Meta-  │
                │  Ensemble        │
                │  + Isotonic Cal  │
                │  + Conformal     │
                │  Prediction Sets │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     LAYER 6      │
                │  SHAP Sentence   │
                │  Attribution     │
                └────────┬─────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  OUTPUT                         │
        │  document_score: float [0,1]    │
        │  prediction_set: {AI,Human,Unk} │
        │  confidence_interval: [lo, hi]  │
        │  sentence_scores: list[float]   │
        │  signal_breakdown: dict         │
        │  highlighted_spans: list[Span]  │
        └─────────────────────────────────┘
```

### 4.2 Key Design Decisions

**Decision 1: No LLM at inference time for Layers 1–3.**  
Layers 1, 2, and 3 are entirely model-free. They use classical information theory, statistics, and lightweight NLP. This makes the system fast, cheap, and privacy-preserving — no text needs to leave a secure boundary.

**Decision 2: Contrastive metric learning, not binary classification.**  
Layer 4 learns a *geometry* — a manifold where AI and human texts are spatially separated — rather than a discriminative boundary. This generalizes to new AI models: new AI texts fall near the AI manifold even without retraining, as long as they share the core generative process.

**Decision 3: Conformal prediction, not hard thresholds.**  
The output is a calibrated probability interval. This directly addresses the false-accusation risk that governs Turnitin's entire architecture, but from a statistically principled framework rather than empirical threshold-setting.

---

## 5. Mathematical Foundations

### 5.1 Layer 1: Compression-Theoretic Analysis

#### 5.1.1 Lempel-Ziv Complexity

Given text $T$ as a token sequence, LZ76 complexity $\text{LZC}(T)$ counts the number of distinct phrases in the Lempel-Ziv parsing. Asymptotically:

$$\text{LZC}(T) \approx \frac{H(T) \cdot n}{\log_2 n}$$

where $H(T)$ is the source entropy rate. AI text produced by high-probability sampling has lower $H(T)$ and therefore lower $\text{LZC}(T)$ at equivalent length.

**Feature $\ell_1$** (normalized LZ complexity):

$$\ell_1 = \frac{\text{LZC}(T) \cdot \log_2 |T|}{|T|}$$

#### 5.1.2 Normalized Compression Distance with Reference Corpora

Using a 10 MB reference corpus $\mathcal{C}_{AI}$ of known AI texts and $\mathcal{C}_{H}$ of known human texts:

$$\text{NCD}(T, \mathcal{C}) = \frac{C(T \cdot \mathcal{C}) - \min(C(T), C(\mathcal{C}))}{\max(C(T), C(\mathcal{C}))}$$

where $C(\cdot)$ is zstd level-22 compression and $\cdot$ denotes concatenation.

**Features $\ell_2, \ell_3$**: $\ell_2 = \text{NCD}(T, \mathcal{C}_{AI})$, $\ell_3 = \text{NCD}(T, \mathcal{C}_{H})$

AI text compresses more efficiently against other AI text (lower $\ell_2$); human text compresses better against human text (lower $\ell_3$).

#### 5.1.3 Within-Document Compression Asymmetry

Divide into halves $T_A$ (first half) and $T_B$ (second half):

$$\ell_4 = \frac{C(T_A T_B)}{C(T_A) + C(T_B)} \qquad (\text{redundancy ratio})$$

$$\ell_5 = \left|\frac{C(T_A \mid T_B)}{|T_A|} - \frac{C(T_B \mid T_A)}{|T_B|}\right| \qquad (\text{directional asymmetry})$$

#### 5.1.4 Window Entropy Profile

Compute Shannon byte-entropy in sliding windows of 500 tokens with 250-token stride:

$$H_w = -\sum_{b=0}^{255} p_b \log_2 p_b$$

**Features $\ell_6, \ell_7$**: $\ell_6 = \text{Var}(\{H_w\})$, $\ell_7 = \text{AutoCorr}(\{H_w\}, \text{lag}=1)$

Human texts have higher entropy variance; AI entropy profiles are smooth.

---

### 5.2 Layer 2: Cognitive Load Fingerprint

#### 5.2.1 Sentence Length Distribution Shape

Fit a Gaussian $\hat{\mathcal{N}}(\hat\mu, \hat\sigma^2)$ and a Weibull $\hat{W}(\hat{k}, \hat\lambda)$ via MLE to observed sentence lengths $\{l_k\}$:

$$c_1 = D_{KL}(P_{SL} \| \hat{\mathcal{N}}) - D_{KL}(P_{SL} \| \hat{W})$$

AI text sentence lengths fit a Gaussian better ($c_1 > 0$); human text fits a Weibull better ($c_1 < 0$, reflecting right-skewed, heavy-tailed prose).

#### 5.2.2 Vocabulary Novelty Curve — Hurst Exponent

Define $V(i)$ = count of distinct words in tokens $1{\ldots}i$. Estimate the Hurst exponent $H$ via rescaled range analysis:

$$\frac{R(n)}{S(n)} \propto n^{H} \implies c_2 = \hat{H}$$

- $H > 0.65$: persistent (vocabulary bursts cluster — human behavior)
- $H \approx 0.5$: random Poisson arrival (typical AI default temperature)
- $H < 0.4$: anti-persistent (some low-temperature AI text)

#### 5.2.3 Semantic Coherence Trajectory

Using MiniLM-L6-v2 (22 M parameters), embed all sentences $\{e_k\}$. Compute consecutive cosine similarities:

$$r_k = \frac{e_k \cdot e_{k+1}}{\|e_k\| \|e_{k+1}\|}$$

$$c_3 = \text{Var}(\{r_k\}), \quad c_4 = \text{AutoCorr}(\{r_k\}, \text{lag}=2), \quad c_5 = \overline{r_k}$$

Human writing: low mean $c_5$, high variance $c_3$. AI writing: high mean, low variance ("too coherent").

#### 5.2.4 Epistemic Marker Density

Using a curated 240-entry lexicon of English epistemic expressions (*perhaps*, *I think*, *it seems*, *arguably*, *one might argue*, etc.):

$$c_6 = \frac{\text{count(epistemic markers)}}{|T|} \times 1000 \qquad (\text{per 1,000 words})$$

Empirically: AI at default sampling $\approx 0.3$–$0.8$ per 1k words; human academic prose $\approx 1.5$–$3.2$ per 1k words.

#### 5.2.5 Self-Reference and Correction Markers

$$c_7 = \frac{\text{count(first-person opinion markers)}}{|T|} \times 1000$$

$$c_8 = \frac{\text{count(self-correction phrases)}}{|T|} \times 1000$$

---

### 5.3 Layer 3: Discourse Topology Analysis

#### 5.3.1 Co-Reference Graph

Build undirected graph $G = (V, E)$ where $V$ = entity clusters and $E$ = co-occurrence within sentences:

$$d_1 = \frac{2|E|}{|V|(|V|-1)} \quad (\text{density}), \quad d_2 = \frac{1}{|V|}\sum_v C(v) \quad (\text{mean clustering coeff.}), \quad d_3 = \text{std(degree)}$$

AI text → higher $d_1$ and $d_2$ (denser, more clustered); human text → sparser, hub-and-spoke pattern.

#### 5.3.2 Topic Boundary Sharpness

Embed each paragraph. Compute Jensen-Shannon divergence between LDA topic distributions $\phi_i, \phi_{i+1}$ of consecutive paragraphs:

$$\text{JS}(i) = \frac{1}{2}D_{KL}(\phi_i \| m) + \frac{1}{2}D_{KL}(\phi_{i+1} \| m), \quad m = \frac{\phi_i + \phi_{i+1}}{2}$$

$$d_4 = \text{Var}(\{\text{JS}(i)\}), \quad d_5 = \max_i \text{JS}(i)$$

Human text: high $d_4$ (abrupt and gradual transitions mixed). AI text: low $d_4$, gradual throughout.

#### 5.3.3 Syntactic Depth Distribution

Using spaCy dependency parser, compute maximum dependency depth $D_k$ per sentence:

$$d_6 = \text{Var}(\{D_k\}), \quad d_7 = \text{Skewness}(\{D_k\})$$

AI text has more uniform syntactic complexity (low $d_6$); human text has higher variance.

---

### 5.4 Layer 4: Contrastive Manifold Embedding

#### 5.4.1 Architecture

```
Input text
  → DeBERTa-v3-base (86 M params) → [CLS] vector (768-dim)
  → Projection MLP (768 → 512 GELU → 256) → L2-normalize → z ∈ S^255
```

#### 5.4.2 Training Objective

Train on triplets with hard negative mining within each batch:

$$\mathcal{L} = \max\!\left(0,\; \|z_a - z_p\|^2 - \|z_a - z_n\|^2 + \alpha\right) + \lambda\,\|z_a - z_p\|^2$$

Where:
- $z_a$: anchor (human text)
- $z_p$: positive (different human text, same domain)
- $z_n$: hardest AI negative within the batch
- $\alpha = 0.3$ (margin), $\lambda = 0.05$ (compactness regularizer)

This is **not** a binary cross-entropy classifier. The model learns a geometry, not a decision boundary.

#### 5.4.3 Inference: GMM Manifold Distance Score

Fit Gaussian Mixture Models on training embeddings:

$$p_{AI}(z) = \sum_{j=1}^{M} \pi_j \,\mathcal{N}(z \mid \mu_j, \Sigma_j), \qquad p_H(z) \text{ analogously}$$

**Layer 4 score** (density ratio):

$$m_1 = \frac{p_{AI}(z_q)}{p_{AI}(z_q) + p_H(z_q)}$$

At inference, no training samples are needed in memory — only the fitted GMM parameters. New AI models generalize by geometric proximity in the contrastive space, not by explicit training on their outputs.

---

### 5.5 Layer 5: Ensemble and Calibration

#### 5.5.1 Feature Vector

$$\mathbf{x} = [\underbrace{\ell_1, \ldots, \ell_7}_{\text{Layer 1}},\; \underbrace{c_1, \ldots, c_8}_{\text{Layer 2}},\; \underbrace{d_1, \ldots, d_7}_{\text{Layer 3}},\; \underbrace{m_1}_{\text{Layer 4}}] \in \mathbb{R}^{23}$$

#### 5.5.2 Meta-Learner

Bayesian logistic regression with Laplace (L1) prior, trained on a held-out calibration set (never seen during Layers 1–4 training):

$$P(AI \mid \mathbf{x}) = \sigma\!\left(\beta_0 + \boldsymbol{\beta}^\top \mathbf{x}\right), \qquad \boldsymbol{\beta} \sim \text{Laplace}(0, b)$$

#### 5.5.3 Isotonic Calibration

Post-hoc isotonic regression on a separate isotonic calibration fold:

$$\hat{p}_{\text{cal}} = \text{IsotonicRegression}\!\left(P(AI \mid \mathbf{x})\right)$$

This is distinct from GPTZero's temperature scaling and Turnitin's Neyman-Pearson threshold.

#### 5.5.4 Conformal Prediction

Split conformal prediction with nonconformity score $s(x, y)$. Given target miscoverage $\alpha$:

$$\hat{q} = \text{Quantile}\!\left(\frac{\lceil(n+1)(1-\alpha)\rceil}{n},\; \{s_i\}_{i=1}^n\right)$$

$$\mathcal{C}(x) = \{y : s(x,y) \leq \hat{q}\}$$

Guarantees marginal coverage: $P(\hat{y} \in \mathcal{C}(x)) \geq 1 - \alpha$.

---

### 5.6 Layer 6: Sentence-Level Attribution

Per-sentence score as weighted combination of:

1. Per-sentence LZC (compression signal)
2. Inverse coherence to adjacent sentence $(1 - r_k)^{-1}$
3. SHAP contribution from Layer 4 embedding

$$\hat{p}_k = w_1 \cdot \text{CompScore}(s_k) + w_2 \cdot \frac{1}{1 - r_k} + w_3 \cdot \text{SHAP}(s_k)$$

Weights $\{w_i\}$ learned jointly with the meta-learner.

---

## 6. Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Language | Python 3.12 | Ecosystem breadth, strict typing |
| Compression | `python-zstandard`, `zlib` | Fast, high-ratio compression for NCD |
| LZ complexity | `lempel-ziv-complexity` | Direct LZ76 implementation |
| NLP pipeline | `spaCy` v3.8 (`en_core_web_trf`) | Dep-parse, sentence segmentation |
| Co-reference | `coreferee` (spaCy extension) | Lightweight neural co-reference |
| Fast encoder | `sentence-transformers` MiniLM-L6-v2 | 22 M params, ~5 ms/sentence |
| Contrastive encoder | DeBERTa-v3-base (HuggingFace) | Best embedding quality at <100 M params |
| Topic modeling | `BERTopic` | Paragraph-level topic distribution |
| Graph analysis | `networkx` | Co-reference graph features |
| Training | PyTorch 2.3 + HuggingFace `Trainer` | Contrastive loss custom module |
| Meta-learner | `scikit-learn` | Bayesian logistic regression, isotonic cal. |
| Conformal prediction | `crepes` | Split conformal, RAPS |
| Explainability | `shap` (KernelSHAP) | Sentence-level attribution |
| API | `FastAPI` + `uvicorn` | Async, Pydantic v2 validation |
| Caching | `Redis` | Embedding cache, deduplication |
| Experiment tracking | `MLflow` | Model versioning, metrics |
| Database | `PostgreSQL` | Audit log, document history |
| Testing | `pytest`, `hypothesis` | Unit + property-based |
| CI/CD | GitHub Actions | Lint + test on every PR |
| Containerization | Docker + docker-compose | Dev/prod parity |

---

## 7. Implementation Phases

### Phase 1 — Scaffold & Data Pipeline
**Duration**: Weeks 1–2

1. **Project scaffold** — Initialize package with `uv`, configure ruff/mypy/pytest, GitHub Actions CI.

2. **Data pipeline** (`cipher/data/pipeline.py`) — Ingest RAID-2 (5 M documents), HC3, M4, MAGE, and BLUFF-EN. Stratify by domain (news, academic, creative) and by generator (GPT-4o, Claude 3.5+, Gemini 2.0, Llama 3.3, Mistral-Large, human). Shard into train/cal/test (70/15/15). Keep calibration set isolated from all training.

3. **Evaluation harness** (`cipher/eval/harness.py`) — AUROC, TPR@FPR=1%, balanced accuracy, ECE, prediction-set coverage. Connect to RAID and M4 leaderboard formats.

---

### Phase 2 — Layer 1: Compression-Theoretic Analysis
**Duration**: Weeks 3–5

4. **LZ complexity feature** (`cipher/layers/compression.py`) — Normalized LZ76 on BPE token sequences.

5. **NCD with reference corpora** — Build 10 MB reference corpus (balanced domain). Compute NCD via zstd concatenation; cache per document by SHA-256.

6. **Within-document asymmetry features** — Redundancy ratio ($\ell_4$) and directional asymmetry ($\ell_5$).

7. **Window entropy profile** — Sliding 500-token windows; compute variance and lag-1 autocorrelation.

8. **Phase 2 evaluation** — Verify 3+ features individually reach AUROC > 0.65 on RAID test split.

---

### Phase 3 — Layer 2: Cognitive Load Fingerprint
**Duration**: Weeks 5–8

9. **Sentence length distribution** — MLE Gaussian and Weibull fit; KL-divergence difference $c_1$.

10. **Hurst exponent** — Rescaled range analysis on vocabulary novelty curve. Implement DFA fallback for documents < 500 tokens.

11. **Semantic coherence trajectory** — MiniLM-L6-v2 embeddings; consecutive cosine similarity variance, autocorrelation, mean.

12. **Epistemic marker lexicon** (`cipher/data/epistemic_lexicon.json`) — Curate 240-entry lexicon across 8 categories; exact-match + morphological normalization.

13. **Phase 3 evaluation** — $c_3$ (coherence variance) expected as strongest single cognitive feature.

---

### Phase 4 — Layer 3: Discourse Topology
**Duration**: Weeks 8–11

14. **Co-reference graph** — coreferee extraction → networkx graph → density, clustering coefficient, degree std. Chunked processing for long documents.

15. **Topic boundary analysis** — BERTopic (20 topics) on paragraph embeddings; JS divergence between consecutive paragraphs; variance and max jump.

16. **Syntactic depth distribution** — spaCy dep-parse; max depth per sentence; variance and skewness.

17. **Phase 4 evaluation** — $d_1$ (co-reference density) and $d_4$ (topic transition variance) expected as strongest features.

---

### Phase 5 — Layer 4: Contrastive Manifold Embedding
**Duration**: Weeks 11–17

18. **Triplet dataset** (`cipher/training/contrastive_dataset.py`) — 500k triplets from RAID-2; stratified domain sampling; anchor = human, positive = same-domain human, negative = AI from any model.

19. **DeBERTa projection head** (`cipher/models/contrastive_encoder.py`) — DeBERTa-v3-base + 2-layer MLP (768→512 GELU→256 L2-norm); custom `ContrastiveLoss` with hard negative mining.

20. **Training loop** (`cipher/training/train_contrastive.py`) — AdamW lr=2e-5, warmup 10%, batch=64, max_seq=512, 10 epochs. Hard negative mining after epoch 3. Save best checkpoint by validation AUROC (not contrastive loss). Gradient clipping; fallback to cosine-only loss if triplet loss diverges.

21. **GMM manifold fitting** (`cipher/models/manifold.py`) — Fit `GaussianMixture(n_components=16)` separately to AI and human training embeddings. Serialize parameters.

22. **Phase 5 evaluation** — Layer 4 alone target: AUROC > 0.88 on GPT-4o, AUROC > 0.82 on Llama 3.3 (unseen at training). UMAP visualization of manifold.

---

### Phase 6 — Layer 5: Ensemble and Calibration
**Duration**: Weeks 17–19

23. **Feature assembler** (`cipher/layers/assembler.py`) — Run all four layers; assemble 23-feature vector; cache by content SHA-256.

24. **Meta-learner** (`cipher/training/train_meta.py`) — Bayesian logistic regression with L1 regularization on calibration set only. Stratified k-fold cross-validation. Log feature importances to MLflow.

25. **Isotonic calibration** (`cipher/calibration/isotonic.py`) — Separate isotonic fold; validate ECE < 0.05.

26. **Conformal prediction** (`cipher/calibration/conformal.py`) — Split conformal via `crepes`; RAPS; validate empirical coverage at 90%, 95%, 99%.

27. **Bias audit** (`cipher/eval/bias_audit.py`) — Segment test set by: (a) non-native English (EFCAMDAT proxy), (b) short texts 150–300 words, (c) casual vs. academic. Target: max FPR ratio < 2× across groups.

---

### Phase 7 — Layer 6: Sentence Attribution
**Duration**: Weeks 19–21

28. **Per-sentence signals** — LZC and coherence score at sentence granularity; normalize by document baseline.

29. **SHAP attribution** — KernelSHAP on meta-learner with sentence segments as coalitions. Cache SHAP values; approximate SHAP for real-time API paths.

30. **Span highlighting** (`cipher/output/highlighter.py`) — Merge sentence scores into contiguous spans above threshold 0.65; output character offsets.

---

### Phase 8 — Adversarial Hardening
**Duration**: Weeks 21–24

31. **Adversarial test suite** — Apply three tiers to 2,000 AI documents: DIPPER paraphrase, Quillbot API, manual humanization prompts. Not included in training.

32. **Robustness evaluation** — Target: AUROC drop < 15% under DIPPER (vs. 40–60% degradation reported for perplexity-based systems).

33. **Adversarial augmentation** — If target missed, retrain Layer 4 with DIPPER-attacked examples as additional negatives. Monitor for attack-specific overfitting.

---

### Phase 9 — API and Deployment
**Duration**: Weeks 24–26

34. **REST endpoints** (`cipher/api/router.py`) — `POST /v1/detect`, `POST /v1/detect/batch`, `GET /v1/health`, `GET /v1/models`. Full Pydantic v2 validation.

35. **Async batch processing** — asyncio + ThreadPoolExecutor for CPU-bound Layers 1–3; process pool for GPU Layer 4. Redis request queuing.

36. **Caching** — SHA-256 content hash; 7-day TTL; skip recomputation for duplicates.

37. **Docker** — Two-stage build: CPU image (Layers 1–3 + meta-learner) for lightweight deployment; GPU image (adds DeBERTa) for full accuracy. Separate compose profiles.

---

### Phase 10 — Validation and Documentation
**Duration**: Weeks 26–28

38. **Full benchmarks** — RAID-2, M4, BLUFF-EN, HC3-English. Per-domain and per-model breakdown. Comparison against published GPTZero and Turnitin numbers.

39. **Ablation study** — Performance by text length bucket (150–300, 300–1k, 1k–5k, 5k+ words). Identify failure modes.

40. **API documentation** — OpenAPI spec from FastAPI; integration guide and quickstart.

---

## 8. Directory Structure

```
cipher/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── cipher/
│   ├── __init__.py
│   ├── api/
│   │   ├── router.py                # FastAPI routes
│   │   ├── batch.py                 # Async batch handler
│   │   ├── cache.py                 # Redis caching
│   │   └── schemas.py               # Pydantic v2 models
│   ├── layers/
│   │   ├── compression.py           # Layer 1: LZ, NCD, entropy
│   │   ├── cognitive.py             # Layer 2: SL dist, Hurst, coherence, epistemic
│   │   ├── discourse.py             # Layer 3: co-ref graph, topic, syntax
│   │   ├── assembler.py             # 23-feature vector assembly
│   │   └── sentence_attribution.py  # Layer 6: SHAP + compression per sentence
│   ├── models/
│   │   ├── contrastive_encoder.py   # DeBERTa + projection head
│   │   └── manifold.py              # GMM manifold models
│   ├── training/
│   │   ├── contrastive_dataset.py
│   │   ├── train_contrastive.py
│   │   ├── train_meta.py
│   │   └── adversarial_augment.py
│   ├── calibration/
│   │   ├── isotonic.py
│   │   └── conformal.py
│   ├── eval/
│   │   ├── harness.py
│   │   ├── adversarial_suite.py
│   │   ├── bias_audit.py
│   │   ├── full_benchmark.py
│   │   └── ablation.py
│   ├── data/
│   │   ├── pipeline.py
│   │   └── epistemic_lexicon.json
│   └── output/
│       └── highlighter.py
├── tests/
│   ├── layers/
│   │   ├── test_compression.py
│   │   ├── test_cognitive.py
│   │   └── test_discourse.py
│   ├── models/
│   │   └── test_manifold.py
│   └── eval/
│       └── test_adversarial.py
├── docs/
│   ├── api.md
│   └── openapi.yaml
└── scripts/
    ├── download_datasets.sh
    └── train_all.sh
```

---

## 9. API Contract

### Single Document Detection

```
POST /v1/detect
Content-Type: application/json

{
  "text": "string (min 150 words)",
  "options": {
    "include_sentence_scores": true,
    "include_signal_breakdown": true,
    "coverage_level": 0.95
  }
}
```

**Response**:

```json
{
  "document_score": 0.87,
  "prediction_set": ["AI"],
  "confidence_interval": [0.81, 0.93],
  "coverage": 0.95,
  "sentence_scores": [
    { "index": 0, "text": "...", "score": 0.91, "is_highlighted": true },
    { "index": 1, "text": "...", "score": 0.43, "is_highlighted": false }
  ],
  "signal_breakdown": {
    "layer1_compression": 0.72,
    "layer2_cognitive":   0.81,
    "layer3_discourse":   0.68,
    "layer4_manifold":    0.94
  },
  "highlighted_spans": [
    { "start": 0, "end": 412, "score": 0.91 }
  ],
  "metadata": {
    "word_count": 847,
    "processing_ms": 340,
    "model_version": "cipher-1.0"
  }
}
```

### Batch Detection

```
POST /v1/detect/batch
Content-Type: application/json

{
  "documents": [
    { "id": "doc_001", "text": "..." },
    { "id": "doc_002", "text": "..." }
  ],
  "options": { "coverage_level": 0.95 }
}
```

Response: array of single-document responses keyed by `id`.

---

## 10. Testing Strategy

| Test Type | Scope | Target |
|-----------|-------|--------|
| Unit (per layer) | Deterministic feature computations, hand-checked values | All functions |
| Property-based (`hypothesis`) | LZC monotone under repetition; NCD in [0,1]; conformal empirical coverage ≥ specified level | Critical invariants |
| Integration | Full pipeline on 100 labelled documents | AUROC > 0.85 |
| API | All schemas, error codes, timeout handling | All endpoints |
| Benchmark (gated) | RAID-2, M4, BLUFF-EN; run on every main-branch merge | See §12 |
| Adversarial (gated) | DIPPER, Quillbot, manual humanization | < 15% AUROC drop |
| Bias audit | Non-native English, short texts, casual vs. academic | FPR ratio < 2× |

**Coverage target**: ≥ 80% line coverage via `pytest-cov`.

---

## 11. Risk Register

| ID | Risk | Probability | Impact | Mitigation |
|----|------|-------------|--------|------------|
| R1 | Compression features unreliable for texts < 200 words | High | Medium | Require ≥ 150-word minimum; fall back to Layers 2+4 only for short texts; document in API |
| R2 | Hurst exponent noisy on documents < 500 tokens | Medium | Low | Implement DFA as alternative estimator; return `null` with warning if CI too wide |
| R3 | coreferee accuracy poor on informal text | Medium | Medium | Fallback to spaCy NER-based co-occurrence graph when coreferee confidence is low |
| R4 | Contrastive training loss divergence | Medium | High | Monitor every 100 steps; early stopping; fallback to cosine-only loss |
| R5 | Layer 4 fails to generalize to future LLMs | Medium | High | Quarterly manifold recalibration; automated pipeline to add new models within 7 days of public release |
| R6 | High FPR for non-native English writers | Medium | High | Bias audit mandatory (Step 27); per-population isotonic calibration if FPR ratio > 2× |
| R7 | GPU unavailable in deployment | Low | Medium | CPU-only mode skips Layer 4; documented to degrade to ~0.80 AUROC; separate Docker profile |
| R8 | NEC US12596869B2 scope interpreted broadly | Low | High | CIPHER does not truncate text, does not regenerate with an LLM, does not compute n-gram overlap between regenerated and original text — all three elements of Claims 1 and 5 are absent |

---

## 12. Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Overall AUROC | ≥ 0.90 on RAID-2 test set | eval harness |
| TPR @ FPR = 1% | ≥ 0.75 | eval harness |
| FPR for human text | < 1% at score ≥ 0.50 threshold | eval harness |
| Non-native English FPR ratio | < 2× vs. native speakers | bias audit |
| AUROC on Llama 3.3 (unseen at training) | ≥ 0.82 | generalization eval |
| AUROC under DIPPER attack | ≥ 0.75 (< 15% drop from clean) | adversarial eval |
| Conformal coverage at 95% target | 94%–96% empirical | conformal eval |
| CPU inference latency (1,000-word doc) | < 500 ms (Layers 1–3 + meta-learner) | load test |
| GPU inference latency (1,000-word doc) | < 800 ms (all layers) | load test |
| Test coverage | ≥ 80% | pytest-cov |

---

## Appendix A: Legal Distinctness Summary

**vs. NEC US12596869B2** (truncation + regeneration + n-gram overlap):  
CIPHER does not truncate the document, does not call an LLM to regenerate continuations, and does not compute n-gram overlap between regenerated and original text. All three elements required by Claims 1 and 5 are absent.

**vs. GPTZero**:  
CIPHER does not compute token-level or sentence-level perplexity (PPL), does not use log-probability curvature under any LLM, and does not use burstiness (variance of sentence PPL).

**vs. Turnitin**:  
CIPHER does not train a binary cross-entropy classifier on human/AI document pairs. It does not use word probability sequences as features. The three-model writing/paraphrase/bypasser architecture is not replicated.

**vs. ObjectSecurity US20240202405A1**:  
CIPHER analyzes output text using compression and cognitive signal features; it does not analyze AI components within computing systems.

**Primary novel contributions eligible for IP protection**:
1. Lempel-Ziv complexity + NCD with reference corpora as primary text-authorship signals
2. Hurst exponent of the vocabulary novelty curve as a detection feature
3. The combination of compression-theoretic + cognitive load + discourse topology signals in a contrastive manifold ensemble
4. Conformal prediction sets applied to AI text detection output (coverage-guaranteed prediction intervals)
