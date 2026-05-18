# Architecture

> Cross-references: [User Guide](user_guide.md) · [API Reference](api_reference.md) · [Dataset Documentation](datasets.md) · [Index](index.md)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Layer Descriptions](#layer-descriptions)
   - [Layer 1 — Compression-Theoretic Features](#layer-1--compression-theoretic-features)
   - [Layer 2 — Cognitive Load Fingerprint](#layer-2--cognitive-load-fingerprint)
   - [Layer 3 — Discourse Topology](#layer-3--discourse-topology)
   - [Layer 4 — Contrastive Manifold (Optional)](#layer-4--contrastive-manifold-optional)
   - [Layer 5 — Bayesian Meta-Ensemble](#layer-5--bayesian-meta-ensemble)
   - [Layer 6 — Sentence Attribution](#layer-6--sentence-attribution)
3. [Feature Vector](#feature-vector)
4. [Calibration](#calibration)
5. [Conformal Prediction](#conformal-prediction)
6. [Model Performance](#model-performance)
7. [Design Decisions](#design-decisions)

---

## System Overview

```
Input text
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│  Layer 1: Compression-theoretic (LZ76 complexity, NCD)         │
│  → 7 features: l1…l7                                           │
├────────────────────────────────────────────────────────────────┤
│  Layer 2: Cognitive load fingerprint (Hurst, epistemic)        │
│  → 8 features: c1…c8                                           │
├────────────────────────────────────────────────────────────────┤
│  Layer 3: Discourse topology (co-reference graph, topic, syntax)│
│  → 7 features: d1…d7                                           │
├────────────────────────────────────────────────────────────────┤
│  Layer 4: Contrastive manifold (DeBERTa-v3-base, optional)     │
│  → 1 feature: m1                                               │
├────────────────────────────────────────────────────────────────┤
│  Layer 5: Logistic regression meta-ensemble                     │
│  → raw score s ∈ [0, 1]                                        │
├────────────────────────────────────────────────────────────────┤
│  Calibration (isotonic regression + conformal prediction)      │
│  → calibrated score + prediction set + confidence interval     │
├────────────────────────────────────────────────────────────────┤
│  Layer 6: Sentence attribution                                  │
│  → per-sentence AI probability + highlighted spans             │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
DetectionResult
```

---

## Layer Descriptions

### Layer 1 — Compression-Theoretic Features

**Module:** `cipher.layers.compression`  
**Features:** `l1`–`l7`

| Name | Symbol | Description |
|---|---|---|
| `l1_lz76` | $C_{LZ}$ | Lempel–Ziv 76 normalised complexity: fraction of distinct patterns in the bit-string encoding of the text. Human text is more complex than LLM output. |
| `l2_zstd_ratio` | $\rho_z$ | zstd compression ratio. Lower ratio → more compressible → more AI-like. |
| `l3_vocab_richness` | $V/N$ | Type–token ratio (unique words / total words). AI text has unusually high but consistent vocabulary richness. |
| `l4_redundancy` | $R$ | $1 - H / \log_2 V$ where $H$ is Shannon entropy of the unigram distribution. Measures predictability. |
| `l5_asym` | $A$ | Asymmetry of bigram probabilities vs. reversed bigrams. AI text tends toward more symmetric bigram distributions. |
| `l6_ncd_ai` | $d_{AI}$ | Normalised Compression Distance to an AI reference corpus. Low NCD → text is similar to AI corpus. |
| `l7_ncd_human` | $d_H$ | Normalised Compression Distance to a human reference corpus. |

**Key Formula:**

Lempel–Ziv 76 complexity:

$$C_{LZ}(s) = \frac{c(s)}{n / \log_2 n}$$

where $c(s)$ is the number of distinct phrases in the Lempel–Ziv parsing of the binary string $s$, and $n = |s|$.

Normalised Compression Distance:

$$\text{NCD}(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$$

where $C(\cdot)$ denotes the compressed size.

**Fallback:** When reference corpora are unavailable, `l6_ncd_ai` and `l7_ncd_human` are set to 0.5 (neutral).

---

### Layer 2 — Cognitive Load Fingerprint

**Module:** `cipher.layers.cognitive`  
**Features:** `c1`–`c8`

| Name | Symbol | Description |
|---|---|---|
| `c1_sl_kl` | $D_{KL}(P_s \| P_r)$ | KL divergence of sentence-length distribution from a reference distribution. AI text has narrower, more uniform sentence lengths. |
| `c2_hurst` | $H$ | Hurst exponent of the sentence-length time series. $H > 0.5$ indicates long-range correlation (human); $H \approx 0.5$ is AI-like (Brownian motion). |
| `c3_hedge_density` | $\lambda_h$ | Density of hedging/epistemic markers (I think, perhaps, maybe, etc.) per 1000 tokens. Human text hedges more. |
| `c4_function_ratio` | $r_f$ | Ratio of function words to content words. AI text has a subtly lower function-word ratio. |
| `c5_discourse_conn` | $\rho_{dc}$ | Density of explicit discourse connectives (however, therefore, although). AI text over-uses certain connectives. |
| `c6_epistemic` | $\lambda_e$ | Count of epistemic modality markers per sentence. |
| `c7_passive_ratio` | $r_p$ | Fraction of sentences in passive voice. |
| `c8_avg_dep_dist` | $\bar{d}$ | Mean dependency arc length across all sentences. Measures syntactic complexity. |

**Hurst Exponent (R/S Analysis):**

$$H = \frac{\log(R/S)}{\log(n)}$$

where $R/S$ is the rescaled range statistic computed over the sentence-length time series.

---

### Layer 3 — Discourse Topology

**Module:** `cipher.layers.discourse`  
**Features:** `d1`–`d7`

| Name | Symbol | Description |
|---|---|---|
| `d1_coref_density` | $\rho_G$ | Edge density of the co-reference graph: how densely entities are co-mentioned. Human text has richer entity co-reference. |
| `d2_coref_clustering` | $\bar{c}$ | Mean clustering coefficient of the co-reference graph. |
| `d3_coref_deg_std` | $\sigma_k$ | Standard deviation of co-reference graph node degrees. AI text has more uniform entity distributions. |
| `d4_topic_var` | $\text{Var}(JS)$ | Variance of Jensen–Shannon divergences between adjacent paragraph topic distributions. Human text has higher topic boundary variance. |
| `d5_topic_max` | $\max(JS)$ | Maximum JS divergence between adjacent paragraphs. |
| `d6_depth_var` | $\text{Var}(D)$ | Variance of per-sentence dependency tree depths. Human writing has more varied syntactic depth. |
| `d7_depth_skew` | $\gamma_1(D)$ | Skewness of sentence dependency depths. |

**Co-reference Graph Construction:**

1. Try coreferee (if installed) for full coreference resolution.
2. Fallback: spaCy NER — build entity co-occurrence graph where two entities share an edge if they appear in the same sentence.
3. If no graph with ≥ 2 nodes can be built: use fallback values (0.3, 0.3, 1.0).

**Topic Boundary Features (d4, d5):**

Paragraphs are encoded as TF-IDF vectors, reduced to 10 topics via truncated SVD, then JS divergence is computed between adjacent paragraphs:

$$JS(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{P + Q}{2}$$

**Important:** Texts with fewer than 3 paragraphs return `(0.0, 0.0)` rather than the training distribution mean. This prevents z-score inflation for short texts.

---

### Layer 4 — Contrastive Manifold (Optional)

**Module:** `cipher.models.contrastive_encoder`  
**Feature:** `m1`  
**Requires:** `pip install cipher-detect[neural]`

A DeBERTa-v3-base encoder fine-tuned with triplet loss:

$$\mathcal{L} = \max(0, \|z_a - z_p\|^2 - \|z_a - z_n\|^2 + \alpha) + \lambda \|z_a - z_p\|^2$$

where anchor and positive are human texts, negative is AI text. The `λ` compactness term tightens the human-text cluster.

At inference, two Gaussian Mixture Models (GMMs) are pre-fitted on AI and human embeddings. The manifold score is:

$$m_1 = \frac{p_{AI}(z)}{p_{AI}(z) + p_{Human}(z)}$$

computed via numerically stable log-sum-exp. When this layer is unavailable (no torch), the feature is simply omitted from the feature vector.

---

### Layer 5 — Bayesian Meta-Ensemble

**Module:** `cipher.models.meta_learner`

A `LogisticRegression` meta-learner (scikit-learn) trained on the 22 (or 23) features from Layers 1–4:

```python
LogisticRegression(
    l1_ratio=1.0,
    solver="saga",
    C=0.1,
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)
```

- `l1_ratio=1.0` applies pure L1 regularisation (ElasticNet with full sparsity), which automatically selects the most informative features.
- `class_weight="balanced"` compensates for class imbalance in training data.
- `C=0.1` (inverse regularisation strength) was chosen by 5-fold cross-validation.

**Heuristic Mode:** When no trained model is available, a hand-tuned linear combination of the most reliable features (LZ76, Hurst, hedge density, KL divergence) produces a fast approximate score. AUROC ≈ 0.72 in heuristic mode.

**Feature Importances:** After training, the signed logistic regression coefficients are accessible via `MetaLearner.feature_importances()`.

---

### Layer 6 — Sentence Attribution

**Module:** `cipher.output.highlighter`

Per-sentence AI probability is estimated by combining three signals:

1. **LZ76 score** — low normalised complexity → high AI signal. Short sentences (< 5 words) receive the document median.
2. **Semantic coherence** — high cosine similarity between adjacent sentence embeddings → AI signal (AI text is unnaturally coherent). Falls back to position-weighted heuristic when no encoder is available.
3. **SHAP contribution** — when a trained meta-learner is available, the SHAP value of the sentence (computed by perturbing individual sentences) is included. *(Planned for v1.1.)*

The three signals are combined by equal-weight averaging. Sentences scoring above `highlight_threshold` (default 0.60) are marked `is_highlighted=True` and their character offsets are included in `highlighted_spans`.

---

## Feature Vector

The full feature vector has **23 dimensions**:

```
l1_lz76          l2_zstd_ratio    l3_vocab_richness  l4_redundancy
l5_asym          l6_ncd_ai        l7_ncd_human
c1_sl_kl         c2_hurst         c3_hedge_density    c4_function_ratio
c5_discourse_conn c6_epistemic    c7_passive_ratio    c8_avg_dep_dist
d1_coref_density  d2_coref_clustering d3_coref_deg_std
d4_topic_var     d5_topic_max
d6_depth_var     d7_depth_skew
m1_manifold      (optional — included only when Layer 4 is active)
```

Feature names are available as `FeatureVector.FEATURE_NAMES` (a 23-element tuple).

See [API Reference § FeatureVector](api_reference.md#featurevector).

---

## Calibration

### Problem

Logistic regression outputs raw scores that are not necessarily well-calibrated probabilities. Additionally, the heuristic meta-learner uses hard-coded weights rather than learned ones.

### Isotonic Regression

After training, an `IsotonicCalibrator` is fitted on 20% held-out calibration data:

```
raw_score ─────► IsotonicRegression ────► calibrated_score
```

Isotonic regression is a monotone, piecewise-constant mapping. It is non-parametric: it only assumes scores should be increasing in probability.

**Important:** When using a trained sklearn `LogisticRegression`, isotonic calibration is **bypassed** because sklearn's `predict_proba()` already applies Platt scaling internally. Applying isotonic on top would re-introduce calibration error.

```python
if self._meta.mode == "trained":
    calibrated_score = float(np.clip(raw_score, 0.0, 1.0))
else:
    calibrated_score = self._calibrator.transform(raw_score)
```

### Solved Calibration Collapse Bug

In an earlier version, the `d4_topic_var` fallback for texts with < 3 paragraphs was `0.1` instead of `0.0`. Since the training distribution had a mean near `0.004` and std `0.012`, this produced a z-score of `+8.27`. With a logistic regression weight of `+0.926` on `d4`, this single feature contributed `+7.66` to the logit, forcing `sigmoid(7.66) ≈ 1.0` for all short texts. The fix: return `(0.0, 0.0)` from `topic_boundary_features()` for short texts.

---

## Conformal Prediction

CIPHER wraps the calibrated score in a **split-conformal prediction set**, providing coverage-guaranteed uncertainty quantification.

**Setup (calibration phase):**

For each calibration document with score $s_i$ and true label $y_i \in \{0,1\}$:

$$\alpha_i = \begin{cases} 1 - s_i & \text{if } y_i = 1 \text{ (AI)} \\ s_i & \text{if } y_i = 0 \text{ (Human)} \end{cases}$$

The conformal quantile $\hat{q}$ is set at level $\lceil(n+1)(1-\alpha)\rceil/n$.

**Inference:**

The prediction set includes:

- `"AI"` if $1 - s \leq \hat{q}$
- `"Human"` if $s \leq \hat{q}$

An empty set falls back to `["AI", "Human"]`.

**Coverage guarantee:** For any new test document from the same distribution:

$$P(y^* \in \mathcal{C}(x^*)) \geq 1 - \alpha$$

At `coverage=0.95`, the prediction set contains the true label at least 95% of the time.

---

## Model Performance

Trained on 86 534 documents from 6 public datasets, evaluated on 15 270 held-out documents:

| Metric | Heuristic | Trained (no L4) |
|---|---|---|
| **AUROC** | 0.72 | **0.915** |
| **Accuracy @0.5** | 65% | **89%** |
| F1 (AI class) | 0.63 | 0.88 |
| F1 (Human class) | 0.67 | 0.90 |

**Score distribution (trained model):**

- AI texts: median 0.878 (IQR: 0.71–0.96)
- Human texts: median 0.165 (IQR: 0.06–0.30)

**Calibration curve:** The trained model is well-calibrated — the mean predicted probability tracks the empirical fraction of AI texts across all score deciles.

For dataset details, see [Dataset Documentation](datasets.md).

---

## Design Decisions

### Why Not Fine-Tune an LLM Detector?

Fine-tuning a classifier on top of a large language model (e.g., RoBERTa) produces excellent in-distribution accuracy but:
1. Requires GPU at inference time.
2. Is brittle against adversarial paraphrasing.
3. Provides no explainability — the score is a black box.
4. Cannot run on air-gapped systems.

CIPHER's feature-based approach trades some accuracy for interpretability and robustness.

### Why Conformal Prediction?

Standard confidence intervals from logistic regression are model-dependent and can be poorly calibrated out of distribution. Conformal prediction provides distribution-free, finite-sample coverage guarantees as long as the calibration and test sets are exchangeable.

### Why L1 Regularisation?

The 22-feature vector contains correlated features (e.g., `l2_zstd_ratio` and `l1_lz76` are correlated). L1 regularisation produces sparse solutions, automatically selecting the most discriminative features and ignoring redundant ones. The top features by absolute coefficient are: `d4_topic_var`, `c2_hurst`, `c1_sl_kl`, `l5_asym`, `c6_epistemic`.

### Why `class_weight="balanced"`?

Real-world AI detection datasets often have 60–70% AI content. Without balancing, the model learns to over-predict AI. Balanced weights rescale the loss to treat each class equally regardless of prevalence.
