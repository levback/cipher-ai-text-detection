# AI-Generated Text Detection: Comprehensive Research Report

**Version**: 2.0  
**Date**: May 2026  
**Scope**: Academic literature (2019–2026), vendor disclosures, open-source tools  
**Sources**: arXiv, ACL Anthology, Nature, NeurIPS, ICML, ICLR, vendor whitepapers, independent evaluations

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Method Taxonomy](#2-method-taxonomy)
3. [Zero-Shot Statistical Methods](#3-zero-shot-statistical-methods)
4. [Trained Neural Classifiers](#4-trained-neural-classifiers)
5. [Structural & Linguistic Feature Methods](#5-structural--linguistic-feature-methods)
6. [Watermarking-Based Detection](#6-watermarking-based-detection)
7. [Hybrid and Ensemble Approaches](#7-hybrid-and-ensemble-approaches)
8. [Vendor Analysis: Turnitin / iThenticate / GPTZero](#8-vendor-analysis-turnitin--ithenticate--gptzero)
9. [Benchmark Results and Comparative Performance](#9-benchmark-results-and-comparative-performance)
10. [Strengths of Current Methods](#10-strengths-of-current-methods)
11. [Weaknesses and Failure Modes](#11-weaknesses-and-failure-modes)
12. [Gaps in the Field (2026 Update)](#12-gaps-in-the-field-2026-update)
13. [Open-Source Implementation Pathways](#13-open-source-implementation-pathways)
14. [Recommendations for Building AEGIS](#14-recommendations-for-building-aegis)
15. [Full Bibliography](#15-full-bibliography)

---

## 1. Executive Summary

AI text detection in 2026 is a mature but deeply unsolved problem. Generating convincing machine text is computationally cheap; reliably detecting it post-hoc remains **provably hard** in the general case (Sadasivan et al., 2023). Yet practical tools — both academic and commercial — achieve operationally useful performance, provided their limits are understood.

**The field in 2026** has evolved across four dimensions relative to 2023:

| Dimension | 2023 State | 2026 State |
|-----------|-----------|-----------|
| Zero-shot methods | DetectGPT, Binoculars | LAPD / Alignment Imprint, improved Binoculars variants |
| Trained classifiers | RoBERTa binary heads | Multi-task detectors (MELD, DRGD) with generalization focus |
| Watermarking | KGW, SynthID-Text | ECC watermarks, GaussMark, statistical framework (Ann. Stat.) |
| Benchmarks | RAID, M4 | BLUFF (58 languages), C-ReD (Chinese), MELD-eval |

**Key finding for implementors**: No single method achieves simultaneously high accuracy, low false positives, robustness to paraphrasing, and cross-domain generalization. The best open-source path is a **calibrated multi-signal ensemble** with: (1) a multi-task neural detector trained with adversarial augmentation, (2) zero-shot statistical signals as cross-check, (3) structural/syntactic features for robustness, and (4) optional watermark detection as a high-confidence override.

---

## 2. Method Taxonomy

### 2.1 Full Classification Tree

```
AI-Generated Text Detection
├── Post-hoc Detection (applied after generation)
│   ├── Zero-Shot / Statistical
│   │   ├── Perplexity-based (GLTR, GPTZero v1)
│   │   ├── Perturbation-based curvature (DetectGPT, Fast-DetectGPT)
│   │   ├── Log-rank based (DetectLLM)
│   │   ├── Cross-model ratio (Binoculars)
│   │   ├── Alignment-based (LAPD / Alignment Imprint, 2026)
│   │   └── Topological (PHD / persistent homology dimension)
│   ├── Structural / Linguistic
│   │   ├── Syntactic dependency patterns (DependencyAI, 2026)
│   │   ├── Stylometric features (TTR, burstiness, POS distributions)
│   │   └── Semantic consistency profiling (Lightweight Stylistic, 2026)
│   ├── Trained Classifiers
│   │   ├── Fine-tuned encoders (RoBERTa, XLM-R)
│   │   ├── Adversarially trained (RADAR, MELD, 2026)
│   │   ├── Disentangled representation (DRGD, 2026)
│   │   ├── Feature-engineered (Ghostbuster)
│   │   └── Commercial (Turnitin, GPTZero, Originality.ai)
│   └── Hybrid / Ensemble
│       ├── Multi-signal fusion (GPTZero production, MELD)
│       └── Retrieval-augmented (DIPPER + retrieval, DNA-GPT)
└── Proactive Detection (applied during generation)
    └── Watermarking
        ├── Token-level sampling (KGW — red/green lists)
        ├── Tournament sampling (SynthID-Text)
        ├── Structural / model-weight (GaussMark, 2025)
        ├── Error-correcting codes (ECC Watermark, 2024)
        ├── Semantic-level (PMark, k-SemStamp)
        ├── Statistical framework (Li et al. Annals of Statistics, 2025)
        └── Undetectable / cryptographic (Christ et al., 2024)
```

### 2.2 Key Dimensions for Comparison

| Dimension | Description |
|-----------|-------------|
| **Access requirements** | White-box (weights), grey-box (logits), black-box (text only) |
| **Training required** | Zero-shot vs. supervised vs. semi-supervised |
| **Granularity** | Document → paragraph → sentence → token |
| **Source model specificity** | Model-specific vs. model-agnostic |
| **Robustness** | Clean text vs. paraphrased vs. adversarially rewritten |
| **Computational cost** | Inference time, GPU requirements |
| **Explainability** | Black-box score vs. feature attribution |

---

## 3. Zero-Shot Statistical Methods

### 3.1 GLTR — Giant Language model Test Room (Gehrmann et al., 2019)

**Method**: Visualizes token-level likelihood ranks under a reference LM. Highlights tokens by color band based on rank in the predicted distribution.

**Mathematical basis**: For token $x_t$, rank $r(x_t) = |\{v \in V : P(v \mid x_{<t}) > P(x_t \mid x_{<t})\}|$. AI text concentrates tokens in low-rank (high-probability) positions.

**Strengths**: Interpretable visual output; no training; works with any reference model.  
**Weaknesses**: Requires human judgment; defeated by high-temperature sampling; reference model mismatch.

---

### 3.2 DetectGPT (Mitchell et al., 2023)

**Method**: Perturbs input with a mask-filling model (T5), then checks whether the original sits at a local log-probability maximum. AI text tends to occupy curvature maxima.

$$d(x, p_\theta, q) = \log p_\theta(x) - \mathbb{E}_{\tilde{x} \sim q(\cdot|x)}\bigl[\log p_\theta(\tilde{x})\bigr]$$

**Strengths**: Zero-shot; theoretically motivated; cross-domain.  
**Weaknesses**: Requires 50–100 perturbations (slow); needs source model logits; paraphrasing erases curvature signal.

---

### 3.3 Fast-DetectGPT (Bao et al., 2023)

**Method**: Replaces expensive perturbation sampling with a direct entropy estimate at each position — ~340× faster than DetectGPT with comparable accuracy.

$$\hat{d}(x) \approx \frac{1}{N}\sum_{t=1}^{N}\!\left(\log p_\theta(x_t \mid x_{<t}) - \sum_{v \in V} p_\theta(v \mid x_{<t})\log p_\theta(v \mid x_{<t})\right)$$

**Strengths**: Fast; zero-shot; SOTA among perturbation-based methods.  
**Weaknesses**: Still needs logits; curvature signal weakens with advanced generation strategies.

---

### 3.4 Binoculars (Hans et al., 2024)

**Method**: Uses two LLMs to compute a cross-perplexity / perplexity ratio. Human text is "surprising" to both models similarly; AI text from one model has suspiciously low self-perplexity.

$$\text{score}(x) = \frac{\text{CE}(x;\, M_1)}{\text{PPL}(x;\, M_2)}$$

**Strengths**: SOTA zero-shot AUROC at publication; model-agnostic; no training data.  
**Weaknesses**: Requires two capable LLMs (cost); degrades on very short texts.

---

### 3.5 Alignment Imprint / LAPD (Wu et al., 2026)

**Method** *(new in 2026)*: Exploits the measurable distributional imprint left by RLHF / preference tuning. The alignment process leaves a log-likelihood ratio that decomposes into implicit instructional biases and preference rewards — the "Alignment Imprint." LAPD (Log-likelihood Alignment Preference Discrepancy) standardizes this as an information-weighted statistic.

**Key result**: 45.82% relative improvement over the strongest existing baselines; provably dominates Fast-DetectGPT.

**Why it matters**: As model outputs become statistically indistinguishable from human text, alignment-based signals remain detectable because RLHF imposes a systematic distributional shift that is independent of content quality.

**Strengths**: Provable improvement over Fast-DetectGPT; stable across high-entropy regions; zero-shot.  
**Weaknesses**: Requires access to both base model and aligned model logits (grey-box).

---

### 3.6 PHD — Persistent Homology Dimension (Tulchinskii et al., 2024)

**Method**: Computes intrinsic dimensionality of sentence-embedding clouds via persistent homology. AI text occupies a lower-dimensional manifold than human text.

**Strengths**: Paraphrase-robust (topology is invariant to surface rewording); language-agnostic in principle.  
**Weaknesses**: Computationally expensive; requires embedding model; less mature tooling.

---

### 3.7 DetectLLM (Su et al., 2023)

**Method**: Uses log-rank statistics from a reference LM rather than log-probability. Log-rank is less sensitive to calibration mismatches between the reference model and the generating model.

**Strengths**: More robust to model mismatch than perplexity; fast inference.  
**Weaknesses**: Still zero-shot statistical — limited robustness to adversarial attacks.

---

### 3.8 DNA-GPT (Yang et al., 2023)

**Method**: Re-generates text from the suspected source model using truncated prompts; measures n-gram overlap. AI text reproduces with higher overlap.

**Strengths**: Black-box (needs only API, not logits).  
**Weaknesses**: Multiple API calls; fails when model is updated; temperature changes reduce overlap.

---

## 4. Trained Neural Classifiers

### 4.1 RoBERTa-based Classifiers

**Method**: Fine-tune an encoder (RoBERTa, DeBERTa, XLM-R) on (human, AI) pairs as a binary classifier.

**Strengths**: High in-distribution accuracy; fast inference; can capture subtle patterns.  
**Weaknesses**: Distribution shift degrades performance sharply; encodes dataset biases; labeled data becomes stale.

---

### 4.2 RADAR (Hu et al., 2023)

**Method**: Adversarial training loop: a paraphraser tries to evade the detector; the detector retrains on the paraphrased outputs. Iterates over multiple rounds.

**Strengths**: Explicitly addresses paraphrasing robustness; better generalization to unseen models than standard classifiers.  
**Weaknesses**: Expensive to train; adversarial ceiling limited by paraphraser quality.

---

### 4.3 Ghostbuster (Verma et al., 2024)

**Method**: Structured search over probability features from multiple reference models, then a linear classifier. Features include per-model probability, ratios, and deltas.

**Strengths**: Interpretable; good cross-model generalization; efficient inference.  
**Weaknesses**: Requires multiple reference models; linear capacity limits complex patterns.

---

### 4.4 MELD — Multi-Task Equilibrated Learning Detector (Li et al., 2026)

**Method** *(new, open-source)*: Attaches four heads to a shared encoder: (1) binary AI/human, (2) generator family, (3) attack type, (4) source domain. Losses are balanced with **homoscedastic uncertainty weighting**. Uses an EMA teacher/student distillation setup where the student sees attack-augmented inputs. A **hard-negative pairwise ranking loss** enlarges the margin between AI text and the most confusable human text.

**Results**: Strongest open-source detector on the RAID leaderboard; 99.9% TPR at 1% FPR on MELD-eval (recent chat models from four major providers) without fine-tuning.

**Why implementable**: Model weights and code publicly released. Architecture is a standard transformer encoder with auxiliary heads — no proprietary components.

**Strengths**: SOTA open-source performance; multi-task generalization; explicit attack robustness; low FPR focus.  
**Weaknesses**: Requires training on diverse labeled data; GPU-intensive; auxiliary heads discarded at inference but increase training complexity.

---

### 4.5 DRGD — Disentangled Representation for Generalizable AI-Text Detection (Pu et al., 2026)

**Method** *(new, open-source)*: Progressively disentangles AI-detection semantics from generator-specific artifacts through: (1) compact latent encoding (semantic minimality), (2) perturbation-based regularization to remove residual entanglement, (3) discriminative adaptation to task objectives. Evaluated on MAGE benchmark covering 20 LLMs across 7 categories.

**Results**: Up to 24.2% accuracy gain and 26.2% F1 improvement over prior SOTA; performance improves as generator diversity in training increases.

**Code**: `https://github.com/PuXiao06/DRGD`

**Strengths**: Explicitly designed for unseen-generator generalization; scalable with generator diversity; open-source.  
**Weaknesses**: Disentanglement training is complex; requires diverse multi-generator training data.

---

### 4.6 NOTAI.AI — Explainable Detection via Curvature and Feature Attribution (Marchenko et al., 2026)

**Method** *(new)*: Combines Fast-DetectGPT-style curvature scoring with SHAP-based feature attribution to produce both a detection score and a per-token explanation highlighting which tokens most contributed to the AI classification.

**Strengths**: Explainability-first design; competitive detection performance; open Python package.  
**Weaknesses**: SHAP computation overhead; curvature still requires logit access.

---

### 4.7 DependencyAI (Ahmed & Hammond, 2026)

**Method** *(new, lightweight)*: Uses only syntactic dependency relation labels (no embeddings, no neural networks) as features for a traditional ML classifier. Achieves competitive performance across monolingual, multi-generator, and multilingual settings.

**Why it matters**: Demonstrates that structural syntactic patterns alone reliably distinguish AI from human text — a computationally minimal, interpretable, and language-portable signal.

**Strengths**: No GPU required; interpretable; multilingual; fast.  
**Weaknesses**: Lower ceiling than neural approaches; generator-specific writing styles cause cross-domain overprediction.

---

### 4.8 Why Detection Fails: Dataset Artifact Analysis (Pudasaini et al., 2026)

This study trained a 30-feature linguistic classifier achieving F1 = 0.9734 on PAN CLEF 2025 and COLING 2025 benchmarks — then used SHAP to show it was **exploiting dataset-specific artifacts rather than stable machine-authorship signals**. Key findings:

- The most discriminative features on in-domain data are the most susceptible to domain shift, formatting variation, and text-length effects.
- Classifiers that excel in-domain degrade significantly under distribution shift.
- The same features that separate human from AI text within a corpus may not generalize across corpora.

**Implication for AEGIS**: Evaluation must include cross-domain and cross-generator splits, not just held-out in-domain samples.

---

## 5. Structural & Linguistic Feature Methods

### 5.1 Stylometric Features

Human and AI writing differ systematically in measurable surface features:

| Feature | AI Tendency | Human Tendency |
|---------|-------------|----------------|
| Type-Token Ratio | Lower (less lexical diversity) | Higher |
| Burstiness (surprisal variance) | Low (uniform surprise) | High (varied) |
| Mean sentence length | Longer, more uniform | More variable |
| Hapax legomena ratio | Lower | Higher |
| Punctuation patterns | Formulaic, less varied | More idiosyncratic |
| Named entity density | Moderate, context-dependent | Domain-specific variation |
| Flesch-Kincaid readability | Higher (cleaner prose) | More variable |

These features are **paraphrase-resistant** (a paraphrase preserves most structural patterns) and **language-portable** (can be adapted to non-English with appropriate NLP tools).

### 5.2 Syntactic Dependency Features (DependencyAI)

AI text produces systematically different distributions of dependency arc types (e.g., `nsubj`, `dobj`, `amod`, `advmod`). These patterns are detectable with no neural components and transfer across languages that have dependency parsers.

### 5.3 Lightweight Stylistic Consistency Profiling (Li et al., 2026)

Analyzes consistency of stylistic signals across a document. AI text tends to maintain more uniform stylistic levels throughout a document (consistent formality, consistent sentence structure), while human text shows natural variation and style drift.

---

## 6. Watermarking-Based Detection

Watermarking is the **only family with provable statistical guarantees** — but requires control of the generation process.

### 6.1 KGW Watermark (Kirchenbauer et al., 2023)

**Method**: At each token position $t$, hash $x_{t-1}$ to partition vocabulary $V$ into green list $G_t$ and red list $R_t$. Bias sampling toward $G_t$ by adding $\delta$ to green-token logits. Detection: count green tokens and compute z-statistic.

$$z = \frac{|G| - N/2}{\sqrt{N/4}} \sim \mathcal{N}(0,1) \text{ under null}$$

**Strengths**: Provable false-positive bounds; minimal quality impact; fast detection.  
**Weaknesses**: Fragile to paraphrasing; reverse-engineerable with many samples.

---

### 6.2 SynthID-Text (Dathathri et al., 2024 — Google DeepMind)

**Method**: Tournament sampling modifies the token selection process to embed a watermark signal. Detection uses a learned scoring function that does not require access to the generator model.

**Deployed at scale**: Google Gemini family.

**Strengths**: Deployed at scale; minimal quality impact; detection without generator weights.  
**Weaknesses**: Proprietary; model-provider cooperation required; still limited against heavy paraphrasing.

---

### 6.3 Undetectable Watermarks (Christ et al., 2024)

**Theoretical result**: Watermarks can be made computationally **undetectable to third parties** (cannot be distinguished from no-watermark text without the key) while remaining detectable by the key holder. Uses one-way cryptographic functions. Proves that strong security and strong detectability are simultaneously achievable.

---

### 6.4 GaussMark (Block et al., 2025)

**Method** *(new, open-source)*: A **structural watermark** embedded in the model's weight matrices rather than in the sampling process. Small Gaussian perturbations are added to specific weight matrices at generation time; detection computes the correlation between output text statistics and the perturbation pattern.

**Key advantage**: Works even when the generating model's sampling strategy is unknown or varied (temperature, top-p changes do not remove the signal).

**Strengths**: Robust to decoding strategy changes; open-source; does not require modifying the sampling procedure.  
**Weaknesses**: Requires modifying model weights (open-weight models only); not applicable to API-only models.

---

### 6.5 ECC Watermark — Error Correcting Codes (Chao et al., 2024)

**Method** *(new)*: Encodes a binary message using an error-correcting code (e.g., Reed-Solomon), then maps code bits to token selection biases. Detection recovers the message and checks parity.

**Key advantage**: The ECC structure provides **inherent robustness** — up to a certain fraction of tokens can be changed (by paraphrasing) and the watermark can still be detected.

**Strengths**: Mathematically principled robustness; multi-bit payload (not just present/absent); tunable robustness-capacity trade-off.  
**Weaknesses**: Lower per-token efficiency than KGW; requires planning ECC parameters at deployment time.

---

### 6.6 Robust Watermark Detection Under Human Edits (Li et al., 2024)

**Method** *(new)*: Derives optimal detection statistics when a human has edited a fraction of tokens in a watermarked text. Establishes a statistical test that remains powerful even after substantial editing — accepted in *Journal of the Royal Statistical Society: Series B*.

**Practical implication**: The watermark-then-edit scenario (student submits AI text with some edits) can still be detected with proper statistical testing.

---

### 6.7 Online Watermark Detection via E-Processes (Su et al., 2026)

**Method** *(new)*: Formulates watermark detection as a **sequential hypothesis testing** problem using e-processes (anytime-valid p-values). Allows detection as tokens arrive, with no fixed sample-size requirement.

**Why it matters**: Enables real-time detection during document streaming, and the test remains valid regardless of when the analyst decides to stop.

---

### 6.8 Mark My Words — Watermark Evaluation Framework (Piet et al., 2023/2024)

**Contribution**: Systematic evaluation of 8 watermarking schemes across quality metrics, detection power, and robustness to attacks. Found that: (1) no scheme is simultaneously robust and quality-preserving; (2) paraphrasing attacks break all tested schemes to varying degrees; (3) multi-bit schemes are more practical for real deployments.

---

## 7. Hybrid and Ensemble Approaches

### 7.1 The Case for Ensembles

No single detection signal is sufficient across all conditions. The table below shows how orthogonal signals complement each other:

| Signal | Paraphrase-robust? | Short-text? | Multilingual? | No model access? |
|--------|:-----------------:|:-----------:|:-------------:|:----------------:|
| Perplexity / Binoculars | ✗ | ✗ | Partial | ✗ |
| Alignment Imprint (LAPD) | Partial | ✗ | Partial | ✗ |
| PHD (topological) | ✓ | ✗ | ✓ | ✓ |
| Syntactic (DependencyAI) | ✓ | Partial | ✓ | ✓ |
| Stylometric | ✓ | ✗ | ✓ | ✓ |
| Multi-task neural (MELD) | ✓ | Partial | ✓ | ✓ |
| KGW watermark | ✓ (when present) | ✓ | ✓ | Needs key |

**Key insight**: PHD, syntactic, and stylometric features are most robust to paraphrasing — they should receive higher weight when a paraphrase attack is suspected.

### 7.2 Conformal Prediction for Calibrated Uncertainty

Standard classifiers output uncalibrated scores. **Split-conformal prediction** converts any classifier into a valid prediction set with coverage guarantee $1 - \alpha$:

$$C_\alpha(x) = \{y : s(x,y) \leq \hat{q}_{1-\alpha}\}$$

where $\hat{q}_{1-\alpha}$ is the $(1-\alpha)$-quantile of nonconformity scores on a held-out calibration set. This gives principled "UNCERTAIN" labels when the detector is genuinely unsure.

### 7.3 Retrieval-Augmented Detection

Maintaining a vector index of known AI outputs (from API honeypots, public benchmarks, monitored feeds) allows detection of **copy-paste reuse** without any statistical analysis. Near-duplicate retrieval is highly reliable and does not depend on model priors.

---

## 8. Vendor Analysis: Turnitin / iThenticate / GPTZero

### 8.1 Turnitin / iThenticate

**Publicly disclosed architecture elements**:

| Aspect | Disclosed | Not Disclosed |
|--------|-----------|---------------|
| Model type | Transformer-based deep learning | Layer count, parameters |
| Signals | Perplexity-related + learned features | Full feature set |
| Processing | Overlapping text segments | Segment size, overlap ratio |
| Granularity | Sentence-level scores (0–1) aggregated to document | Aggregation function |
| Thresholding | Scores < 20% suppressed (marked ✱) | Calibration methodology |
| Detection scope | AI-generated, AI-paraphrased, AI-bypassed | Bypass detection mechanism |
| Languages | English, Spanish, Japanese, others | Per-language accuracy |
| Training data | Not disclosed | Critical gap |

**Operational safeguards** (from public guides): Minimum prose word count required; scores should not be used as sole evidence; explicit caveats on non-native English false positives.

**Independent evaluation (Chaka, 2023)**: Variable accuracy depending on generating model and prompt type; non-negligible false positive rates for academic writing by non-native speakers.

---

### 8.2 GPTZero (Adam et al., 2026 — arXiv:2602.13042)

GPTZero published a detailed technical paper in 2026, the most transparent disclosure by any commercial AI detector:

**Disclosed architecture**:
- **Hierarchical multi-task architecture**: Enables flexible taxonomy of human/AI text with granular predictions
- **Statistical layer**: Perplexity + burstiness as first-pass signals
- **Neural layer**: Multi-task deep learning built on top of statistical signals
- **Robustness mechanism**: Multi-tiered automated red teaming with adversarial rewrites
- **Explainability**: Per-sentence scores + highlighted tokens

**Claims**: SOTA accuracy on a variety of domains; superior robustness to adversarial attacks and paraphrasing via red teaming; educates users on responsible use.

**Gaps in disclosure**: Training data specifics, red-teaming methodology details, and model architecture depth remain proprietary.

---

## 9. Benchmark Results and Comparative Performance

### 9.1 RAID Benchmark (Dugan et al., 2024)

The most comprehensive published evaluation. Tests detectors across:
- **8 generators** (GPT-4, Claude, Gemini, Mistral, etc.)
- **11 domains** (news, creative writing, reviews, academic, etc.)
- **4 attack types** (paraphrasing, synonym substitution, upper-casing, whitespace)
- **Included commercial detectors** (Turnitin, GPTZero, Originality.ai, Winston.ai)

**Key findings**:

| Condition | Best AUROC |
|-----------|-----------|
| Default (no attack) | > 0.95 for best detectors |
| Paraphrase attack | Drops to 0.55–0.75 for most |
| Model shift (train GPT-3.5, test GPT-4) | −10–30% |
| Domain shift (train news, test academic) | −15–25% |

**MELD on RAID leaderboard** (2026): Best open-source detector; competitive with leading commercial.

### 9.2 BLUFF Benchmark (Lucas et al., 2026)

First benchmark spanning **58 low-resource languages**. Key finding: Detection accuracy degrades substantially for non-English low-resource languages. Most detectors were effectively random on languages outside the top 10. Highlights a critical equity gap — AI-generated misinformation in minority languages is nearly undetectable.

### 9.3 C-ReD Chinese Benchmark (Qing et al., 2026)

First comprehensive Chinese AI-text detection benchmark built from real-world prompts across diverse domains. Showed that English-trained detectors transfer poorly to Chinese, and Chinese-specific stylometric features (character n-gram entropy, punctuation patterns) differ substantially from English analogs.

### 9.4 MELD-eval (Li et al., 2026)

Held-out evaluation pool of recent chat models from four major LLM providers (released after MELD's training cutoff):

| Detector | TPR @ 1% FPR |
|----------|:------------:|
| MELD | 99.9% |
| Most open-source baselines | 30–80% |
| Commercial detectors | Varies (not fully disclosed) |

### 9.5 Weber-Wulff et al. (2023) — Tool Comparison

Tested 14 commercial and open-source tools: No tool achieved both high precision and high recall. Translation and back-translation defeated most tools. Non-native English produced high false positive rates across all tools.

### 9.6 Summary Performance Table

| Method | AUROC (clean) | AUROC (paraphrased) | Access Needed | Open-Source |
|--------|:------------:|:-------------------:|:-------------:|:-----------:|
| DetectGPT | 0.85–0.95 | 0.55–0.70 | Logits | ✓ |
| Fast-DetectGPT | 0.85–0.95 | 0.55–0.70 | Logits | ✓ |
| LAPD / Alignment Imprint | +45% rel. vs Fast-DetectGPT | Partial | Base + aligned logits | ✓ |
| Binoculars | 0.90–0.98 | 0.60–0.75 | 2× LLM logits | ✓ |
| PHD (topological) | 0.82–0.90 | 0.70–0.82 | Embedding model | ✓ |
| DependencyAI | Competitive | Partial | Dependency parser | ✓ |
| RADAR | 0.88–0.95 | 0.70–0.85 | None (inference) | ✓ |
| MELD (2026) | RAID SOTA | Best open-source | None (inference) | ✓ |
| DRGD (2026) | +24.2% acc vs SOTA | High generalization | None (inference) | ✓ |
| Ghostbuster | 0.85–0.92 | 0.60–0.75 | Multiple LLMs | ✓ |
| Turnitin | ~0.90–0.95* | Unknown | None | ✗ |
| GPTZero | ~0.88–0.94* | ~0.60–0.70* | None | ✗ |
| KGW Watermark | 0.99+ | 0.70–0.85 | Generator control | ✓ |

*Vendor-reported or estimated from independent evaluations.

---

## 10. Strengths of Current Methods

### 10.1 Zero-Shot Methods
- **No training data needed**: Immediately applicable to new models
- **Theoretically grounded**: Statistical properties of LMs provide principled signal
- **Alignment Imprint (2026)**: New signal class that remains detectable as models improve

### 10.2 Trained Classifiers (2026 state)
- **MELD architecture**: Multi-task supervision forces learning of generator/domain/attack structure, not just binary patterns
- **DRGD**: Disentanglement explicitly addresses the generalization-to-unseen-generators problem
- **Adversarial training**: RADAR/MELD's EMA distillation provides durable robustness

### 10.3 Watermarking
- **Provable guarantees**: Statistical false-positive bounds remain valid
- **ECC watermarks**: Robustness to partial editing is now mathematically grounded
- **Structural watermarks (GaussMark)**: Not affected by sampling strategy changes

### 10.4 Structural/Linguistic Features
- **Paraphrase robustness**: Syntactic and stylometric signals survive surface rewording
- **No GPU required**: DependencyAI runs on CPU with off-the-shelf parsers
- **Multilingual**: Dependency parsers exist for 50+ languages

---

## 11. Weaknesses and Failure Modes

### 11.1 Theoretical Impossibility (Sadasivan et al., 2023)

As LLM outputs converge toward human text distributions, all post-hoc detectors must face a **fundamental trade-off**:

$$\text{TPR} + \text{TNR} \leq 1 + \exp\!\left(-D_{\text{TV}}(P_{\text{AI}} \| P_{\text{human}})\right)$$

As total variation distance $D_{\text{TV}} \to 0$ (models improve), the test becomes uninformative. **Watermarking is the only family that bypasses this limit.**

### 11.2 Paraphrasing Vulnerability

Even light paraphrasing (DIPPER at lexical level 40) reduces most statistical detectors to near-random performance. Krishna et al. (2024) showed retrieval is an effective defense, but only for known AI outputs — generative attacks on novel text remain difficult.

**2026 update**: Paraphrase resilience study (Shportko & Verbitsky, 2026) confirmed that multi-signal ensembles combining structural and neural features maintain AUROC > 0.75 under paraphrase attacks vs. single-signal AUROC < 0.60.

### 11.3 Bias Against Non-Native English (Liang et al., 2023)

AI detectors systematically flag non-native English as AI-generated, because non-native writers use simpler vocabulary and more formulaic structures — the same patterns detectors associate with machine text. This creates discriminatory outcomes in educational settings.

**No complete solution exists**, but bias-aware calibration (per-proficiency-level score correction) reduces disparate impact.

### 11.4 Short Text Failure

All statistical methods need sufficient text for signal accumulation. Below ~150–200 tokens: signals are unreliable. **Length-gating** (returning "INSUFFICIENT TEXT" rather than a score) is the only honest response.

### 11.5 Model Drift

Detectors trained on 2023 AI outputs may fail on 2025/2026 models. Continuous retraining is expensive and always lags. MELD-eval shows that even recent SOTA detectors can drop from 99.9% to 30–80% TPR on newly released models without re-training.

### 11.6 Mixed Human-AI Documents

When a document mixes human-written and AI-generated paragraphs, document-level detectors give misleading scores. Only sentence-level or token-level attribution methods (e.g., Turnitin's sentence scoring, or a CRF-based sequence labeler) address this.

### 11.7 Dataset Artifacts (Pudasaini et al., 2026)

High benchmark scores are often due to **dataset-specific artifacts** rather than robust detection of machine authorship. SHAP analysis shows that the most predictive features shift dramatically between datasets, indicating the classifier is fitting distributional quirks rather than genuine authorship signals.

### 11.8 Domain and Language Gaps

- Most published methods are English-only
- BLUFF (2026): 58 low-resource languages show near-random detection accuracy
- C-ReD (2026): Chinese requires language-specific features
- Legal, scientific, and code domains remain underexplored

---

## 12. Gaps in the Field (2026 Update)

| Gap | Status | Severity |
|-----|--------|----------|
| Low-resource multilingual detection | BLUFF (2026) maps the gap; no solution | Critical |
| AI-assisted vs. AI-generated distinction | No reliable method | High |
| Long-term temporal robustness | Not studied | High |
| Mixed human-AI attribution at token level | Research stage | High |
| Cross-domain generalization (legal, code) | Partial (DRGD, DependencyAI) | High |
| Calibrated uncertainty for deployment | Conformal methods exist; rarely deployed | Medium |
| Human-AI collaborative writing detection | No published method | Medium |
| Standardized evaluation (universal benchmark) | RAID is best; not fully universal | Medium |
| Open watermarking for closed models | Industry cooperation required | Medium |

---

## 13. Open-Source Implementation Pathways

This section maps each detection component to concrete, openly available resources.

### 13.1 Neural Classifier Foundation

| Component | Resource | License |
|-----------|----------|---------|
| Multilingual base encoder | `xlm-roberta-large` (HuggingFace) | MIT |
| Multi-task classifier architecture | MELD (arXiv:2605.06903) | CC BY 4.0 |
| Disentangled encoder | DRGD (github.com/PuXiao06/DRGD) | Open |
| Adversarial training | RADAR (arXiv:2307.03838) | Open |

### 13.2 Zero-Shot Statistical Signals

| Signal | Library / Implementation | Notes |
|--------|--------------------------|-------|
| Perplexity / Fast-DetectGPT | HuggingFace `transformers` | Any causal LM |
| Binoculars | GitHub: `baoguangsheng/fast-detect-gpt` | Two LLMs needed |
| LAPD / Alignment Imprint | arXiv:2604.16923 (code TBA) | Base + aligned LM |
| PHD (topological) | `ripser` + `persim` Python packages | GPU optional |
| DetectLLM | GitHub: `mbzuai-nlp/LLMDet` | Logit-based |

### 13.3 Structural / Linguistic Features

| Feature | Tool | Language coverage |
|---------|------|-------------------|
| Dependency parsing | `spacy` (en, es, zh, ar, de, fr, ja, ...) | 25+ languages |
| POS, NER, syntax depth | `spacy` pipelines | 25+ languages |
| Readability scores | `textstat` Python package | English |
| Stylometric features | NLTK + custom | Language-dependent |
| DependencyAI features | arXiv:2602.15514 | Multilingual |

### 13.4 Watermark Detection

| Watermark | Implementation | License |
|-----------|---------------|---------|
| KGW detector | `jwkirchenbauer/lm-watermarking` (GitHub) | MIT |
| SynthID-Text (public algorithm) | Google DeepMind paper (Nature, 2024) | Proprietary (algorithm public) |
| ECC Watermark | arXiv:2406.10281 | Open |
| GaussMark | arXiv:2501.13941 | Open |

### 13.5 Ensemble & Calibration

| Component | Resource |
|-----------|----------|
| CatBoost meta-learner | `catboost` Python package (Apache 2.0) |
| Conformal prediction | `MAPIE` Python library (BSD 3-Clause) |
| Calibration plots | `sklearn.calibration` |

### 13.6 Infrastructure

| Component | Tool | License |
|-----------|------|---------|
| REST API | FastAPI | MIT |
| Async queue | Celery + Redis | MIT / BSD |
| Vector search | pgvector (PostgreSQL extension) | MIT |
| Experiment tracking | MLflow | Apache 2.0 |
| Training orchestration | PyTorch + HuggingFace Trainer | BSD / Apache |

---

## 14. Recommendations for Building AEGIS

### 14.1 Architecture Principle

Build a **layered pipeline** where each layer addresses a specific failure mode:

```
Layer 1: Confidence Gate       → Refuse short texts honestly
Layer 2: Watermark Check       → High-confidence override if present
Layer 3: Zero-Shot Statistics  → Fast cross-check (Binoculars + LAPD)
Layer 4: Structural Features   → Paraphrase-robust signal (Dependency + PHD)
Layer 5: Neural Classifier     → Main signal (MELD / DRGD architecture)
Layer 6: Retrieval             → Near-duplicate catch
Layer 7: Ensemble Fusion       → CatBoost meta-learner
Layer 8: Conformal Calibration → Calibrated probability + UNCERTAIN label
Layer 9: Bias Correction       → Per-language / per-proficiency FPR correction
```

### 14.2 Open-Source Feasibility Assessment

| Component | Open-source feasible? | Main challenge |
|-----------|:---------------------:|----------------|
| Multi-task neural classifier (MELD-style) | ✓ Yes | Training data diversity |
| Fast-DetectGPT / Binoculars | ✓ Yes | Logit access to reference LMs |
| PHD topological signal | ✓ Yes | Compute cost for long docs |
| DependencyAI syntactic | ✓ Yes | None — trivial to implement |
| Stylometric features | ✓ Yes | Language-specific calibration |
| KGW watermark detection | ✓ Yes | Blind detection (no key) is heuristic |
| ECC watermark detection | ✓ Yes | Only works if ECC was used at generation |
| Conformal prediction calibration | ✓ Yes | Calibration set must match deployment distribution |
| Bias correction per language | ✓ Yes | Requires labeled per-language human text |
| Retrieval-augmented detection | ✓ Yes | Index freshness maintenance |

### 14.3 Prioritized Build Order

1. **Core neural classifier** (MELD architecture with multi-task heads) — highest ROI
2. **Statistical signals** (Binoculars + Fast-DetectGPT) — fast, zero-shot cross-check
3. **Structural signals** (DependencyAI + stylometric) — paraphrase robustness
4. **Ensemble fusion + conformal calibration** — reliability and uncertainty
5. **KGW watermark detector** — high-confidence flag when applicable
6. **Bias correction** — fairness requirement
7. **Retrieval-augmented layer** — catches copy-paste reuse
8. **Sentence-level attribution** — mixed-document support
9. **Continuous learning pipeline** — model drift management

### 14.4 Key Design Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Base encoder | `xlm-roberta-large` | Multilingual from day one; 560M params; proven |
| Multi-task heads | MELD-style (generator, attack, domain + binary) | Forces generalization |
| Loss balancing | Homoscedastic uncertainty weights | Avoids manual tuning |
| Calibration | Split-conformal (MAPIE) | Coverage guarantees, simple |
| Bias correction | Per-language FPR equalization | Fairness requirement |
| Short-text handling | Hard gate at 100 tokens; partial at 50–100 | Honest uncertainty |
| Serving | FastAPI + Celery + Redis | Standard, production-tested |
| Watermark | Implement KGW + ECC detectors | Only open algorithms |

---

## 15. Full Bibliography

### Surveys
1. Wu, J., et al. (2023). "A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions." arXiv:2310.14724. `literature/surveys/wu2023_survey.pdf`
2. Tang, R., et al. (2024). "A Survey on Detection of LLMs-Generated Content." arXiv:2310.15654. `literature/surveys/tang2024.pdf`
3. Crothers, E., et al. (2023). "Machine-Generated Text: A Comprehensive Survey of Threat Models and Detection Methods." arXiv:2210.07321. `literature/surveys/crothers2023.pdf`
4. Weber-Wulff, D., et al. (2023). "Detecting AI-Generated Text: Factors Influencing Detectability with Current Methods." arXiv:2306.05871. `literature/surveys/weber_wulff2023.pdf`

### Zero-Shot Detection
5. Mitchell, E., et al. (2023). "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature." ICML. `literature/detection/mitchell2023_detectgpt.pdf`
6. Bao, G., et al. (2023). "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature." arXiv:2310.05130. `literature/detection/bao2023_fast_detectgpt.pdf`
7. Gehrmann, S., et al. (2019). "GLTR: Statistical Detection and Visualization of Generated Text." ACL. `literature/detection/gehrmann2019_gltr.pdf`
8. Hans, A., et al. (2024). "Binoculars: Zero-Shot Detection of LLM-Generated Text." arXiv:2401.12070. `literature/detection/hans2024_binoculars.pdf`
9. Su, J., et al. (2023). "DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text." arXiv:2306.05540. `literature/detection/su2023_detectllm.pdf`
10. Yang, X., et al. (2023). "DNA-GPT: Divergent N-Gram Analysis with LLM Probability for Machine-Generated Text Detection." arXiv:2305.17359. `literature/detection/yang2023_dnagpt.pdf`
11. Wu, J., et al. (2026). "Alignment Imprint: Zero-Shot AI-Generated Text Detection via Provable Preference Discrepancy." arXiv:2604.16923. `literature/detection/alignment_imprint2026.pdf`

### Topological Methods
12. Tulchinskii, E., et al. (2024). "Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts." NeurIPS. arXiv:2306.04723. `literature/detection/tulchinskii2024_phd.pdf`

### Trained Classifiers
13. Hu, X., et al. (2023). "RADAR: Robust AI-Text Detection via Adversarial Learning." arXiv:2307.03838. `literature/detection/hu2023_radar.pdf`
14. Verma, V., et al. (2024). "Ghostbuster: Detecting Text Ghostwritten by Large Language Models." arXiv:2305.15047. `literature/detection/verma2024_ghostbuster.pdf`
15. Liang, W., et al. (2023). "GPT detectors are biased against non-native English writers." *Patterns*. arXiv:2304.02819. `literature/detection/liang2023_bias.pdf`

### New 2026 Classifiers & Structural Methods
16. Li, C., Wan, C., & Paetzold, J. (2026). "MELD: Multi-Task Equilibrated Learning Detector for AI-Generated Text." arXiv:2605.06903. `literature/detection/meld2026.pdf`
17. Pu, X., et al. (2026). "Breaking the Generator Barrier: Disentangled Representation for Generalizable AI-Text Detection." arXiv:2604.13692. `literature/detection/disentangled_detection2026.pdf`
18. Marchenko, O., et al. (2026). "NOTAI.AI: Explainable Detection of Machine-Generated Text via Curvature and Feature Attribution." arXiv:2603.05617. `literature/detection/notai_explainable2026.pdf`
19. Ahmed, S. & Hammond, T. (2026). "DependencyAI: Detecting AI Generated Text through Dependency Parsing." arXiv:2602.15514. `literature/detection/dependency_ai2026.pdf`
20. Pudasaini, S., et al. (2026). "Why AI-Generated Text Detection Fails: Evidence from Explainable AI Beyond Benchmark Accuracy." arXiv:2603.23146. `literature/detection/why_detection_fails2026.pdf`
21. Li, S., et al. (2026). "Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content." arXiv:2605.05950. `literature/detection/stylistic_consistency2026.pdf`

### Vendor / Industry
22. Adam, G. A., et al. (2026). "GPTZero: Robust Detection of LLM-Generated Texts." arXiv:2602.13042. `literature/detection/gptzero2026.pdf`
23. Chaka, C. (2023). "Detecting AI content in responses generated by ChatGPT, YouChat, and Chatsonic." *Journal of Applied Learning and Teaching*. `literature/vendor/chaka2023_turnitin_eval.pdf`

### Watermarking
24. Kirchenbauer, J., et al. (2023). "A Watermark for Large Language Models." ICML. arXiv:2301.10226. `literature/watermarking/kirchenbauer2023.pdf`
25. Dathathri, S., et al. (2024). "SynthID-Text: Scalable Watermarking for Identifying Large Language Model Outputs." *Nature*. `literature/watermarking/synthid2024.pdf`
26. Kuditipudi, R., et al. (2023). "Robust Distortion-Free Watermarks for Language Models." arXiv:2307.15593. `literature/watermarking/kuditipudi2023.pdf`
27. Christ, M., et al. (2024). "Undetectable Watermarks for Language Models." arXiv:2306.09194. `literature/watermarking/christ2024_undetectable.pdf`
28. Block, A., Sekhari, A., & Rakhlin, A. (2025). "GaussMark: A Practical Approach for Structural Watermarking of Language Models." arXiv:2501.13941. `literature/watermarking/gaussmark2025.pdf`
29. Chao, P., Sun, Y., Dobriban, E., & Hassani, H. (2024). "Watermarking Language Models with Error Correcting Codes." arXiv:2406.10281. `literature/watermarking/ecc_watermark2024.pdf`
30. Li, X., Ruan, F., Wang, H., Long, Q., & Su, W. (2024). "Robust Detection of Watermarks for Large Language Models Under Human Edits." arXiv:2411.13868. *JRSS-B*. `literature/watermarking/robust_under_edits2024.pdf`
31. Li, X., et al. (2025). "A Statistical Framework of Watermarks for Large Language Models: Pivot, Detection Efficiency and Optimal Rules." *Annals of Statistics*. arXiv:2404.01245.
32. Piet, J., et al. (2024). "Mark My Words: Analyzing and Evaluating Language Model Watermarks." arXiv:2312.00273. `literature/watermarking/mark_my_words2023.pdf`
33. Su, W., Wang, R., & Zhao, Z. (2026). "Online LLM watermark detection via e-processes." arXiv:2602.14286. `literature/watermarking/online_eprocess2026.pdf`

### Benchmarks & Evaluation
34. Dugan, L., et al. (2024). "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors." arXiv:2405.07940. `literature/benchmarks/dugan2024_raid.pdf`
35. Wang, Y., et al. (2024). "M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection." EACL. arXiv:2305.14902. `literature/benchmarks/wang2024_m4.pdf`
36. Weber-Wulff, D., et al. (2023). "Testing of Detection Tools for AI-Generated Text." arXiv:2306.15666. `literature/benchmarks/weber_wulff2023_testing.pdf`
37. Lucas, J., et al. (2026). "BLUFF: Benchmarking the Detection of False and Synthetic Content across 58 Low-Resource Languages." arXiv:2603.00634. `literature/benchmarks/bluff_multilingual2026.pdf`
38. Qing, C., et al. (2026). "C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts." arXiv:2604.11796. `literature/benchmarks/c_red_chinese2026.pdf`

### Adversarial & Theoretical
39. Sadasivan, V. S., et al. (2023). "Can AI-Generated Text be Reliably Detected?" arXiv:2303.11156. `literature/adversarial/sadasivan2023_impossibility.pdf`
40. Krishna, K., et al. (2024). "Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense." NeurIPS. arXiv:2303.13408. `literature/adversarial/krishna2024_dipper.pdf`
41. Shi, Z., et al. (2024). "How Reliable Are AI-Generated-Text Detectors? An Assessment Framework Using Evasive Soft Prompts." arXiv:2310.05095. `literature/adversarial/shi2024_evasive.pdf`
42. Shportko, A. & Verbitsky, I. (2026). "Paraphrasing Attack Resilience of Various AI-Generated Text Detection Methods." NAACL 2025. arXiv:2605.14240. `literature/adversarial/paraphrase_resilience2026.pdf`
