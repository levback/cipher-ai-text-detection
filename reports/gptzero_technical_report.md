# GPTZero: Comprehensive Technical Report
## Architecture, Algorithms, and Mathematical Foundations

**Source**: arXiv:2602.13042 — "GPTZero: Robust Detection of LLM-Generated Texts"  
**Authors**: George Alexandru Adam, Alexander Cui, Edwin Thomas, Emily Napier, Nazar Shmatko, Jacob Schnell, Jacob Junqi Tian, Alekhya Dronavalli, Edward Tian et al.  
**Organization**: GPTZero Inc.  
**Report Date**: 2026

---

## 1. Overview

GPTZero is a state-of-the-art commercial AI text detection system designed to distinguish human-authored text from AI-generated text at production scale. Initially released in January 2023 by Edward Tian as a public tool, GPTZero evolved into an industrial-grade platform and formalized its technical approach in a 2026 arXiv publication (arXiv:2602.13042).

The system is notable for three core contributions:
1. A **hierarchical, multi-task architecture** enabling a flexible taxonomy of human and AI texts
2. **State-of-the-art accuracy** across diverse domains with granular sentence-level predictions
3. **Superior robustness** to adversarial attacks and paraphrasing via multi-tiered automated red teaming

---

## 2. Core Problem Framing

### 2.1 The Detection Challenge

The fundamental challenge addressed by GPTZero is the binary (or graded) classification task:

$$\mathcal{F}: \mathcal{T} \rightarrow [0, 1]$$

Where $\mathcal{T}$ is the space of text documents and the output is the probability that a given text is AI-generated. In practice, this extends to a **sentence-level** task:

$$\mathcal{F}_s: \mathcal{S} \rightarrow [0, 1] \quad \forall s_i \in \text{doc}$$

Where each sentence $s_i$ in a document receives an individual AI-probability score.

### 2.2 Why AI Text is Distinguishable

Large Language Models (LLMs) generate text through autoregressive sampling from a learned probability distribution. For a token sequence $w_1, w_2, \ldots, w_T$:

$$P(w_1, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})$$

At each step, the model selects tokens from high-probability regions of its distribution. This produces text with characteristically:
- **High predictability** — next tokens tend toward maximum-likelihood choices
- **Low entropy** locally — probability mass concentrated on a small vocabulary
- **Consistent register** — stylistic uniformity across the document
- **Smooth log-probability curves** — absent the idiosyncratic variance of human writing

Human writing, by contrast, exhibits higher perplexity, more unpredictable word choices, and irregular stylistic variance — properties that GPTZero's classifier exploits.

---

## 3. System Architecture

### 3.1 High-Level Pipeline

```
Input Text
    │
    ▼
┌─────────────────────────┐
│  Text Preprocessing     │  Tokenization, cleaning, segmentation
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Hierarchical Multi-    │  Document-level + sentence-level tasks
│  Task Classification    │
│  Architecture           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Ensemble / Aggregation │  Multi-model ensemble, confidence scoring
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Red Team Validation    │  Adversarial robustness checks
└────────────┬────────────┘
             │
             ▼
Output: Document score + sentence-level highlights + explanation
```

### 3.2 Hierarchical Multi-Task Architecture

GPTZero's central innovation is a **hierarchical classification framework** that models the relationship between:
- **Document-level AI probability** — Is this document predominantly AI-generated?
- **Sentence-level AI probability** — Which specific sentences are AI-generated?
- **Source model identification** — Which LLM likely generated this text?
- **Mixed-authorship detection** — What fraction is human vs. AI?

Formally, the architecture solves:

$$\mathcal{L}_\text{total} = \lambda_1 \mathcal{L}_\text{doc} + \lambda_2 \mathcal{L}_\text{sent} + \lambda_3 \mathcal{L}_\text{src} + \lambda_4 \mathcal{L}_\text{mix}$$

Where $\mathcal{L}_\text{doc}$, $\mathcal{L}_\text{sent}$, $\mathcal{L}_\text{src}$, $\mathcal{L}_\text{mix}$ are the individual task losses and $\lambda_i$ are tunable weights that prioritize different tasks during training.

### 3.3 Foundation Model Backbone

GPTZero uses a large **pre-trained language model** as its encoder backbone (the specific architecture is proprietary but consistent with transformer-based models such as BERT/RoBERTa variants or larger decoder models used as feature extractors). The encoder produces contextualized representations:

$$\mathbf{H} = \text{Encoder}(w_1, w_2, \ldots, w_T) \in \mathbb{R}^{T \times d}$$

Where $d$ is the hidden dimension and $T$ is the sequence length.

**Sentence-level representation**: pooled over token positions within each sentence span $[i_{\text{start}}, i_{\text{end}}]$:

$$\mathbf{h}_{s_k} = \frac{1}{|s_k|} \sum_{t \in s_k} \mathbf{H}_t$$

**Document-level representation**: typically a CLS token or mean-pool over all tokens:

$$\mathbf{h}_{\text{doc}} = \text{MeanPool}(\mathbf{H}) \quad \text{or} \quad \mathbf{h}_{\text{doc}} = \mathbf{H}_{\text{[CLS]}}$$

---

## 4. Probabilistic Signal Features

### 4.1 Perplexity

**Perplexity** is one of the foundational signals used by GPTZero (and indeed was the basis of its earliest public version). For a text sequence $W = w_1, w_2, \ldots, w_N$:

$$\text{PPL}(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i \mid w_1, \ldots, w_{i-1})\right)$$

AI-generated text exhibits **low perplexity** because the model systematically chooses high-probability next tokens. Human text exhibits higher and more variable perplexity.

**Interpretation**:
- Low PPL → text is "expected" by the language model → likely AI-generated
- High PPL → text is "surprising" → likely human-written

### 4.2 Burstiness

**Burstiness** (also called perplexity variance or per-sentence perplexity fluctuation) measures the variance of perplexity across individual sentences:

$$\text{Burstiness} = \text{Var}\left(\{{\text{PPL}(s_k)}\}_{k=1}^{K}\right)$$

Human writers alternate between complex and simple sentences in an irregular pattern. AI models generate more uniformly-distributed sentence-level perplexities.

Mathematically, let $p_k = \text{PPL}(s_k)$ for sentence $k$:

$$\bar{p} = \frac{1}{K}\sum_{k=1}^{K} p_k, \quad \text{Burstiness} = \frac{1}{K}\sum_{k=1}^{K}(p_k - \bar{p})^2$$

**High burstiness** → more human-like; **Low burstiness** → more AI-like.

### 4.3 Log-Probability Curvature (DetectGPT-inspired)

Inspired by Mitchell et al. (2023), GPTZero incorporates **probability curvature** analysis. The key insight is that AI-generated text lies near **local maxima** of the model's log-probability function. Perturbations of AI text (via paraphrasing) tend to decrease log-probability, while perturbations of human text show no systematic directional change.

For a text $x$ and its perturbations $\tilde{x}_1, \ldots, \tilde{x}_m$ generated by a secondary model (e.g., T5):

$$d(x) = \log p_\theta(x) - \frac{1}{m}\sum_{i=1}^{m} \log p_\theta(\tilde{x}_i)$$

If $d(x) > 0$ consistently, the text occupies a local maximum of the log-probability landscape, indicating AI generation.

GPTZero uses a **signed test**:

$$\hat{y} = \mathbb{1}\left[\frac{d(x) - \mu_d}{\sigma_d} > \tau\right]$$

Where $\mu_d$, $\sigma_d$ are estimated from a held-out calibration set, and $\tau$ is a decision threshold.

---

## 5. Training Data and Methodology

### 5.1 Dataset Composition

GPTZero is trained on carefully curated datasets spanning:
- **Academic writing**: essays, dissertations, theses across diverse disciplines
- **News articles**: real journalism vs. GPT-generated news
- **Creative writing**: fiction, narrative prose
- **Technical writing**: scientific abstracts, reports
- **Non-native English**: writing from second-language speakers to reduce demographic bias
- **Multiple AI sources**: GPT-3.5, GPT-4, GPT-4o, Claude, Gemini, LLaMA, Mistral, and others

### 5.2 Taxonomy Labels

The hierarchical labeling scheme includes:
- **Fully human-written**
- **Fully AI-generated**
- **AI-generated + human-edited**
- **Human-written + AI-assisted**
- **AI paraphrased**
- **Adversarially bypassed** (humanized AI text)

### 5.3 Multi-Tiered Red Teaming

GPTZero implements **automated red teaming** at multiple tiers:

**Tier 1 – Direct AI output**: Standard GPT/Claude/Gemini completions  
**Tier 2 – AI paraphrasing**: AI text passed through word spinners (QuillBot, Grammarly, etc.)  
**Tier 3 – Humanization attacks**: Specialized "AI bypass" tools (e.g., GPTMinus, Undetectable.ai) that inject typos, restructure syntax, and vary vocabulary to defeat detectors  
**Tier 4 – Adversarial optimization**: Gradient-based attacks that iteratively modify text to maximize the probability of being classified as human  

Models are retrained after each red team round to improve robustness.

---

## 6. Scoring and Output

### 6.1 Document Score

The final document-level AI probability is computed as:

$$p_\text{AI}(\text{doc}) = \sigma\left(\mathbf{W}_\text{doc} \cdot \mathbf{h}_\text{doc} + b_\text{doc}\right)$$

Where $\sigma$ is the sigmoid function. Additionally, an ensemble of $M$ independent classifiers may be used:

$$p_\text{AI}^{\text{ensemble}} = \frac{1}{M}\sum_{m=1}^{M} p_\text{AI}^{(m)}(\text{doc})$$

### 6.2 Sentence-Level Scores

For each sentence $s_k$:

$$p_\text{AI}(s_k) = \sigma\left(\mathbf{W}_\text{sent} \cdot \mathbf{h}_{s_k} + b_\text{sent}\right) \in [0, 1]$$

Sentences with $p_\text{AI}(s_k) > \tau_\text{sent}$ are highlighted in the report as likely AI-generated.

### 6.3 Calibration

GPTZero applies **temperature scaling** (Guo et al., 2017) to calibrate confidence:

$$\hat{p} = \sigma(z / T)$$

Where $z$ is the pre-sigmoid logit and $T > 1$ is a temperature parameter learned post-training to align confidence scores with empirical accuracy.

---

## 7. Robustness and Adversarial Defense

### 7.1 Paraphrase Resistance

The multi-task training objective includes a **paraphrase consistency loss**:

$$\mathcal{L}_\text{para} = \text{KL}\left(p_\text{AI}(x) \| p_\text{AI}(\tilde{x})\right)$$

This encourages the model to produce consistent AI-probability scores for semantically equivalent text, regardless of surface-level paraphrasing.

### 7.2 Adversarial Training

GPTZero incorporates **adversarial examples** into training:

$$\mathcal{L}_\text{adv} = \mathcal{L}_\text{CE}(f(x_\text{adv}), y)$$

Where $x_\text{adv}$ is an adversarially perturbed version of $x$ designed to evade detection, and $y$ is the ground-truth AI-generated label.

---

## 8. Model Coverage

As of 2026, GPTZero detects text from:
- **OpenAI**: GPT-3, GPT-3.5, GPT-4, GPT-4o, GPT-4 Turbo, o1, o3, o4-mini
- **Anthropic**: Claude 2, Claude 3 (Haiku/Sonnet/Opus), Claude 3.5 Sonnet
- **Google**: Gemini Pro, Gemini 1.5, Gemini 2.0 Flash, Gemini 2.5 Pro
- **Meta**: LLaMA 2, LLaMA 3, LLaMA 3.1, LLaMA 3.3
- **Mistral**: Mistral 7B, Mixtral 8x7B
- **Others**: Cohere Command, AI21 Jurassic, and derivative tools

---

## 9. Physical Infrastructure

GPTZero operates as a **cloud-based SaaS platform** with the following infrastructure characteristics:

- **API-first architecture**: REST API accepting text inputs, returning JSON with scores
- **Asynchronous batch processing**: for large document volumes
- **LMS integrations**: Canvas, Blackboard, Moodle, Google Classroom, Microsoft Teams via LTI 1.3
- **PDF/DOCX parsing pipeline**: document extraction layer before text analysis
- **Model serving**: GPU-accelerated inference clusters (likely A100/H100 GPUs)
- **Latency**: single-document responses within seconds; batch within minutes
- **Data retention policy**: opt-out options for institutional users

---

## 10. Performance Metrics

| Metric | Value |
|--------|-------|
| Document-level AUROC | >0.95 (claimed state-of-the-art across benchmarks) |
| False positive rate | <1% on fully human-written documents |
| Sentence-level accuracy | Granular predictions competitive with document-level |
| Adversarial robustness | Superior to baselines on paraphrase + bypass attack suites |

---

## 11. Limitations and Failure Modes

- **Very short texts** (<150 words): insufficient signal for reliable detection
- **Non-English text**: primary capability is English; multilingual support limited
- **Domain-specific professional writing**: highly formulaic academic or legal language may trigger false positives
- **RLHF-tuned models**: models fine-tuned to mimic human style exhibit lower detection rates
- **Advanced humanization tools**: Tier 4 adversarial tools specifically designed to defeat classifiers remain a challenge

---

## 12. References

- Adam, G.A., et al. (2026). *GPTZero: Robust Detection of LLM-Generated Texts*. arXiv:2602.13042
- Mitchell, E., et al. (2023). *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature*. ICML 2023. arXiv:2301.11305
- Guo, C., et al. (2017). *On Calibration of Modern Neural Networks*. ICML 2017.
- Yang, X., et al. (2023). *DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text*. arXiv:2305.17359
- Tian, E. (2023). GPTZero — Detecting AI-generated text. *Princeton University senior thesis project*.
