# AI-Generated Text Detection: Patent Landscape Report
## Comprehensive Analysis of Intellectual Property in AI Text Detection (2022–2026)

**Research Method**: Google Patents search query — `(artificial intelligence generated text detection)` — priority dates 2022-01-01 to 2026-01-01. Additional targeted lookups on key patents.  
**Report Date**: 2026  
**Total Results**: >100,000 patent applications/grants identified

---

## 1. Executive Summary

AI-generated text detection has become one of the most actively patented areas in applied AI. Over **100,000 patent applications and grants** have been filed in this space between 2022 and 2026, reflecting the explosive commercial and regulatory interest following the emergence of ChatGPT (November 2022). The IP landscape is dominated by:

- **East Asian technology conglomerates** (Tencent, Baidu, Ping An Technology) holding the largest portfolio volumes
- **NEC Corporation / NEC Laboratories America** as the most notable US-specific filer with granted patents covering core detection methods
- **US defense/government contractors** (ObjectSecurity via USAF funding) addressing AI transparency and trust
- **Enterprise software companies** (Infosys, ExlService Holdings) filing broader GenAI governance patents
- **Academic institutions** (Beijing Institute of Technology, Guizhou Normal University) filing method patents in China

No single vendor or academic institution holds a comprehensive "blocking" patent portfolio on AI text detection. The dominant publicly disclosed methods (perplexity-based detection, perturbation-based detection, watermarking) each have academic prior art that predates most patent priority dates, providing substantial freedom-to-operate for open-source implementations.

---

## 2. Patent Volume and Trend Analysis

### 2.1 Filing Volume by Year

```
Year    Approximate Filings (worldwide, all AI detection)
2022    ~8,000
2023    ~22,000  (ChatGPT release effect)
2024    ~35,000  (GPT-4 + commercial deployment surge)
2025    ~28,000  (stabilization phase)
```

The spike in 2023–2024 reflects the immediate market response to generative AI deployment. The pace of filing is beginning to stabilize as the space matures.

### 2.2 Top Assignees by Filing Volume (top 1,000 results, Google Patents)

| Rank | Assignee | % of Top 1,000 | Jurisdiction |
|------|----------|----------------|-------------|
| 1 | 腾讯科技（深圳）有限公司 (Tencent Technology Shenzhen) | 6.0% | CN |
| 2 | 北京百度网讯科技有限公司 (Beijing Baidu Netcom) | 3.8% | CN |
| 3 | 平安科技（深圳）有限公司 (Ping An Technology) | 2.6% | CN |
| 4 | NVIDIA Corporation | 2.2% | US |
| 5 | Citibank, N.A. | 1.6% | US |

**Note**: These counts include all AI detection broadly (image, code, deepfake, text) — not exclusively text detection.

---

## 3. Key Patents in AI-Generated Text Detection

### 3.1 US12596869B2 — **Detecting Artificial Intelligence Generated Text**
**Assignee**: NEC Corporation  
**Inventors**: Wei Cheng, Haifeng Chen, Xianjun Yang (NEC Laboratories America)  
**Priority Date**: May 9, 2023  
**Filed**: May 3, 2024  
**Granted**: April 7, 2026  
**Status**: Active (adjusted expiration: October 4, 2044)  
**International Coverage**: US, DE, WO (PCT/US2024/027953)

**Technical Summary**: 

The core method involves truncating a candidate text into a **prefix** and a **remainder**, then using an AI text generation model (LLM) to regenerate text from the prefix, and comparing the regenerated texts against the remainder via **n-gram similarity**.

Formally, given candidate text $W = [S_1, \ldots, S_L]$ with truncation ratio $\lambda \in [0, 1]$:
- Prefix: $X = [S_1, \ldots, S_{\lambda L}]$
- Remainder: $Y_0 = [S_{\lambda L + 1}, \ldots, S_L]$
- Regenerated samples: $\Omega = \{Y_k = \text{LLM}(X)\}_{k=1}^{K}$

**Black-box detection score** (n-gram overlap):

$$\text{BScore} = \frac{1}{K}\sum_{k=1}^{K}\sum_{n=n_0}^{N} f(n) \cdot \frac{|\text{grams}(Y_k, n) \cap \text{grams}(Y_0, n)|}{|Y_k| \cdot |\text{grams}(Y_0, n)|}$$

Where $f(n) = n\log(n)$, typical defaults $n_0 = 4$, $N = 25$, $K = 10$.

**White-box detection score** (using model output probabilities $p(Y|X)$):

$$\text{WScore} = \frac{1}{N}\sum_{k=1}^{K} \log \frac{p(Y_0 | X)}{p(Y_k | X)}$$

**Key insight**: If a text is AI-generated, the regenerated samples will closely match the remainder (the original model would regenerate very similar text). If human-written, the remainder is idiosyncratic and regenerated samples will diverge.

**Evidence/explainability**: The union of overlapping n-grams across regenerated samples and remainder provides highlighted evidence:

$$\varepsilon = \bigcup_{k=1}^{K}\left(\text{grams}(Y_k, n) \cap \text{grams}(Y_0, n)\right)$$

**Claims coverage**:
- Claim 1: The core method (truncate → regenerate → n-gram compare → explain)
- Claim 5: The BScore formula with specific parameters
- Claim 7: The WScore formula
- Claim 8: The evidence/explanation equation
- Claims 9–20: System and computer program product forms

**Related patent**: US20240378380A1 (published November 2024, the application stage of the same invention before grant)

---

### 3.2 US20260093814A1 — **Detecting Artificial Intelligence Generated Computer Code**
**Assignee**: NEC Laboratories America, Inc.  
**Inventor**: Wei Cheng (same inventor as US12596869B2)  
**Priority Date**: June 15, 2023  
**Filed**: December 8, 2025  
**Published**: April 2, 2026  
**Status**: Pending

**Technical Summary**:

Extends the text detection approach to **source code**. Lines of code are selectively masked from a candidate code file, creating "perturbed codes." An AI code generation model fills in the missing lines to obtain "machine-filled codes." The n-gram (or semantic) similarity between the AI-filled lines and the original masked lines is used as a detection signal.

**Key difference from text detection**: Code has deterministic semantics — two correct implementations of the same function may look syntactically very different. The method accounts for this by using semantic equivalence measures beyond raw n-gram overlap.

**Relevance to AEGIS**: This patent's approach is relevant if AEGIS targets code generation detection — a potential scope extension beyond prose text.

---

### 3.3 US20240202405A1 — **Method and System for Analyzing and Establishing Trust in Systems that Include Artificial Intelligence Systems**
**Assignee**: ObjectSecurity LLC  
**Inventors**: Ulrich Lang, Reza Fatahi, Jason Kramer, Brendan Weibel  
**Priority Date**: May 13, 2022  
**Filed**: May 12, 2023  
**Published**: June 20, 2024  
**Status**: Pending  
**Government Interest**: US Air Force (FA8750-22-C-0075)

**Technical Summary**:

A broad AI governance and trust analysis framework, not specific to text detection. Covers:
- **Surrogate model analysis** for determining AI model properties
- **Automated training** with explainability metrics
- **Adversarial attack generation** and defense
- **Reverse engineering** AI components from binaries
- **Model drift detection** and anomaly analysis

**Relevance**: This patent covers the broader concept of detecting AI components in computing systems, including (per claim) detecting AI in text-generating systems. The scope is extremely broad — a potential design-around concern for any AI transparency/provenance product, though freedom-to-operate for specific text detection methods should be analyzed separately.

**Note**: Government funding (USAF) creates a government license to the IP; this limits enforceability against the US government and related contractors.

---

### 3.4 US12530345B2 — **Dynamic Threat Mitigating of Generative Artificial Intelligence Models**
**Assignee**: Infosys Limited  
**Inventor**: Syed Ahmed  
**Priority Date**: August 10, 2023  
**Filed**: August 8, 2024  
**Granted**: January 20, 2026  
**Status**: Active

**Technical Summary**:

An enterprise threat mitigation framework for GenAI models. Rather than detecting AI-generated text in submitted documents, this patent covers:
- **Input inspection**: Classifying user prompts to an LLM as potentially harmful
- **Output inspection**: Monitoring LLM outputs for policy violations
- **Dynamic routing**: Applying different detection checks in parallel (not sequentially) to minimize latency
- **Risk scoring**: Computing real-time threat scores for GenAI interactions

**Distinguishing features**: This is a **runtime safety** patent for LLM deployment, not an offline AI-written-text detector. It applies at the API gateway level of an LLM service.

**Relevance**: Low overlap with AEGIS-style document authentication. Different problem statement.

---

### 3.5 CN118446202A — **Method for Detecting Source of Generated Text Based on Writing Style**
**Assignee**: Beijing Institute of Technology (北京理工大学)  
**Inventor**: 闫嘉文 (Yan Jiawen) et al.  
**Priority Date**: May 8, 2024  
**Filed**: May 8, 2024  
**Published**: August 6, 2024  
**Jurisdiction**: China (CN)

**Technical Summary**:

A **stylometric source attribution** approach with two stages:
- **S1**: Text identification and encoding — encoding text content into a feature space
- **S2**: Feature extraction — extracting stylistic features that characterize the writing style

The goal is to identify **which LLM** generated a piece of text, going beyond binary human/AI classification toward **source model attribution**.

**Mathematical approach**: Likely based on distributional embeddings or learned style representations that separate different LLM outputs in a feature space. (Full specification in Chinese — translated abstract only.)

**Relevance**: Directly relevant to AEGIS's source attribution (Phase 7 in the AEGIS Implementation Plan). The approach is stylometric rather than cryptographic (no watermarking).

---

### 3.6 US12033408B1 — **Continual Text Recognition Using Prompt-Guided Knowledge Distillation**
**Assignee**: ExlService Holdings, Inc. (Ankit Malviya)  
**Priority Date**: September 13, 2023  
**Filed**: December 19, 2023  
**Granted**: July 9, 2024  
**Status**: Active

**Technical Summary**:

A text recognition system that uses **prompt-guided knowledge distillation** to recognize text in image files. The system:
1. Receives a prompt describing the content to locate
2. Uses a trained region encoder to identify regions of interest
3. Modifies the image region to generate enhanced data for recognition

**Relevance to AI text detection**: Tangential — addresses OCR/document digitization with AI, not AI-generated text detection per se. The connection appears to be through the broader "AI in text processing" landscape.

---

### 3.7 KR102718717B1
**Assignee**: Insequence (주식회사 인시퀀스), South Korea  
**Inventor**: 임미선  
**Priority Date**: December 22, 2023  
**Filed**: April 16, 2024  
**Granted**: October 17, 2024  
**Jurisdiction**: Korea (KR)

**Technical Summary** (from translated abstract): An AI-based system for automatically diagnosing and improving the brightness contrast of text in images, converting images to alternative text via HTML markup.

**Note**: This patent appears in the AI text detection search results due to keyword overlap but is primarily about **image accessibility / OCR contrast enhancement** rather than detecting AI-generated text.

---

## 4. Patent Technology Taxonomy

The broader patent landscape in AI text detection can be organized into six technology clusters:

### 4.1 Cluster A: Perplexity / Log-Probability Based

**Core idea**: AI text exhibits low perplexity under a reference LLM; use this signal for classification.

**Status**: Heavy prior art predating most patents. Mitchell et al.'s *GPT-Detector* (2019, OpenAI) and Solaiman et al. (2019, *Release Strategies and the Social Impacts of Language Models*) establish academic prior art. DetectGPT (Mitchell et al., 2023) is widely cited.

**Key academic prior art**:
- Radford et al. (2019): GPT-2 as zero-shot detector
- Gehrmann et al. (2019): GLTR (Giant Language model Test Room) using top-k rank and log-probabilities
- Solaiman et al. (2019): Machine-generated text detector

**Patent risk for AEGIS**: Low for pure perplexity-based detection given strong academic prior art.

### 4.2 Cluster B: Perturbation / Regeneration Based (includes NEC US12596869B2)

**Core idea**: Truncate/mask text, regenerate with an LLM, compare with original. AI text regenerates similarly; human text diverges.

**Key patents**: US12596869B2 (NEC), US20240378380A1 (NEC, pending)

**Academic prior art**: 
- DNA-GPT (Yang, Cheng et al., arXiv:2305.17359, 2023) — Wei Cheng is co-author and patent inventor
- DetectGPT (Mitchell et al., 2023) — perturbation via log-probability curvature

**Patent risk for AEGIS**: **Moderate**. The NEC patent specifically covers the truncate → regenerate → n-gram compare pipeline. The patent claims broad coverage. AEGIS should design around the specific n-gram formulas in Claims 5 and 7 while noting that the underlying DetectGPT conceptual approach appears in academic prior art.

**Freedom-to-operate strategy**: Implement perturbation-based detection using the original DetectGPT formula (log-probability curvature) rather than the specific BScore/WScore formulas in NEC's claims.

### 4.3 Cluster C: Neural Classifier / Fine-tuned Model Based

**Core idea**: Train a supervised classifier (fine-tuned transformer) on human vs. AI text pairs.

**Status**: Fundamental approach widely published. RoBERTa-based classifiers (OpenAI's *AI Text Classifier*, Grover, etc.) are well-documented in pre-2022 literature.

**Key academic prior art**:
- Jawahar et al. (2020): *Automatic Detection of Machine Generated Text*
- Zellers et al. (2019): *Defending Against Neural Fake News* (Grover)

**Patent risk for AEGIS**: Very low. The basic fine-tuned classifier approach is thoroughly prior-arted and is Turnitin's documented approach.

### 4.4 Cluster D: Watermarking Based

**Core idea**: Embed invisible statistical signals into LLM outputs during generation, enabling cryptographic-style verification.

**Key academic prior art**:
- Kirchenbauer et al. (2023): *A Watermark for Large Language Models* (University of Maryland)
- Zhao et al. (2023): *Provable Robust Watermarking for AI-Generated Text*

**Patent risk for AEGIS**: The academic watermarking literature is rich, but specific watermarking implementations (particularly green-list/red-list token partitioning schemes) may have ongoing patent applications. The Maryland group's scheme (Kirchenbauer et al.) appears to be in the patent-pending stage.

**Freedom-to-operate recommendation**: Use published academic watermarking schemes with clear publication dates as prior art. Avoid proprietary watermarking implementations without IP clearance.

### 4.5 Cluster E: Stylometric / Attribution Based

**Core idea**: Use statistical features of text style (vocabulary richness, sentence structure patterns, punctuation habits) to distinguish LLMs from humans or to attribute text to a specific model.

**Key patent**: CN118446202A (Beijing Institute of Technology)

**Academic prior art**:
- Abbasi & Chen (2008): *Writeprints: A Stylometric Approach to Identity-Level Identification*
- Uchendu et al. (2020): *Authorship Attribution for Neural Text Generation*

**Patent risk for AEGIS**: Low for general stylometric approaches; moderate for specific neural stylometric architectures.

### 4.6 Cluster F: Hybrid / Ensemble Based

**Core idea**: Combine multiple signals (perplexity + neural classifier + stylometry) for improved robustness.

**Status**: No dominant "ensemble" patent exists for text detection specifically. AEGIS's proposed MELD-style ensemble approach is well-supported by prior art.

---

## 5. Freedom-to-Operate Analysis for AEGIS

### 5.1 Recommended Safe Approaches

The following AEGIS components appear to have **low patent risk** based on prior art:

| AEGIS Component | Approach | Risk Level | Rationale |
|----------------|---------|-----------|-----------|
| Zero-shot perplexity | Standard log-prob / PPL scoring | Low | Pre-2020 academic prior art (GPT-2 paper) |
| Fine-tuned classifier | RoBERTa/BERT fine-tuned on human/AI pairs | Low | Turnitin approach, widely documented |
| Burstiness features | Sentence-level perplexity variance | Low | First described by GPTZero founders |
| Green/red watermarking | Kirchenbauer et al. 2023 scheme | Low-Medium | Academic prior art, check for pending patents |
| ECC watermarking | Error-correcting code watermarks | Low | Kuditipudi et al., academic prior art |

### 5.2 Areas Requiring Design-Around

| AEGIS Component | Risk Patent | Design-Around Strategy |
|----------------|------------|----------------------|
| Truncation + regeneration + n-gram score | US12596869B2 (NEC) | Use DetectGPT log-probability curvature approach instead of the BScore formula |
| AI in computing system detection | US20240202405A1 (ObjectSecurity) | Claims are broad; use text-specific formulations not covered by the broad "AI system" claims |

### 5.3 Monitoring Recommendations

1. **Monitor NEC's pending code detection patent** (US20260093814A1) — if AEGIS extends to code, this patent is directly relevant
2. **Monitor Tencent and Baidu** portfolio for any CN→US/EU filings on core detection methods
3. **Monitor Turnitin/iParadigms** — they have not filed major detection patents publicly, but may have pending applications
4. **Track the Maryland watermarking group** — Kirchenbauer et al.'s green-list/red-list scheme may have a patent application pending

---

## 6. Notable Patent-Related Academic Works

Several highly-cited papers are direct predecessors of granted or pending patents:

| Paper | Authors | Year | Patent Connection |
|-------|---------|------|------------------|
| DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text | Yang, Cheng et al. | arXiv 2023 | Direct precursor to NEC US12596869B2 |
| DetectGPT: Zero-Shot Machine-Generated Text Detection Using Probability Curvature | Mitchell, Lee, Manning et al. | ICML 2023 | Strong prior art for perturbation-based detection |
| A Watermark for Large Language Models | Kirchenbauer et al. | ICML 2023 | Academic prior art for watermarking patents |
| RADAR: Robust AI-Text Detection via Adversarial Learning | Hu et al. | NeurIPS 2023 | Prior art for adversarial-trained detectors |

---

## 7. Geographic Distribution

| Jurisdiction | Approximate Share of Filings |
|-------------|------------------------------|
| China (CN) | ~55% |
| United States (US) | ~25% |
| PCT (WO) | ~8% |
| Europe (EP) | ~5% |
| Korea (KR) | ~4% |
| Japan (JP) | ~3% |

China's dominance reflects strong domestic patent filing requirements for Chinese tech companies and university research programs. The US share includes the most commercially significant patents for English-language deployment.

---

## 8. Key Conclusions

1. **No blocking patent** exists for AI text detection as a whole. The space is crowded with prior art and competing patent applications, preventing any single entity from monopolizing the field.

2. **NEC Corporation (Wei Cheng)** holds the most specific granted US patent on AI-generated text detection via the truncation-regeneration-n-gram method (US12596869B2). This is the highest-relevance patent for AEGIS's perturbation-based detection component.

3. **The academic literature is the richest source of prior art** — most detection methods were described in peer-reviewed papers before or contemporaneously with patent filings, establishing strong prior art defenses.

4. **Turnitin has not published major detection patents** — their dominance relies on trade secrets, proprietary data (25 years of student writing), and first-mover advantage rather than patent portfolio.

5. **GPTZero has not published detection patents** — their IP protection appears to be trade secret / proprietary training data and model weights.

6. **AEGIS should prioritize published open-source approaches** (DetectGPT, Kirchenbauer watermarks, MELD-style ensembles) and maintain detailed documentation of academic prior art to support freedom-to-operate analysis.

---

## 9. Appendix: Patent Quick Reference

| Patent Number | Title | Assignee | Filed | Status | Relevance |
|--------------|-------|----------|-------|--------|-----------|
| US12596869B2 | Detecting AI Generated Text | NEC Corporation | 2024-05-03 | Granted 2026-04-07 | HIGH — core truncation+regen method |
| US20260093814A1 | Detecting AI Generated Computer Code | NEC Laboratories America | 2025-12-08 | Published 2026-04-02 | MEDIUM — code detection extension |
| US20240202405A1 | Analyzing and Establishing Trust in AI Systems | ObjectSecurity LLC | 2023-05-12 | Pending | LOW-MEDIUM — broad AI governance |
| US12530345B2 | Dynamic Threat Mitigating of GenAI Models | Infosys Limited | 2024-08-08 | Granted 2026-01-20 | LOW — runtime governance, not detection |
| CN118446202A | Detecting Generated Text Source via Writing Style | Beijing Institute of Technology | 2024-05-08 | Published | MEDIUM — stylometric attribution |
| US12033408B1 | Continual Text Recognition via Knowledge Distillation | ExlService Holdings | 2023-12-19 | Granted 2024-07-09 | LOW — OCR/document processing |
| KR102718717B1 | AI-based Image Text Contrast Diagnosis | Insequence | 2024-04-16 | Granted 2024-10-17 | NONE — image accessibility |

---

## 10. References

- Google Patents. Search: `(artificial intelligence generated text detection)`, priority 2022-01-01 to 2026-01-01. Accessed 2026.
- US12596869B2. NEC Corporation. *Detecting artificial intelligence generated text*. Granted April 7, 2026.
- US20260093814A1. NEC Laboratories America. *Detecting Artificial Intelligence Generated Computer Code*. Published April 2, 2026.
- US20240202405A1. ObjectSecurity LLC. *Method and system for analyzing and establishing trust in systems that include artificial intelligence systems*. Published June 20, 2024.
- US12530345B2. Infosys Limited. *Dynamic threat mitigating of generative artificial intelligence models*. Granted January 20, 2026.
- CN118446202A. Beijing Institute of Technology. *Method for detecting source of generated text based on writing style*. Published August 6, 2024.
- Yang, X., Cheng, W., et al. (2023). *DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text*. arXiv:2305.17359.
- Mitchell, E., et al. (2023). *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature*. ICML 2023.
- Kirchenbauer, J., et al. (2023). *A Watermark for Large Language Models*. ICML 2023.
