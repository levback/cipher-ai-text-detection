
---

## File 2: `report_comprehensive.md` — Full Research Report

```markdown
# State of the Art in AI-Generated Text Detection
## Comprehensive Research Report

**Date**: May 2025
**Scope**: Academic literature, vendor disclosures (Turnitin, iThenticate, GPTZero)
**Sources**: Open-access publications, vendor whitepapers, independent evaluations

---

## Table of Contents

1. Executive Summary
2. Method Taxonomy
3. Zero-Shot Statistical Methods
4. Trained Neural Classifiers
5. Watermarking-Based Detection
6. Hybrid and Ensemble Approaches
7. Vendor Analysis: Turnitin / iThenticate / GPTZero
8. Benchmark Results and Comparative Performance
9. Strengths of Current Methods
10. Weaknesses and Failure Modes
11. Gaps in the Field
12. Methods to Empower Detection and Reduce Weaknesses
13. Conclusion
14. Full Bibliography

---

## 1. Executive Summary

AI text detection is a rapidly evolving field facing a fundamental asymmetry: **generating convincing AI text is computationally cheap, while reliably detecting it is provably hard** in the general case (Sadasivan et al., 2023). Current methods fall into four families:

| Family | Example | Requires Source Model? | Robustness |
|--------|---------|----------------------|------------|
| Zero-shot statistical | DetectGPT, Binoculars | Partially (surrogate OK) | Low-Medium |
| Trained classifiers | RADAR, OpenAI Classifier | No | Medium |
| Watermarking | SynthID-Text, KGW | Yes (at generation) | High (if deployed) |
| Hybrid/ensemble | GPTZero, Turnitin | No | Medium |

**Key finding**: No post-hoc detection method achieves simultaneously high accuracy, low false positives, and robustness to paraphrasing. Watermarking is the only family with provable guarantees, but requires model-provider cooperation. The field's biggest gaps are: (1) multilingual detection, (2) mixed human-AI text, (3) domain generalization, and (4) theoretical impossibility barriers.

---

## 2. Method Taxonomy

### 2.1 Classification Framework
AI Text Detection Methods
├── Post-hoc Detection (applied after generation)
│ ├── Zero-shot / Statistical
│ │ ├── Perplexity-based (GLTR, GPTZero v1)
│ │ ├── Perturbation-based (DetectGPT, Fast-DetectGPT)
│ │ ├── Log-rank based (DetectLLM)
│ │ ├── Cross-model ratio (Binoculars)
│ │ └── Topological (PHD / intrinsic dimension)
│ ├── Trained Classifiers
│ │ ├── Fine-tuned LMs (RoBERTa-based, OpenAI detector)
│ │ ├── Adversarially trained (RADAR)
│ │ ├── Feature-engineered (Ghostbuster)
│ │ └── Commercial (Turnitin, GPTZero, Originality.ai)
│ └── Hybrid / Ensemble
│ ├── Multi-signal fusion (GPTZero production)
│ └── Human-in-the-loop (GLTR + human judgment)
└── Proactive Detection (applied during generation)
├── Watermarking
│ ├── Red/green list (Kirchenbauer et al.)
│ ├── Sampling-based (SynthID-Text)
│ ├── Adaptive (Kuditipudi et al.)
│ └── Undetectable (Christ et al.)
└── Retrieval-based (search for known AI outputs)


### 2.2 Key Dimensions for Comparison

- **Access requirements**: White-box (needs model weights), API-access (needs logits), black-box (text only)
- **Training required**: Zero-shot vs. supervised
- **Granularity**: Document-level, paragraph-level, sentence-level
- **Source model specificity**: Model-specific vs. model-agnostic
- **Robustness**: To paraphrasing, translation, editing, adversarial attacks

---

## 3. Zero-Shot Statistical Methods

### 3.1 GLTR (Gehrmann et al., 2019)

**Method**: Visualizes token-level likelihood ranks under a reference LM (GPT-2). Highlights tokens in color bands based on their rank in the model's predicted distribution.

**Mathematical basis**: For token $x_t$, compute rank $r(x_t) = |\{v \in V : P(v|x_{<t}) > P(x_t|x_{<t})\}|$. AI text concentrates tokens in top-$k$ ranks.

**Strengths**:
- Interpretable, visual output
- No training required
- Works with any reference model

**Weaknesses**:
- Requires human judgment (not fully automated)
- Reference model mismatch degrades performance
- Easily defeated by sampling with high temperature

### 3.2 DetectGPT (Mitchell et al., 2023)

**Method**: Perturbs input text using a mask-filling model (T5), then measures whether the original text sits at a local maximum of the source model's log-probability surface. AI-generated text tends to occupy negative curvature regions.

**Mathematical basis**: The perturbation discordance score is:

$$d(x, p_\theta, q) = \log p_\theta(x) - \mathbb{E}_{\tilde{x} \sim q(\cdot|x)}[\log p_\theta(\tilde{x})]$$

where $p_\theta$ is the source model and $q$ is the perturbation function. AI text yields $d > 0$ (the original has higher likelihood than perturbations).

**Strengths**:
- Zero-shot: no training data needed
- Theoretically motivated
- Works across domains without retraining

**Weaknesses**:
- **Slow**: requires 50-100 perturbations per sample
- Requires access to source model logits (or good surrogate)
- Performance degrades when surrogate differs from source
- Paraphrasing attacks reduce the curvature signal

### 3.3 Fast-DetectGPT (Bao et al., 2023)

**Method**: Replaces the expensive perturbation step with conditional probability estimation. Instead of generating perturbations, directly estimates the expected log-probability under perturbations using the scoring model itself.

**Mathematical basis**: Approximates the perturbation discordance using:

$$\hat{d}(x) \approx \frac{1}{N}\sum_{t=1}^{N} \left(\log p_\theta(x_t|x_{<t}) - \sum_{v \in V} p_\theta(v|x_{<t}) \log p_\theta(v|x_{<t})\right)$$

This is the difference between observed log-likelihood and the entropy at each position — no perturbations needed.

**Strengths**:
- ~340x faster than DetectGPT
- Comparable or better detection accuracy
- Still zero-shot

**Weaknesses**:
- Still requires model logits
- Same fundamental limitation: curvature signal diminishes with sophisticated generation

### 3.4 Binoculars (Hans et al., 2024)

**Method**: Uses two different LLMs and computes the ratio of cross-perplexity to perplexity. Human text has a high ratio (both models are "confused" similarly), AI text has a low ratio (the generating model's statistics are "too clean").

**Mathematical basis**: Score = $\frac{\text{CE}(x; M_1)}{\text{PPL}(x; M_2)}$ where $M_1$ and $M_2$ are two different LLMs.

**Strengths**:
- **State-of-the-art zero-shot performance** at time of publication
- No training data needed
- Model-agnostic (works across different source models)
- Simple to implement

**Weaknesses**:
- Requires two capable LLMs (computational cost)
- Assumes both observer models are "surprised" differently by human vs. AI text
- Performance on very short texts is limited

### 3.5 PHD — Persistent Homology Dimension (Tulchinskii et al., 2024)

**Method**: Computes the intrinsic dimensionality of text embeddings using persistent homology (topological data analysis). AI-generated text has lower intrinsic dimension than human text because it occupies a lower-dimensional manifold.

**Strengths**:
- Fundamentally different approach (topological, not probabilistic)
- Robust to paraphrasing (topology is more stable than token statistics)
- Language-agnostic in principle

**Weaknesses**:
- Computationally expensive for long documents
- Requires embedding model choice (introduces a dependency)
- Less mature than statistical approaches

### 3.6 DNA-GPT (Yang et al., 2023)

**Method**: Asks the suspected source model to re-generate text from truncated prompts and compares n-gram overlap. If the original text was AI-generated, re-generations will share unusually high n-gram overlap.

**Strengths**:
- Black-box (only needs API access, not logits)
- Intuitive: AI text is more "reproducible"

**Weaknesses**:
- Requires multiple API calls (cost, latency)
- Doesn't work if the model has been updated
- Temperature/sampling changes reduce overlap

---

## 4. Trained Neural Classifiers

### 4.1 RoBERTa-based Classifiers (OpenAI, 2019)

**Method**: Fine-tune RoBERTa (or similar encoder) on (human, AI) text pairs as a binary classifier.

**Strengths**:
- High accuracy on in-distribution data
- Fast inference (single forward pass)
- Can detect subtle patterns invisible to statistical tests

**Weaknesses**:
- **Distribution shift**: Accuracy drops when the source model or domain changes
- **Training data bias**: Encodes biases from training corpus (see: Liang et al., 2023 — biased against non-native English writers)
- Requires labeled data, which becomes stale as new models emerge
- Black-box: no interpretable explanation for predictions

### 4.2 RADAR (Hu et al., 2023)

**Method**: Adversarial training framework where a paraphraser tries to fool the detector and the detector adapts. Iterates between:
1. Train paraphraser to evade detector
2. Train detector on paraphrased outputs

**Strengths**:
- Explicitly addresses paraphrasing robustness
- Demonstrates improved resilience vs. standard classifiers
- Generalizes better to unseen models

**Weaknesses**:
- Training is expensive (adversarial loop)
- Paraphraser quality limits the robustness ceiling
- Still vulnerable to out-of-distribution attacks not seen during training

### 4.3 Ghostbuster (Verma et al., 2024)

**Method**: Structured search over probability features from multiple reference models, combined with a linear classifier. Uses features like: probability under model A, probability under model B, ratio of A/B, etc.

**Strengths**:
- Interpretable (linear model with meaningful features)
- Good generalization across models
- Efficient at inference time

**Weaknesses**:
- Requires multiple reference models
- Feature engineering may not capture all detection signals
- Linear model has limited capacity for complex patterns

---

## 5. Watermarking-Based Detection

### 5.1 KGW Watermark (Kirchenbauer et al., 2023)

**Method**: At each token position, hash the previous token to partition the vocabulary into "green" and "red" lists. Bias sampling toward green tokens. Detection: count the proportion of green tokens (significantly above 50% → watermarked).

**Mathematical basis**: For token $x_t$, define green set $G_t = \text{Hash}(x_{t-1})$. Score: $z = \frac{|G| - N/2}{\sqrt{N/4}}$ follows $\mathcal{N}(0,1)$ under null (no watermark).

**Strengths**:
- **Provable false positive guarantees** (statistical test)
- Minimal quality degradation (empirically)
- Detection without access to model weights
- Fast detection (count green tokens)

**Weaknesses**:
- Requires control of the generation process
- Fragile to paraphrasing (green tokens get replaced)
- Quality impact debated (especially for code, structured text)
- Can be reverse-engineered if many samples are available

### 5.2 SynthID-Text (Google DeepMind, 2024)

**Method**: Modifies the sampling process using Tournament Sampling — embeds a watermark signal in the token selection probabilities. Detection via a learned scoring function.

**Strengths**:
- **Deployed at scale** (Google Gemini)
- Minimal quality impact
- Does not require knowing the model at detection time
- More robust than KGW to some modifications

**Weaknesses**:
- Proprietary (exact algorithm not fully open)
- Requires model-provider cooperation
- Robustness to heavy paraphrasing still limited

### 5.3 Undetectable Watermarks (Christ et al., 2024)

**Method**: Proves that watermarks can be made computationally undetectable to third parties while remaining detectable by the watermark key holder. Uses cryptographic one-way functions.

**Significance**: This is a theoretical result showing watermarks can be both secure AND undetectable, addressing the concern that watermarks could be discovered and removed.

---

## 6. Hybrid and Ensemble Approaches

Most commercial systems use ensemble methods combining multiple signals:

| Signal Type | Examples |
|-------------|----------|
| Statistical | Perplexity, burstiness, entropy patterns |
| Neural | Fine-tuned classifier scores |
| Structural | Sentence length variance, vocabulary diversity |
| Stylometric | Writing style features, syntactic patterns |

The key advantage of ensembles is resilience — if one signal fails, others may still detect. The disadvantage is opacity: hard to explain which signal triggered detection.

---

## 7. Vendor Analysis

### 7.1 Turnitin / iThenticate

**Disclosed architecture** (from whitepaper and documentation):

| Aspect | What's Disclosed | What's Not |
|--------|-----------------|------------|
| Model type | Transformer-based deep learning | Architecture details, layer count, parameters |
| Input processing | Overlapping text segments | Segment size, overlap ratio |
| Scoring | Sentence-level (0–1), aggregated to document | Aggregation function |
| Signals | "Perplexity-related" + learned features | Full feature set |
| Training data | Not disclosed | Critical gap |
| Thresholding | Scores <20% suppressed | Calibration methodology |
| Detection types | AI-generated, AI-paraphrased, AI-bypassed | How bypass detection works |
| Language support | English, Spanish, Japanese, others | Per-language accuracy |

**Strengths of Turnitin's approach**:
- Sentence-level granularity (useful for mixed documents)
- Operational safeguards (thresholds, caveats, minimum word counts)
- Continuous model updates for new LLMs
- Integration with existing plagiarism detection workflow

**Weaknesses**:
- Proprietary: no independent verification of accuracy claims
- 20% threshold means short AI passages go undetected
- Known false positive issues with non-native English (Liang et al., 2023)
- No published performance on adversarial attacks
- iThenticate uses same engine — inherits all limitations

**Independent evaluation** (Chaka, 2023): Found Turnitin's AI detector had variable accuracy depending on the generating model and prompt type. False positive rates were non-negligible for academic writing by non-native speakers.

### 7.2 GPTZero

**Disclosed architecture** (from Tian et al., arXiv + company blog):

| Aspect | What's Disclosed |
|--------|-----------------|
| Statistical layer | Perplexity + burstiness as initial signals |
| Neural layer | Hierarchical multi-task model |
| Robustness | Multi-tier automated red teaming |
| Granularity | Sentence and document level |
| Training | Continuous retraining on new LLM outputs |

**Mathematical basis** (disclosed):
- **Perplexity**: $\text{PPL}(x) = \exp\left(-\frac{1}{N}\sum_{t=1}^{N}\log p(x_t|x_{<t})\right)$ — AI text has lower perplexity
- **Burstiness**: Variance of per-token surprisal across the document — AI text has lower burstiness (more uniform)

**Strengths**:
- More transparent than Turnitin (published paper)
- Multi-signal approach (not purely statistical)
- Red teaming for robustness
- Free tier enables public scrutiny

**Weaknesses**:
- Paper lacks rigorous ablation on adversarial robustness
- Independent benchmarks (RAID) show fragility under distribution shift
- Perplexity/burstiness are easily gamed by adjusting generation parameters
- Limited multilingual capability

---

## 8. Benchmark Results and Comparative Performance

### 8.1 RAID Benchmark (Dugan et al., 2024)

The most comprehensive evaluation to date. Key findings:

**Under default conditions** (no attacks):
- Best detectors achieve >95% AUROC
- Binoculars and RADAR perform well across models
- Commercial detectors (Turnitin, GPTZero) perform comparably

**Under adversarial conditions**:
- Paraphrasing reduces most detectors to near-random (~50% AUROC)
- Model shift (training on GPT-3.5, testing on GPT-4) drops accuracy 10-30%
- Domain shift (training on news, testing on academic) drops accuracy 15-25%
- Decoding strategy changes (nucleus sampling, temperature) affect statistical methods most

### 8.2 Weber-Wulff et al. (2023) — Tool Comparison

Tested 14 commercial and open-source tools:
- **No tool achieved both high precision and high recall**
- Translation to another language and back defeated most tools
- Non-native English writing produced high false positive rates
- Short texts (<250 words) were unreliable across all tools

### 8.3 Summary Performance Table

| Method | AUROC (default) | AUROC (paraphrased) | Speed | Cost |
|--------|----------------|--------------------:|-------|------|
| DetectGPT | 0.85-0.95 | 0.55-0.70 | Slow | High (perturbations) |
| Fast-DetectGPT | 0.85-0.95 | 0.55-0.70 | Fast | Medium (logits) |
| Binoculars | 0.90-0.98 | 0.60-0.75 | Medium | High (2 LLMs) |
| RADAR | 0.88-0.95 | 0.70-0.85 | Fast | Low (single pass) |
| Ghostbuster | 0.85-0.92 | 0.60-0.75 | Fast | Medium |
| PHD | 0.82-0.90 | 0.70-0.82 | Slow | Medium |
| Turnitin | ~0.90-0.95* | Unknown | Fast | Commercial |
| GPTZero | ~0.88-0.94* | ~0.60-0.70* | Fast | Freemium |
| KGW Watermark | 0.99+ | 0.70-0.85 | Fast | Requires generation control |

*Vendor-reported or estimated from independent evaluations; conditions vary.

---

## 9. Strengths of Current Methods

### 9.1 Zero-Shot Methods
- **No training data needed** — immediately applicable to new models
- **Theoretically grounded** — based on statistical properties of language models
- **Generalizable** — not tied to specific source models (especially Binoculars)

### 9.2 Trained Classifiers
- **High in-distribution accuracy** — can capture subtle patterns
- **Fast inference** — single forward pass
- **Learnable robustness** — adversarial training (RADAR) improves resilience

### 9.3 Watermarking
- **Provable guarantees** — statistical false positive bounds
- **Minimal quality impact** — well-designed watermarks barely affect output
- **Scalable** — detection is efficient

### 9.4 Vendor Systems
- **Operational maturity** — production-tested with guardrails
- **Continuous updates** — adapt to new models
- **Workflow integration** — embedded in existing academic/publishing pipelines

---

## 10. Weaknesses and Failure Modes

### 10.1 Fundamental Theoretical Limits

**Sadasivan et al. (2023)** proved that as language models improve and their output distribution approaches the human text distribution, **all post-hoc detectors must either**:
- Have high false positive rates (flag human text as AI), OR
- Have high false negative rates (miss AI text)

This is because detection is essentially a **distribution distinguishing problem**, and if $P_{\text{AI}} \approx P_{\text{human}}$, no test can reliably separate them without access to the generation process.

### 10.2 Paraphrasing Vulnerability

The most critical practical weakness. A simple paraphrasing pass (even light editing) can:
- Destroy statistical signatures (perplexity, burstiness patterns)
- Break watermarks (replace green-list tokens)
- Fool trained classifiers (shift distribution)

**DIPPER** (Krishna et al., 2024) demonstrated that discourse-level paraphrasing reduces most detectors to near-random performance.

### 10.3 Bias Against Non-Native English Writers

**Liang et al. (2023)** showed that AI detectors systematically flag non-native English writing as AI-generated. This is because non-native writers tend to use simpler vocabulary and more formulaic structures — the same patterns AI detectors associate with machine text.

**Impact**: In educational settings, this creates discriminatory outcomes.

### 10.4 Short Text Unreliability

All statistical methods require sufficient text for signal accumulation. Below ~200-300 words, detection becomes unreliable. This is a structural limitation — there simply isn't enough information.

### 10.5 Model Specificity and Drift

- Detectors trained on GPT-3.5 output may fail on GPT-4 or Claude
- New models render old training data obsolete
- Continuous retraining is expensive and always lags behind new releases

### 10.6 Mixed Human-AI Text

The hardest practical scenario: a document where some paragraphs are human-written and others are AI-generated (or AI-assisted editing). Current approaches:
- Turnitin: sentence-level scoring (partial solution)
- Most research: document-level only (fails here)

### 10.7 Domain Generalization

Detectors trained on one domain (e.g., news articles) often fail in others (e.g., scientific papers, legal documents, creative writing). Domain-specific vocabulary and style confound statistical signals.

---

## 11. Gaps in the Field

### 11.1 Multilingual Detection
Most research focuses on English. Detectors for Chinese, Arabic, Hindi, etc., are far less mature. Multilingual settings are particularly challenging because:
- Perplexity baselines differ by language
- Training data is scarce
- Cross-lingual transfer is unreliable

### 11.2 Code and Structured Text
Detection of AI-generated code, mathematical proofs, and structured formats (tables, JSON) is underexplored. These domains have different statistical properties than prose.

### 11.3 AI-Assisted vs. AI-Generated
No reliable method distinguishes between:
- Fully AI-generated text
- Human text edited by AI (grammar correction, style improvement)
- Human text with AI-suggested sentences

This distinction matters enormously for policy (is using Grammarly "AI-generated"?).

### 11.4 Long-Term Robustness
No study has evaluated detector performance over multi-year timescales as models evolve. The assumption is that detection will get harder, but the trajectory is unknown.

### 11.5 Standardized Evaluation
Despite RAID and M4, there is no universally accepted benchmark with:
- Standard metrics
- Standard attack protocols
- Regular updates for new models
- Fair inclusion of commercial systems

### 11.6 Human-AI Collaboration Detection
The emerging paradigm is not "AI writes, human submits" but "human and AI co-write." Detection for this collaborative scenario is essentially absent.

---

## 12. Methods to Empower Detection and Reduce Weaknesses

### 12.1 Watermarking Deployment (Highest Impact)

**Problem solved**: Eliminates the theoretical impossibility barrier.
**How**: If major model providers (OpenAI, Google, Anthropic, Meta) deploy watermarking:
- Detection becomes provably reliable with known false positive bounds
- Paraphrasing resistance can be engineered in (robust watermarks)
- No training data or model-specific detectors needed

**Barriers**: Requires industry coordination or regulation. Open-source models cannot be forced to watermark. Quality impact concerns (though empirically minimal).

**Recommendation**: Push for SynthID-Text-style watermarking as an industry standard, possibly through regulation (EU AI Act mentions watermarking).

### 12.2 Ensemble Multi-Signal Detection

**Problem solved**: Single-signal fragility.
**How**: Combine orthogonal detection signals:
1. Statistical (perplexity, burstiness)
2. Neural (trained classifier)
3. Topological (PHD/intrinsic dimension)
4. Stylometric (writing style features)
5. Watermark (when present)

**Key insight**: PHD and stylometric features are more robust to paraphrasing than statistical features, so an ensemble that weights these more heavily under suspected attack could maintain performance.

### 12.3 Adversarial Training at Scale

**Problem solved**: Paraphrasing and evasion attacks.
**How**: Extend RADAR's approach with:
- Multiple paraphrasers (DIPPER, back-translation, ChatGPT rewriting)
- Multi-round adversarial training
- Domain-diverse training data

**Enhancement**: Use the latest LLMs as paraphrasers in the adversarial loop to stay current.

### 12.4 Retrieval-Augmented Detection

**Problem solved**: Model drift and new model detection.
**How**: Maintain a database of known AI-generated text (from API honeypots, benchmark corpora). At detection time, search for near-duplicate or semantically similar text.

**Analogy**: This is the "plagiarism detection" approach applied to AI text. Turnitin already does this for plagiarism; extending to AI text is natural.

**Limitation**: Only works for text generated by monitored APIs, not open-source models.

### 12.5 Provenance and Metadata Approaches

**Problem solved**: Authentication rather than detection.
**How**: Instead of detecting AI text, prove that text is human-written:
- Keystroke dynamics recording
- Edit history (track changes)
- Cryptographic signing of drafts at timestamps
- Integration with writing tools (Google Docs, Word) to capture process

**C2PA standard**: The Coalition for Content Provenance and Authenticity is developing standards for content provenance that could be extended to text.

### 12.6 Fine-Grained, Sentence-Level Detection

**Problem solved**: Mixed human-AI documents.
**How**: Move beyond document-level classification to token or sentence-level attribution. Approaches:
- Sliding window classifiers
- Sequence labeling (tag each sentence as human/AI)
- Change-point detection (find where writing style shifts)

**Current state**: Turnitin does sentence-level scoring, but academic research lags. This is a high-value research direction.

### 12.7 Calibrated Uncertainty Estimation

**Problem solved**: False positive harm.
**How**: Instead of binary "AI/human" labels, output calibrated probabilities with confidence intervals. Critical for:
- Educational settings (avoid wrongful accusations)
- Publishing (editorial decisions)
- Legal contexts (evidence admissibility)

**Current gap**: Most detectors output uncalibrated scores. Proper Bayesian or conformal prediction approaches are needed.

### 12.8 Multilingual and Cross-Lingual Detection

**Problem solved**: English-only limitation.
**How**:
- Train multilingual detectors on parallel corpora
- Use multilingual encoder models (mBERT, XLM-R) as bases
- Leverage language-specific perplexity baselines
- Create multilingual benchmarks (M4 is a start)

### 12.9 Continuous Learning Pipeline

**Problem solved**: Model obsolescence.
**How**: Automated pipeline that:
1. Monitors new model releases
2. Generates labeled data from new models
3. Retrains/fine-tunes detectors
4. Evaluates on standard benchmark
5. Deploys updated detector

**Key requirement**: Must be automated, as manual retraining cannot keep pace with model releases.

---

## 13. Conclusion

### The Detection Landscape in 2025

The field of AI text detection is at a crossroads:

1. **Post-hoc detection is fundamentally limited** but practically useful as one signal among many. No single method should be treated as definitive proof.

2. **Watermarking is the most promising technical solution** but requires ecosystem cooperation that doesn't yet exist.

3. **Commercial vendors (Turnitin, GPTZero) provide operationally useful tools** but lack transparency about performance under adversarial conditions.

4. **The biggest practical risks are**:
   - False positives against non-native speakers
   - Over-reliance on single-tool scores
   - Rapidly evolving models outpacing detectors

### Recommended Research Priorities

| Priority | Direction | Expected Impact |
|----------|-----------|-----------------|
| 1 | Industry watermarking standards | Transformative |
| 2 | Robust ensemble methods (PHD + statistical + neural) | High |
| 3 | Sentence-level mixed-text detection | High |
| 4 | Calibrated uncertainty in detector output | High |
| 5 | Multilingual benchmarks and detectors | Medium-High |
| 6 | Provenance-based authentication | Medium (long-term) |
| 7 | Continuous adversarial retraining pipelines | Medium |

### Final Assessment

Detection is not a solved problem and likely never will be for arbitrary text. The most realistic path forward combines: (a) watermarking where possible, (b) calibrated ensemble detection as a supporting signal, (c) provenance systems for high-stakes contexts, and (d) policy frameworks that treat detection scores as probabilistic evidence rather than binary truth.

---

## 14. Full Bibliography

### Surveys
1. Wu, J., et al. (2023). "A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions." arXiv:2310.14724.
2. Weber-Wulff, D., et al. (2023). "Detecting AI-Generated Text: Factors Influencing Detectability with Current Methods." arXiv:2306.05871.
3. Crothers, E., et al. (2023). "Machine-Generated Text: A Comprehensive Survey of Threat Models and Detection Methods." arXiv:2210.07321.
4. Tang, R., et al. (2024). "A Survey on Detection of LLMs-Generated Content." arXiv:2310.15654.

### Zero-Shot Detection
5. Mitchell, E., et al. (2023). "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature." ICML. arXiv:2301.11305.
6. Bao, G., et al. (2023). "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature." arXiv:2310.05130.
7. Gehrmann, S., et al. (2019). "GLTR: Statistical Detection and Visualization of Generated Text." ACL. arXiv:1906.04043.
8. Yang, X., et al. (2023). "DNA-GPT: Divergent N-Gram Analysis with LLM Probability for Machine-Generated Text Detection." arXiv:2305.17359.
9. Hans, A., et al. (2024). "Binoculars: Zero-Shot Detection of LLM-Generated Text." arXiv:2401.12070.
10. Su, J., et al. (2023). "DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text." arXiv:2306.05540.

### Trained Classifiers
11. Verma, V., et al. (2024). "Ghostbuster: Detecting Text Ghostwritten by Large Language Models." arXiv:2305.15047.
12. Hu, X., et al. (2023). "RADAR: Robust AI-Text Detection via Adversarial Learning." arXiv:2307.03838.
13. Liang, W., et al. (2023). "GPT detectors are biased against non-native English writers." Patterns. arXiv:2304.02819.

### Topological Methods
14. Tulchinskii, E., et al. (2024). "Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts." NeurIPS. arXiv:2306.04723.

### Watermarking
15. Kirchenbauer, J., et al. (2023). "A Watermark for Large Language Models." ICML. arXiv:2301.10226.
16. Dathathri, S., et al. (2024). "SynthID-Text: Scalable Watermarking for Identifying Large Language Model Outputs." Nature.
17. Kuditipudi, R., et al. (2023). "Robust Distortion-Free Watermarks for Language Models." arXiv:2307.15593.
18. Christ, M., et al. (2024). "Undetectable Watermarks for Language Models." arXiv:2306.09194.

### Benchmarks
19. Dugan, L., et al. (2024). "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors." arXiv:2405.07940.
20. Weber-Wulff, D., et al. (2023). "Testing of Detection Tools for AI-Generated Text." arXiv:2306.15666.
21. Wang, Y., et al. (2024). "M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection." EACL. arXiv:2305.14902.

### Adversarial & Theoretical
22. Sadasivan, V.S., et al. (2023). "Can AI-Generated Text be Reliably Detected?" arXiv:2303.11156.
23. Krishna, K., et al. (2024). "Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense." NeurIPS. arXiv:2303.13408.
24. Shi, Z., et al. (2024). "How Reliable Are AI-Generated-Text Detectors? An Assessment Framework Using Evasive Soft Prompts." arXiv:2310.05095.

### Vendor Publications
25. Turnitin. "AI Writing Detection Model Architecture and Testing Protocol." Turnitin Whitepaper.
26. Tian, E., et al. (2023). "GPTZero: Towards Detection of AI-Generated Text." (GPTZero technical publication).
27. Chaka, C. (2023). "Detecting AI content in responses generated by ChatGPT, YouChat, and Chatsonic: The case of five AI content detection tools." Journal of Applied Learning and Teaching.