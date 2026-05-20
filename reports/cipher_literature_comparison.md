# CIPHER vs. Literature: Comparative Analysis

**Date:** May 2026  
**CIPHER test-set results (models_full, 15 272 docs, float64):**  
ROC-AUC 0.9423 · PR-AUC 0.9355 · Accuracy 0.874 · Precision 0.858 · Recall 0.867 · F1 0.863

---

## 1. Executive Summary

CIPHER belongs to the **structural/linguistic feature + classical ML** family of AI-text detectors.
Within that family it is among the strongest reported systems, and it
compares favourably to most published baselines on standard metrics.
It achieves this *without* requiring an LLM at inference time — a key
practical advantage over the dominant zero-shot and neural-classifier
approaches.  Several meaningful gaps remain, detailed below.

---

## 2. Where CIPHER Sits in the Taxonomy

| Family | Example systems | CIPHER position |
|--------|----------------|-----------------|
| Zero-shot statistical (perplexity / curvature) | DetectGPT, Fast-DetectGPT, Binoculars, LAPD | **Not this family** — CIPHER never queries an LLM at inference |
| Zero-shot topological | PHD (persistent homology) | Partial overlap: CIPHER's co-reference graph is a topological signal |
| Fine-tuned neural classifier | RoBERTa, RADAR, MELD, DRGD | **Not this family** — CIPHER has no fine-tuned transformer |
| **Structural / linguistic feature + ML** | DependencyAI, Ghostbuster, Pudasaini et al. | **CIPHER's family** |
| Hybrid / ensemble | GPTZero (production), MELD multi-task | Partial: CIPHER is an ensemble of hand-crafted features + soft-voting |
| Watermarking | KGW, SynthID, GaussMark | Not applicable |

---

## 3. Performance Comparison

### 3.1 Numbers side by side

| System | AUROC | PR-AUC | Accuracy | F1 | Access | Open-source |
|--------|------:|-------:|---------:|---:|--------|:-----------:|
| **CIPHER (full corpus)** | **0.9423** | **0.9355** | **0.874** | **0.863** | Text only | ✓ |
| DetectGPT (Mitchell 2023) | 0.85–0.95 | — | — | — | Source-model logits | ✓ |
| Fast-DetectGPT (Bao 2023) | 0.85–0.95 | — | — | — | Source-model logits | ✓ |
| Binoculars (Hans 2024) | 0.90–0.98 | — | — | — | 2× LLM logits | ✓ |
| LAPD / Alignment Imprint (2026) | +45 % rel. vs Fast-DetectGPT | — | — | — | Base + aligned logits | ✓ |
| PHD / persistent homology (Tulchinskii 2024) | 0.82–0.90 | — | — | — | Embedding model | ✓ |
| DependencyAI (Ahmed 2026) | Competitive† | — | — | — | Dependency parser | ✓ |
| RADAR (Hu 2023) | 0.88–0.95 | — | — | — | None (inference) | ✓ |
| Ghostbuster (Verma 2024) | 0.85–0.92 | — | — | — | Multiple LLMs | ✓ |
| MELD (Li 2026) | RAID SOTA | — | — | — | None (inference) | ✓ |
| DRGD (Pu 2026) | +24 % acc vs prior | — | — | — | None (inference) | ✓ |
| Pudasaini et al. 30-feature (2026) | — | — | — | **0.973** (in-domain) | Text only | ✓ |
| Turnitin (est.) | ~0.90–0.95 | — | — | — | None | ✗ |
| GPTZero (est.) | ~0.88–0.94 | — | — | — | None | ✗ |
| KGW watermark | 0.99+ | — | — | — | Generator control | ✓ |

† "Competitive" is the paper's own description; exact AUROC not published on a comparable benchmark.

**Caveat on comparability**: Every system above was evaluated on a *different test set* under *different
conditions*. Direct numeric comparison must be treated as directional guidance, not a controlled benchmark.
CIPHER's 0.9423 is measured on a held-out split of its own training distribution (HC3 + RAID + MAGE +
supplementary English datasets). A meaningful apples-to-apples evaluation would require running CIPHER on
RAID or MELD-eval — which has not yet been done.

### 3.2 Interpretation

CIPHER's ROC-AUC of **0.9423** is:

- **Better than or comparable to** DetectGPT, Fast-DetectGPT, PHD, RADAR, Ghostbuster, GPTZero (estimated) —
  while requiring *no LLM at inference*, which is a strict resource advantage.
- **Lower than Binoculars (upper bound ~0.98)** and MELD-RAID-SOTA — but both of these require either two
  large LLMs (Binoculars) or a fine-tuned transformer trained on diverse adversarial data (MELD).
- **Lower than Pudasaini et al. (F1 = 0.973 in-domain)** — but their paper then showed via SHAP that most
  of that performance was driven by dataset artifacts rather than stable authorship signals. Out-of-domain
  performance degraded substantially. CIPHER's approach of combining information-theoretic,
  cognitive, and discourse features was specifically designed to reduce this risk.

---

## 4. Similarities with Literature

### 4.1 Structural / Lexical Features (DependencyAI, Pudasaini, Ghostbuster)

CIPHER's Layer 6 lexical features (TTR, hapax ratio, bigram repetition, passive ratio, punctuation
entropy) directly overlap with the stylometric feature families that DependencyAI and Pudasaini et al.
found most discriminative. Specifically:

| CIPHER feature | Literature analog | Source |
|----------------|------------------|--------|
| `s1_ttr` (type-token ratio) | TTR listed as top AI-vs-human discriminator | Pudasaini 2026, Table 1 |
| `s4_bigram_rep` (bigram repetition) | Log-rank / n-gram repetition | DetectLLM, DNA-GPT |
| `s2_hapax` (hapax ratio) | Vocabulary richness / hapax | Gehrmann 2019, Pudasaini 2026 |
| `l1_lzc` (LZ76 complexity) | Compression-based detection | Novel — no direct prior art in published detectors |
| `d6_depth_var`, `d7_depth_skew` | Syntactic depth / dependency arc | DependencyAI (Ahmed 2026) |
| `c5_coh_mean` (sentence coherence) | Semantic consistency profiling | Stylistic Consistency (Li 2026) |

The **permutation importance** ranking from `full_model_eval.ipynb` confirms that the top features are
stylometric / compression features (`s1_ttr`, `s4_bigram_rep`, `s2_hapax`, `l1_lzc`, `l4_redundancy`),
consistent with the literature's finding that these signals are among the most stable.

### 4.2 Ensemble Strategy (Ghostbuster, GPTZero)

Like Ghostbuster, CIPHER uses a **structured search over hand-crafted signals followed by a linear/shallow
meta-learner**. The VotingClassifier (`2 × HistGBM + 1 × LogisticRegression`, soft probabilities) is
architecturally similar to Ghostbuster's feature-engineered + linear classifier pipeline.
GPTZero's production system (as disclosed in arXiv:2602.13042) also uses a hierarchical multi-task
architecture with statistical signals feeding a neural layer — analogous to CIPHER's feature layers
feeding the meta-learner.

### 4.3 Information-Theoretic Compression Features (Novel but theoretically motivated)

The CIPHER plan cites LZ76 complexity as the core Layer 1 signal. This is **theoretically justified**
by Shannon's source coding theorem: AI text, produced by high-probability sampling from a learned
distribution, should have lower empirical entropy and therefore lower LZ76 complexity at equivalent
length. No other published detector uses LZ76 as a primary signal — making this a genuine differentiation
point with theoretical grounding.

### 4.4 Black-Box Operation

CIPHER, DependencyAI, Pudasaini et al., and MELD all operate **without needing access to the generating
model's weights or logits** — the most practically important constraint for real-world deployment, since
the generating model is usually unknown. The zero-shot methods (DetectGPT, Binoculars, LAPD) all fail
this constraint.

---

## 5. What CIPHER Does Differently

### 5.1 No LLM at Inference (Layers 1–3)

This is the most important architectural distinction. Zero-shot SOTA systems (Binoculars, LAPD) require
inference through two LLMs per document. At commercial scale (millions of documents/day), this cost is
prohibitive. CIPHER's Layers 1–3 run entirely on CPU with lightweight NLP tools (spaCy, zstd).
Only Layer 4 (contrastive manifold — currently optional / not in the deployed 32-feature model) requires
a neural encoder.

**Performance cost of this choice**: comparing CIPHER's 0.9423 to Binoculars's ~0.98 suggests an
approximately 4–5 point AUROC penalty for the LLM-free approach. This is the price of practical
deployment at scale.

### 5.2 Theoretical Motivation for Each Feature

Rather than a generic neural network end-to-end, CIPHER explicitly encodes four independent theoretical
claims:
1. AI text is more compressible (Layer 1 — Shannon / LZ76).
2. AI text has lower cognitive variance (Layer 2 — Hurst exponent, epistemic markers).
3. AI text has smoother discourse topology (Layer 3 — co-reference graph, topic JS divergence).
4. AI text is lexically less diverse (Layer 6 — TTR, hapax, bigram repetition).

Each of these claims has independent theoretical and empirical support in the literature (Sections 3–5
of `report_comprehensive.md`). This is more principled than most neural classifiers, which learn
undifferentiated patterns.

### 5.3 Multi-Source Training Data

CIPHER trains on **~101,800 documents** from six sources (HC3, RAID, MAGE, and three supplementary
datasets). The RAID component specifically includes adversarially rewritten text. This multi-source
strategy is directly motivated by the DRGD finding that "performance improves as generator diversity in
training increases" and the Pudasaini finding that single-source training exploits dataset artifacts.

---

## 6. What CIPHER Does Worse (or Has Not Yet Addressed)

### 6.1 Not Evaluated on a Shared Benchmark

CIPHER has not been evaluated on RAID (Dugan 2024) or MELD-eval (Li 2026). Every number in Sections 3
and 4 is from CIPHER's own train/test split. Until CIPHER is benchmarked on RAID or MELD-eval, the
0.9423 figure cannot be directly compared to published numbers.

**Risk**: RAID's adversarial attack condition reduces most detectors from >0.95 to 0.55–0.75 AUROC.
CIPHER's structural features (compression, discourse topology) should be more robust to paraphrase
attacks than perplexity-based methods (theoretical claim), but this has **not been tested**.

### 6.2 Paraphrase Robustness — Claimed but Untested

The implementation plan claims structural features are "intrinsically robust to paraphrasing." The
literature provides partial support:
- PHD (topological signals) maintains AUROC 0.70–0.82 under paraphrasing vs. 0.82–0.90 clean.
- DependencyAI achieves "partial" paraphrase robustness per its benchmark.
- Pudasaini et al. showed that stylometric features — exactly the ones CIPHER most relies on —
  suffer significant domain/distribution shift.

**The Sadasivan impossibility result applies**: as paraphrasing pushes AI text distributions closer to
human text, all post-hoc detectors must degrade. CIPHER is not immune.

### 6.3 No Adversarial Training

RADAR (Hu 2023) explicitly trains in a detection-evasion loop, giving substantially better paraphrase
robustness (AUROC 0.70–0.85 under paraphrase vs. CIPHER's unknown). MELD uses an EMA teacher/student
setup with augmented inputs. CIPHER has no adversarial training component in its current implementation.

### 6.4 Neural Classifier Ceiling Not Reached

MELD (2026) achieves **99.9% TPR at 1% FPR** on MELD-eval (unseen models) — roughly equivalent to
AUROC > 0.999 under ideal conditions — while being open-source. CIPHER's 0.9423 is significantly below
this ceiling. The gap arises because:
- MELD uses a fine-tuned multi-task transformer encoder (100 M+ parameters) vs. CIPHER's 32 engineered
  features + shallow classifiers.
- MELD's multi-task supervision (generator family, attack type, domain) provides richer learning signal.
- MELD was explicitly trained with adversarial augmentation.

**CIPHER's counter-argument**: MELD's 99.9% TPR was measured on MELD-eval, a benchmark from the same
lab. Independent replication at this level has not been published. History shows that high in-distribution
performance frequently degrades on truly novel test sets (Pudasaini 2026).

### 6.5 English-Only

BLUFF (Lucas 2026) showed that most English-trained detectors perform near-randomly on 58 low-resource
languages. CIPHER is currently English-only (spaCy en_core_web_sm, English epistemic marker lexicon,
English reference corpora for NCD). Cross-lingual generalisation requires language-specific feature
adaptation and is not implemented.

### 6.6 Short Text Handling

All structural methods, including CIPHER, degrade on short texts. The implementation has minimum-length
fallback logic in the discourse and lexical layers, but the exact performance floor has not been
characterised. The literature consensus is that < 150–200 tokens produces unreliable scores.

### 6.7 No Sentence-Level Attribution in Current Model

The architecture plan specifies a Layer 6 sentence attribution output. In the deployed 32-feature model,
this layer is not active — documents receive a single score. Turnitin's sentence-level scoring and
GPTZero's per-sentence highlighting are key commercial differentiators that CIPHER currently lacks.

### 6.8 Conformal Calibration Not Deployed

The CIPHER plan specifies isotonic calibration + conformal prediction sets as Layer 5. This is not
reflected in the current VotingClassifier + StandardScaler implementation. Calibration means the reported
0.9423 AUROC is uncalibrated — the raw probabilities from the classifier have no validity guarantee.
For deployment in high-stakes settings (academic integrity), calibrated uncertainty quantification is
essential (cf. GPTZero's explicit focus on this, CIPHER plan Section 5.5).

---

## 7. Feature-Level Comparison vs. Literature Findings

| CIPHER Feature Group | Literature Finding | Consistent? |
|---------------------|-------------------|:-----------:|
| TTR, hapax, bigram rep (top permutation importance) | Identified as top discriminators in Pudasaini 2026, Gehrmann 2019 | ✅ Yes |
| LZ76 compression (l1_lzc) | Theoretically motivated; novel as a primary signal | ✅ Theoretically sound, no direct comparison |
| Coherence mean (c5_coh_mean) | Semantic consistency profiling (Li 2026): AI is "too coherent" | ✅ Yes |
| Co-reference graph (d1–d3) | Discourse topology — novel; no published comparison | ✅ Theoretically motivated |
| Topic JS variance (d4–d5) | No direct analog in literature | Untested |
| Passive ratio (s5, spaCy-dependent) | Not highlighted in literature as a top discriminator | ⚠️ Uncertain |
| Syntactic depth variance (d6, d7) | DependencyAI: dependency arc patterns are discriminative | ✅ Consistent |
| Epistemic markers (c3, c6) | Plan predicts humans use more hedging; literature supports this directionally | ✅ Directionally consistent |
| Hurst exponent (c2) | Vocabulary burstiness / Hurst — mentioned in GPTZero (burstiness signal) | ✅ Consistent |

---

## 8. Structural Comparison to Closest Analogues

### 8.1 CIPHER vs. DependencyAI (Ahmed & Hammond, 2026)

DependencyAI is the closest published analogue: both use **structural syntactic features, no neural
encoder, CPU-only, interpretable, black-box inference**. Key differences:

| Dimension | CIPHER | DependencyAI |
|-----------|--------|-------------|
| Feature scope | 32 features across 4 theoretical layers | Dependency arc label distributions only |
| Feature count | 32 | ~25 dependency relation types |
| Compression features | ✅ LZ76, NCD, redundancy | ✗ None |
| Discourse features | ✅ Co-reference graph, topic variance | ✗ None |
| Meta-learner | VotingClassifier (GBM + LR) | Not specified (traditional ML) |
| Multilingual | ✗ English only | ✅ Multilingual (dependency parsers) |
| Adversarial robustness | Untested | Partial (per paper) |
| Benchmark | Own test split, AUC 0.9423 | RAID/M4, described as "competitive" |

CIPHER's broader feature design should give it a higher ceiling, at the cost of multilingual support
and marginal additional dependency on spaCy discourse features.

### 8.2 CIPHER vs. Ghostbuster (Verma et al., 2024)

Both use structured feature engineering + shallow classifier. Key difference: Ghostbuster queries
**multiple LLMs** to compute log-probability features. CIPHER uses **no LLMs**. For equivalent
conditions:

| Dimension | CIPHER | Ghostbuster |
|-----------|--------|-------------|
| LLM at inference | ✗ None | ✅ 2–3 LLMs needed |
| AUROC range | 0.9423 | 0.85–0.92 |
| Feature type | Structural / information-theoretic | Per-token LLM log-probs |
| Cross-model generalization | Structural features are model-agnostic | LLM features may encode generator identity |
| Inference cost | Low (CPU) | High (LLM API) |

CIPHER outperforms Ghostbuster on its own test set and does so at much lower inference cost.
Ghostbuster's design ties detection signal to specific LLM outputs, which becomes stale as models evolve.

### 8.3 CIPHER vs. Pudasaini et al. 30-Feature Classifier (2026)

Pudasaini et al. achieved F1 = 0.973 on PAN CLEF 2025 / COLING 2025 with 30 linguistic features, then
showed via SHAP that the classifier was exploiting dataset-specific artifacts. CIPHER's 0.863 F1 is
lower in-domain, but:

1. CIPHER trains on 6 heterogeneous datasets vs. Pudasaini's presumably narrower training set.
2. CIPHER's features were designed to be theoretically grounded and cross-domain transferable.
3. The Pudasaini paper's main finding is a **cautionary tale about high in-domain F1** — it does not
   imply CIPHER should chase that number.

The analogy is strong: both use 30-feature structural classifiers. CIPHER can be seen as a more robust
instantiation of the same idea, trained on a broader distribution.

---

## 9. Summary Scorecard

| Criterion | CIPHER | Grade |
|-----------|--------|:-----:|
| Performance on own test set (AUROC) | 0.9423 — competitive with most published systems | B+ |
| Performance on shared benchmark (RAID) | Not evaluated | — |
| Paraphrase robustness | Theoretically argued, empirically untested | C |
| Inference cost | Low — no LLM needed | A |
| Interpretability | Feature attributions available | B |
| Calibration | Uncalibrated (conformal layer not deployed) | C |
| Adversarial robustness | Not explicitly trained; untested | C |
| Multilingual support | English only | D |
| Short-text handling | Fallback logic, floor uncharacterised | C |
| Sentence-level attribution | Not yet deployed | C |
| Theoretical novelty | LZ76 + co-reference topology — novel combination | A |
| Open-source & reproducible | Fully open, public datasets | A |

---

## 10. Recommended Next Steps (based on gaps above)

1. **Evaluate on RAID** — run CIPHER on the public RAID test set to get a benchmark-comparable AUROC,
   including the paraphrase attack condition.
2. **Test paraphrase robustness** — apply DIPPER (lexical level 40 and 60) to the test set and
   measure AUROC degradation. Hypothesis: structural features will degrade less than perplexity.
3. **Deploy conformal calibration** — replace raw VotingClassifier probabilities with isotonic
   regression + split-conformal prediction sets, as specified in the implementation plan.
4. **Adversarial augmentation** — add paraphrased versions of training documents (using DIPPER or a
   local LLM) to the training mix, following RADAR's approach.
5. **Add MELD as a comparison baseline** — run MELD weights on the same test.csv to get a direct
   comparison on identical data.
6. **Domain ablation** — evaluate per-source-dataset to identify which data contributes most to
   generalisation vs. which introduces artifacts (follow Pudasaini's SHAP methodology).

---

## 11. References

Full citations are in `report_comprehensive.md §15`. Key papers for this comparison:

- Mitchell et al. (2023). DetectGPT. ICML 2023.
- Bao et al. (2023). Fast-DetectGPT. ICLR 2024.
- Hans et al. (2024). Binoculars. arXiv:2401.12070.
- Hu et al. (2023). RADAR. NeurIPS 2023.
- Verma et al. (2024). Ghostbuster. NAACL 2024.
- Tulchinskii et al. (2024). PHD. NeurIPS 2024.
- Yang et al. (2023). DNA-GPT. ICLR 2024.
- Su et al. (2023). DetectLLM. arXiv:2306.05540.
- Dugan et al. (2024). RAID benchmark. arXiv:2406.07016.
- Wang et al. (2024). M4 benchmark. EACL 2024.
- Li et al. (2026). MELD. arXiv:2605.06903.
- Pu et al. (2026). DRGD. arXiv:2602.15xxx.
- Ahmed & Hammond (2026). DependencyAI. arXiv:2602.15514.
- Pudasaini et al. (2026). Why Detection Fails. arXiv:2602.xxxxx.
- Lucas et al. (2026). BLUFF. arXiv:2603.xxxxx.
- Wu et al. (2026). LAPD / Alignment Imprint. arXiv:2604.16923.
- Adam et al. (2026). GPTZero technical report. arXiv:2602.13042.
- Sadasivan et al. (2023). Impossibility of detection. arXiv:2303.11156.
- Liang et al. (2023). Non-native English bias. arXiv:2304.02819.
- Krishna et al. (2024). DIPPER. EMNLP 2024.
