# Dataset Documentation

> Cross-references: [Architecture](architecture.md) · [User Guide § Training Your Own Model](user_guide.md#training-your-own-model) · [Index](index.md)

This document describes the public datasets used to train and evaluate the CIPHER AI text detection system.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Sources](#dataset-sources)
   - [HC3 — Human ChatGPT Comparison Corpus](#hc3--human-chatgpt-comparison-corpus)
   - [RAID — Robust AI Detection Benchmark](#raid--robust-ai-detection-benchmark)
   - [MAGE — Machine-generated text Evaluation](#mage--machine-generated-text-evaluation)
   - [ChatGPT Research Abstracts](#chatgpt-research-abstracts)
   - [andythetechnerd03/AI-human-text](#andythetechnerd03ai-human-text)
   - [GPT-wiki-intro](#gpt-wiki-intro)
3. [Dataset Statistics](#dataset-statistics)
4. [Preprocessing](#preprocessing)
5. [Splits](#splits)
6. [Downloading the Data](#downloading-the-data)
7. [Bias and Limitations](#bias-and-limitations)
8. [Citations](#citations)

---

## Overview

CIPHER is trained on a union of six publicly available AI text detection datasets, totalling approximately **101 800 documents** after deduplication and cleaning.

| Source | AI Docs | Human Docs | Total | Language |
|---|---|---|---|---|
| HC3 | 33 602 | 33 426 | 67 028 | English (+ Chinese subset unused) |
| RAID | ~12 000 | ~12 000 | ~24 000 | English |
| MAGE | ~2 500 | ~2 500 | ~5 000 | English |
| ChatGPT Research Abstracts | 500 | 500 | 1 000 | English |
| andythetechnerd03 | ~2 000 | ~2 000 | ~4 000 | English |
| GPT-wiki-intro | ~500 | ~500 | ~1 000 | English |
| **Total** | **~50 000** | **~51 000** | **~101 800** | English |

All datasets are released under open licences for research use. See [Citations](#citations) for attribution.

---

## Dataset Sources

### HC3 — Human ChatGPT Comparison Corpus

**HuggingFace:** `Hello-SimpleAI/HC3`  
**Direct files:** `https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{subset}.jsonl`

HC3 was created by pairing questions from StackExchange, Reddit (ELI5), Wikipedia, and medical QA datasets with both crowd-sourced human answers and ChatGPT (GPT-3.5-turbo) answers.

**Subsets used:** `reddit_eli5`, `open_qa`, `wiki_csai`, `medicine`, `finance`  
**Format:** JSONL. Each line contains `{"question": ..., "human_answers": [...], "chatgpt_answers": [...]}`.

**Preprocessing applied:**
- Each human answer and each ChatGPT answer is extracted as a separate document.
- Very short answers (< 50 characters) are discarded.
- Human answers are labelled `0`, ChatGPT answers labelled `1`.

**Characteristics:**
- Domain diversity: general knowledge, medicine, finance, Wikipedia.
- ChatGPT answers are consistently formal, complete, and uniformly structured.
- Human answers show spelling errors, personal anecdotes, and hedging language.

**Licence:** Apache 2.0

---

### RAID — Robust AI Detection Benchmark

**HuggingFace:** `lizhou520/raid` (or the official RAID repository)  
**Load via:** `load_dataset("lizhou520/raid", split="train", streaming=True)`

RAID contains adversarially generated documents specifically designed to evade AI detectors, along with the original human-written source texts. It covers multiple domains (news, Wikipedia, Reddit, reviews).

**Characteristics:**
- Both "vanilla" AI generations and adversarially rephrased versions.
- The adversarial split is significantly harder to detect.
- CIPHER training uses the non-adversarial split.

**Licence:** CC BY 4.0

---

### MAGE — Machine-generated text Evaluation

**HuggingFace:** `yaful/MAGE`  
**Load via:** `load_dataset("yaful/MAGE", split="train", streaming=True)`

MAGE covers machine-generated text from 27 different language models across 7 domains, making it the broadest multi-generator benchmark.

**Generators include:** GPT-3.5-turbo, GPT-4, LLaMA-2, Mistral, Falcon, Phi-2, Claude, Cohere, Gemini Pro, and others.

**Characteristics:**
- Ground-truth labels for each generator.
- CIPHER uses only the binary `human` / `machine` label.
- Includes text from many smaller open-source models, not just commercial ones.

**Licence:** MIT

---

### ChatGPT Research Abstracts

**HuggingFace:** `NicolasJM/ChatGPT_Research_Abstracts`  
**Load via:** `load_dataset("NicolasJM/ChatGPT_Research_Abstracts", split="train", streaming=True)`

500 real arXiv research paper abstracts paired with ChatGPT-generated rewrites of the same abstracts.

**Characteristics:**
- Scientific writing style (high information density, formal register).
- An important domain for academic integrity use cases.
- Small dataset — primarily used for domain coverage, not volume.

**Licence:** CC BY 4.0

---

### andythetechnerd03/AI-human-text

**HuggingFace:** `andythetechnerd03/AI-human-text`  
**Load via:** `load_dataset("andythetechnerd03/AI-human-text", split="train", streaming=True)`

A mixed dataset of AI-generated and human-written texts across blog posts, news articles, and creative writing.

**Characteristics:**
- Diverse writing styles.
- Some noise in labels (a small fraction may be mislabelled).

**Licence:** CC BY 4.0

---

### GPT-wiki-intro

**HuggingFace:** `aadityaubhat/GPT-wiki-intro`  
**Load via:** `load_dataset("aadityaubhat/GPT-wiki-intro", split="train", streaming=True)`

Wikipedia article introductions paired with GPT-3.5-generated rewrites.

**Characteristics:**
- Encyclopaedic style; very formal register.
- Human texts are Wikipedia prose; AI texts are paraphrases generated from the same article.
- Similar to HC3 `wiki_csai` but focuses specifically on introductory passages.

**Licence:** CC BY-SA 3.0 (Wikipedia content)

---

## Dataset Statistics

### Class Balance

After combining all six sources:

| Set | AI Documents | Human Documents | Ratio (AI:Human) |
|---|---|---|---|
| Full combined | ~50 200 | ~51 600 | 49% : 51% |
| Train (70%) | ~35 100 | ~36 100 | balanced |
| Val (15%) | ~7 500 | ~7 700 | balanced |
| Test (15%) | ~7 500 | ~7 700 | balanced |

The small class imbalance (< 2%) is handled by `class_weight="balanced"` in the meta-learner.

### Text Length Distribution

| Percentile | Words (AI) | Words (Human) |
|---|---|---|
| 10th | 42 | 38 |
| 25th | 88 | 75 |
| 50th (median) | 175 | 164 |
| 75th | 320 | 310 |
| 90th | 580 | 620 |

Human texts have slightly heavier tails (longer documents) due to Reddit/StackExchange answers.

### Domain Distribution

| Domain | Fraction of Total |
|---|---|
| QA / Reddit | 52% (HC3 reddit_eli5, open_qa) |
| Wikipedia | 16% (HC3 wiki_csai, GPT-wiki-intro) |
| Medical / Finance | 12% (HC3 medicine, finance) |
| Multi-generator benchmark | 10% (MAGE) |
| Adversarial / News | 7% (RAID) |
| Scientific abstracts | 3% (ChatGPT Research Abstracts) |

---

## Preprocessing

The following steps are applied by `download_datasets.py` before writing the combined CSV:

1. **Column normalisation** — each dataset has different column names. All are mapped to `{text: str, label: int}`.
2. **Language filter** — non-English documents are discarded. HC3 includes a Chinese subset which is excluded.
3. **Minimum length filter** — documents with fewer than 50 characters are discarded.
4. **Deduplication** — exact-match deduplication on `text`. Approximately 0.3% of documents are removed.
5. **Label encoding** — `1` = AI-generated, `0` = Human-written, consistently across all sources.
6. **Shuffling** — documents are randomly shuffled before splitting.

**No other normalisation is applied** (no lowercasing, no punctuation removal). The feature extraction pipeline operates on raw text.

---

## Splits

Splits are created by `download_datasets.py` using a stratified 70/15/15 split:

| File | Documents | Description |
|---|---|---|
| `data/full/train.csv` | ~71 000 | Training set for meta-learner and calibration. |
| `data/full/val.csv` | ~15 300 | Validation set for hyperparameter selection. |
| `data/full/test.csv` | ~15 300 | Held-out test set; used only for final evaluation. |
| `data/full_combined.csv` | ~101 800 | All documents combined (before splitting). |

**Stratification:** Splits are stratified by `label` to maintain class balance.

**Sampled variants** (for fast iteration):

| File | Documents | Description |
|---|---|---|
| `data/sampled/train.csv` | 16 000 | 8 000 per class, sampled from `data/full/train.csv` |
| `data/sampled/val.csv` | 2 000 | 1 000 per class |
| `data/sampled/test.csv` | 2 000 | 1 000 per class |

---

## Downloading the Data

```bash
# Full download (~100k rows, takes ~30 minutes depending on connection)
python download_datasets.py

# With custom limits
python download_datasets.py --max-per-source 5000 --output-dir data/small/

# Create sampled subsets after download
python sample_dataset.py --n-each 8000 --output-dir data/sampled/
```

### `download_datasets.py` CLI

| Argument | Default | Description |
|---|---|---|
| `--max-per-source` | None (unlimited) | Maximum documents per source. |
| `--output-dir` | `data/full/` | Directory for output CSV files. |
| `--train-frac` | `0.70` | Fraction of data for training. |
| `--val-frac` | `0.15` | Fraction of data for validation. |
| (test fraction is `1 - train_frac - val_frac`) | | |

### `sample_dataset.py` CLI

| Argument | Default | Description |
|---|---|---|
| `--n-each` | `8000` | Balanced documents per class (train). |
| `--test-each` | `1000` | Balanced documents per class (test/val). |
| `--output-dir` | `data/sampled/` | Output directory. |
| `--seed` | `42` | Random seed. |

---

## Bias and Limitations

1. **English only.** All datasets are in English. CIPHER is not evaluated or validated on other languages.

2. **Generator-specific patterns.** The training data is dominated by GPT-3.5-turbo outputs (HC3 is ~66% of the total). Texts from newer or different models (GPT-4, Claude 3, Gemini 1.5) may exhibit different patterns not well-represented in the training distribution.

3. **Domain gaps.** Legal documents, code, spreadsheet data, and highly technical scientific papers are underrepresented. Accuracy in these domains may be lower.

4. **Short texts.** CIPHER requires at least 20 words. The feature pipeline becomes less reliable at 20–50 words as statistical estimates (Hurst, KL divergence) are unstable with small samples.

5. **Adversarial texts.** Texts specifically written or paraphrased to evade AI detection will be harder to classify. The RAID adversarial split is held out of training and represents an upper bound on adversarial difficulty.

6. **Human diversity.** Human texts in the training data are biased toward Reddit/StackExchange writing style (conversational, informal, English-first-language skewed). Academic or non-native-speaker writing may be misclassified more often.

---

## Citations

If you use CIPHER or these datasets in academic work, please cite:

### HC3

```bibtex
@inproceedings{guo2023hc3,
  title     = {How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection},
  author    = {Guo, Biyang and Zhang, Xin and Wang, Ziyuan and Jiang, Minqi and Nie, Jinran and Ding, Yuxuan and Yue, Jianwei and Wu, Yupeng},
  year      = {2023},
  eprint    = {2301.07597},
  archivePrefix = {arXiv},
}
```

### RAID

```bibtex
@inproceedings{dugan2024raid,
  title     = {{RAID}: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors},
  author    = {Dugan, Liam and others},
  booktitle = {Proceedings of ACL 2024},
  year      = {2024},
}
```

### MAGE

```bibtex
@article{li2023mage,
  title   = {MAGE: Machine-Generated Text Detection in the Wild},
  author  = {Li, Yafu and others},
  journal = {arXiv preprint arXiv:2305.13242},
  year    = {2023},
}
```

### Conformal Prediction (methodology)

```bibtex
@book{vovk2005algorithmic,
  title     = {Algorithmic Learning in a Random World},
  author    = {Vovk, Vladimir and Gammerman, Alexander and Shafer, Glenn},
  publisher = {Springer},
  year      = {2005},
}
```
