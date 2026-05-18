"""Layer 2: Cognitive load fingerprint features.

Computes 8 features derived from the measurable properties of human cognition
as expressed in writing. No LLM required; sentence-transformers is optional.

Humans produce cognitively-loaded writing with irregular sentence lengths
(Weibull-distributed, not Gaussian), non-random vocabulary introduction
patterned by working memory (Hurst H > 0.5), variable semantic jumps between
ideas, and hedging/uncertainty language reflecting genuine epistemic states.
AI models trained to optimise fluency tend to produce the opposite.

Features
--------
c1  sentence-length distribution shape (KL Gaussian vs KL Weibull)
c2  Hurst exponent of vocabulary novelty curve
c3  semantic coherence trajectory variance
c4  semantic coherence trajectory lag-2 autocorrelation
c5  semantic coherence mean
c6  epistemic marker density (per 1 000 words)
c7  first-person opinion marker density (per 1 000 words)
c8  self-correction marker density (per 1 000 words)

See Also
--------
cipher.layers.compression : Compression-theoretic features (Layer 1).
cipher.layers.discourse   : Discourse topology features (Layer 3).
cipher.layers.assembler   : Combines all layer outputs into a FeatureVector.
cipher.data               : Epistemic lexicon (epistemic_lexicon.json).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Epistemic lexicon
# ---------------------------------------------------------------------------

_LEXICON_PATH = Path(__file__).parent.parent / "data" / "epistemic_lexicon.json"


@lru_cache(maxsize=1)
def _load_lexicon() -> dict[str, list[str]]:
    """Load the epistemic lexicon JSON once and cache it for the process lifetime.

    The lexicon maps category names (``hedges``, ``epistemic_verbs``,
    ``first_person_opinion``, etc.) to lists of trigger phrases.  It is
    stored in ``cipher/data/epistemic_lexicon.json``.

    Returns:
        dict[str, list[str]]: Mapping from category label to phrase list.

    Raises:
        FileNotFoundError: If the lexicon file cannot be located.
    """
    with _LEXICON_PATH.open() as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _build_patterns() -> dict[str, re.Pattern[str]]:
    """Compile one case-insensitive regex per lexicon category.

    Each regex uses whole-word boundary anchors (``\\b``) to avoid
    false positives on sub-word matches (e.g. ``maybe`` not matching
    inside ``unmaybelike``).

    Returns:
        dict[str, re.Pattern[str]]: Category → compiled regex.
    """
    lex = _load_lexicon()
    patterns: dict[str, re.Pattern[str]] = {}
    for category, phrases in lex.items():
        escaped = [re.escape(p) for p in phrases]
        patterns[category] = re.compile(
            r"\b(" + "|".join(escaped) + r")\b",
            re.IGNORECASE,
        )
    return patterns


@lru_cache(maxsize=1)
def _get_patterns() -> dict[str, re.Pattern[str]]:
    """Return the compiled pattern dict, building it once per process.

    Returns:
        dict[str, re.Pattern[str]]: Category → compiled regex (cached).
    """
    return _build_patterns()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CognitiveFeatures:
    """Immutable container for the 8 cognitive load feature values.

    Attributes:
        c1: KL(empirical ‖ Gaussian) − KL(empirical ‖ Weibull) for
            per-sentence word counts.  Positive → Gaussian fits better
            (uniform-length sentences are an AI signal).
        c2: Hurst exponent H of the binary vocabulary-novelty series.
            H ≈ 0.5 → memoryless (AI); H > 0.65 → persistent bursts (human).
        c3: Variance of consecutive-sentence cosine similarity scores.
            Low variance → unnaturally smooth coherence (AI signal).
        c4: Lag-2 autocorrelation of the coherence similarity series.
        c5: Mean consecutive-sentence cosine similarity.
        c6: Epistemic hedge / uncertainty markers per 1 000 words.
            AI text tends to under-express epistemic uncertainty.
        c7: First-person opinion markers ("I think", "I believe") per
            1 000 words.  Rare in most AI outputs.
        c8: Self-correction / clarification markers ("actually", "I mean")
            per 1 000 words.  Human writers revise their stance; LLMs don't.
        available: Names of features that were successfully computed.
    """
    c1: float  # KL(Gaussian) - KL(Weibull) for sentence lengths
    c2: float  # Hurst exponent of vocabulary novelty
    c3: float  # coherence trajectory variance
    c4: float  # coherence trajectory lag-2 autocorrelation
    c5: float  # coherence trajectory mean
    c6: float  # epistemic density per 1k words
    c7: float  # first-person opinion density per 1k words
    c8: float  # self-correction density per 1k words
    available: list[str] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Return features as a 1-D float64 array in canonical order [c1..c8].

        Returns:
            np.ndarray: Shape (8,), dtype float64.
        """
        return np.array([self.c1, self.c2, self.c3, self.c4, self.c5,
                         self.c6, self.c7, self.c8])


# ---------------------------------------------------------------------------
# Feature: sentence-length distribution shape (c1)
# ---------------------------------------------------------------------------

def sentence_length_kl_diff(sentences: list[str]) -> float:
    """Feature c1: KL-divergence difference for sentence-length distributions.

    Tests whether the observed word-count distribution across sentences
    is better described by a Gaussian (uniform, AI-like) or a Weibull
    (skewed, human-like) distribution.

    Computes:
        c1 = KL(empirical ‖ Gaussian) − KL(empirical ‖ Weibull)

    where the KL divergence is:
        KL(P ‖ Q) = Σ P(x) · log(P(x) / Q(x))

    Both distributions are fit by MLE on the observed sentence lengths.
    Positive c1 → Gaussian fits better → uniform sentence lengths → AI.
    Negative c1 → Weibull fits better → skewed distribution → human.

    Args:
        sentences: List of sentence strings from the document.

    Returns:
        float: c1 value; typically in (−2.0, 2.0).  Falls back to 0.0
        when fewer than 5 sentences are present.
    """
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 5:
        return 0.0

    arr = np.array(lengths, dtype=float)
    if arr.std() < 1e-6:
        return 1.0  # perfectly uniform → strong AI signal

    # Fit distributions via MLE
    mu, sigma = float(arr.mean()), float(arr.std())
    try:
        k, loc, scale = stats.weibull_min.fit(arr, floc=0)
    except Exception:
        return 0.0

    # Discretise over observed range and compute KL divergences
    lo, hi = int(arr.min()), int(arr.max()) + 1
    bins = np.arange(lo, hi + 1, dtype=float)
    observed, _ = np.histogram(arr, bins=np.append(bins, bins[-1] + 1), density=False)
    total = observed.sum()
    if total == 0:
        return 0.0
    empirical = np.clip(observed / total, 1e-10, None)

    mid = bins + 0.5
    gauss_pdf = np.clip(stats.norm.pdf(mid, mu, sigma), 1e-10, None)
    gauss_pdf /= gauss_pdf.sum()
    weib_pdf = np.clip(stats.weibull_min.pdf(mid, k, loc, scale), 1e-10, None)
    weib_pdf /= weib_pdf.sum()

    kl_gauss = float(np.sum(empirical * np.log(empirical / gauss_pdf)))
    kl_weib = float(np.sum(empirical * np.log(empirical / weib_pdf)))
    return kl_gauss - kl_weib


# ---------------------------------------------------------------------------
# Feature: Hurst exponent (c2)
# ---------------------------------------------------------------------------

def hurst_exponent(series: np.ndarray) -> float:
    """Estimate the Hurst exponent via rescaled range (R/S) analysis.

    The R/S statistic for a sub-series of length L is:
        R/S(L) = (max(cumdev) − min(cumdev)) / σ

    where ``cumdev = cumsum(X − mean(X))`` is the cumulative deviation.
    Regressing log(R/S) on log(L) gives the Hurst exponent H:

        E[R/S(L)] ∝ L^H

    Interpretation:
        H ≈ 0.5  → random / memoryless (Poisson) process — typical of
                   AI default-temperature sampling.
        H > 0.65 → persistent long-range correlations — human vocabulary
                   usage follows ``Zipf bursts`` (same words cluster).
        H < 0.4  → anti-persistent — rare; seen in some low-temperature
                   AI text (artificially diverse word choice).

    Uses 20 log-spaced lag values from 4 up to n//2.

    Args:
        series: 1-D numeric array (e.g. binary novelty signal).

    Returns:
        float: Estimated H in [0.0, 1.0]; 0.5 for series shorter than 20.
    """
    n = len(series)
    if n < 20:
        return 0.5

    max_lag = n // 2
    lags = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max_lag), 20)).astype(int))
    lags = lags[lags >= 4]

    log_lags, log_rs = [], []
    for lag in lags:
        n_blocks = n // lag
        if n_blocks < 2:
            continue
        rs_vals = []
        for i in range(n_blocks):
            block = series[i * lag : (i + 1) * lag].astype(float)
            std = block.std(ddof=1)
            if std < 1e-10:
                continue
            cumdev = np.cumsum(block - block.mean())
            rs_vals.append((cumdev.max() - cumdev.min()) / std)
        if rs_vals:
            log_lags.append(np.log(lag))
            log_rs.append(np.log(np.mean(rs_vals)))

    if len(log_lags) < 3:
        return 0.5

    h, _ = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(h, 0.0, 1.0))


def vocabulary_novelty_hurst(text: str) -> float:
    """Feature c2: Hurst exponent of the vocabulary novelty curve.

    Encodes the document as a binary ``novelty series``: each token
    receives 1 if it is appearing for the first time, 0 otherwise.
    The Hurst exponent of this series captures whether new vocabulary
    is introduced in persistent bursts (human: H > 0.65) or randomly
    scattered (AI: H ≈ 0.5).

    Args:
        text: The full document string (whitespace-tokenised).

    Returns:
        float: Hurst exponent in [0.0, 1.0]; 0.5 for texts with fewer
        than 40 tokens.
    """
    tokens = text.lower().split()
    if len(tokens) < 40:
        return 0.5
    seen: set[str] = set()
    novelty = []
    for tok in tokens:
        is_new = 1 if tok not in seen else 0
        seen.add(tok)
        novelty.append(is_new)
    # R/S on the binary novelty series
    return hurst_exponent(np.array(novelty, dtype=float))


# ---------------------------------------------------------------------------
# Feature: semantic coherence trajectory (c3, c4, c5)
# ---------------------------------------------------------------------------

def semantic_coherence_features(
    sentences: list[str],
    encoder: Any | None = None,
) -> tuple[float, float, float]:
    """Features c3, c4, c5: statistics of the inter-sentence coherence trajectory.

    Computes pairwise cosine similarity between consecutive sentences,
    producing a coherence profile {sim(s_i, s_{i+1})}.

    Three statistics are extracted:
        c3 = Var[sim_i]           — low variance → unnaturally smooth (AI)
        c4 = Corr(sim_t, sim_{t+2}) — lag-2 autocorrelation of the profile
        c5 = mean(sim_i)          — mean coherence level

    When *encoder* is provided (sentence-transformers ``SentenceTransformer``),
    dense neural embeddings are used for cosine similarity.  Otherwise, a
    bag-of-words (BoW) cosine fallback is applied — lower quality but no
    extra dependency.

    Args:
        sentences: Pre-tokenised sentence list.
        encoder: Optional sentence-transformers encoder.  When None, BoW
            cosine similarity is used as a fallback.

    Returns:
        tuple[float, float, float]: (c3_variance, c4_lag2_autocorr, c5_mean).
        Falls back to (0.0, 0.0, 0.5) for fewer than 3 sentences.
    """
    clean = [s.strip() for s in sentences if s.strip()]
    if len(clean) < 3:
        return 0.0, 0.0, 0.5

    if encoder is not None:
        embeddings: np.ndarray = encoder.encode(clean, show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings = embeddings / norms
        sims = np.array([
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ])
    else:
        # Fallback: bag-of-words cosine similarity
        vocab: dict[str, int] = {}
        for s in clean:
            for w in s.lower().split():
                if w not in vocab:
                    vocab[w] = len(vocab)
        vecs = np.zeros((len(clean), len(vocab)))
        for i, s in enumerate(clean):
            for w in s.lower().split():
                if w in vocab:
                    vecs[i, vocab[w]] += 1
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vecs = vecs / norms
        sims = np.array([
            float(np.dot(vecs[i], vecs[i + 1]))
            for i in range(len(vecs) - 1)
        ])

    variance = float(np.var(sims))
    mean_val = float(np.mean(sims))

    if len(sims) >= 4 and sims.std() > 1e-9:
        with np.errstate(invalid="ignore"):
            r = np.corrcoef(sims[:-2], sims[2:])[0, 1]
        lag2_autocorr = 0.0 if not np.isfinite(r) else float(r)
    else:
        lag2_autocorr = 0.0

    return variance, lag2_autocorr, mean_val


# ---------------------------------------------------------------------------
# Features: epistemic and marker densities (c6, c7, c8)
# ---------------------------------------------------------------------------

def _count_per_1k(text: str, *pattern_keys: str) -> float:
    """Count regex matches for the given lexicon categories, normalised per 1 000 words.

    Args:
        text: The document string to search.
        *pattern_keys: One or more category keys from the epistemic lexicon
            (e.g. ``"hedges"``, ``"first_person_opinion"``).

    Returns:
        float: Total match count ÷ word_count × 1000.  Returns 0.0 if the
        text is empty or no patterns match any key.
    """
    words = len(text.split())
    if words == 0:
        return 0.0
    count = 0
    patterns = _get_patterns()
    for key in pattern_keys:
        if key in patterns:
            count += len(patterns[key].findall(text))
    return count / words * 1000.0


def epistemic_density(text: str) -> float:
    """Feature c6: epistemic hedge and uncertainty marker density.

    Counts occurrences of hedges, epistemic verbs ("seems", "appears"),
    epistemic phrases ("to my knowledge"), uncertainty adverbs ("perhaps"),
    qualification phrases, and attribution phrases; normalised per 1 000 words.

    AI text tends to assert claims confidently without hedging; human
    experts hedge frequently.  Typical human values: 10–30 per 1k words;
    typical AI values: 2–10 per 1k words.

    Args:
        text: The document string.

    Returns:
        float: Epistemic marker rate per 1 000 words.
    """
    return _count_per_1k(
        text,
        "hedges",
        "epistemic_verbs",
        "epistemic_phrases",
        "uncertainty_adverbs",
        "qualification_phrases",
        "attribution_phrases",
    )


def first_person_opinion_density(text: str) -> float:
    """Feature c7: first-person opinion marker density.

    Counts phrases like "I think", "I believe", "in my opinion" — markers
    of personal stance that are rarely produced by AI models (which default
    to impersonal or third-person phrasing).

    Args:
        text: The document string.

    Returns:
        float: First-person opinion marker rate per 1 000 words.
    """
    return _count_per_1k(text, "first_person_opinion")


def self_correction_density(text: str) -> float:
    """Feature c8: self-correction and clarification marker density.

    Counts phrases like "actually", "I mean", "what I meant was",
    "to clarify" — markers of in-text revision that reflect genuine
    cognition.  LLMs produce auto-regressive text without backtracking.

    Args:
        text: The document string.

    Returns:
        float: Self-correction marker rate per 1 000 words.
    """
    return _count_per_1k(text, "self_correction")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_cognitive_features(
    text: str,
    sentences: list[str] | None = None,
    encoder: Any | None = None,
) -> CognitiveFeatures:
    """Compute all 8 cognitive load features for *text*.

    This is the public entry point for Layer 2 of the CIPHER pipeline.
    All 8 features are always available (no external corpus required),
    though c3–c5 improve significantly with a neural sentence encoder.

    Args:
        text: The document to analyse.  Should be at least 50 words.
        sentences: Pre-tokenised sentence list.  If None, a simple regex
            splitter on ``.!?`` is applied (less accurate than spaCy).
        encoder: Optional sentence-transformers ``SentenceTransformer``
            instance for dense coherence features (c3–c5).

    Returns:
        CognitiveFeatures: Dataclass with all 8 feature values and an
        ``available`` list (all 8 are always in the list).
    """
    if sentences is None:
        # Naive sentence splitter — prefer spaCy-split sentences when available
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    available: list[str] = []

    c1 = sentence_length_kl_diff(sentences)
    available.append("c1")

    c2 = vocabulary_novelty_hurst(text)
    available.append("c2")

    c3, c4, c5 = semantic_coherence_features(sentences, encoder)
    available.extend(["c3", "c4", "c5"])

    c6 = epistemic_density(text)
    available.append("c6")

    c7 = first_person_opinion_density(text)
    available.append("c7")

    c8 = self_correction_density(text)
    available.append("c8")

    return CognitiveFeatures(
        c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
        c6=c6, c7=c7, c8=c8,
        available=available,
    )
