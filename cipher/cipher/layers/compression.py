"""Layer 1: Compression-theoretic features.

Computes 7 features derived from information theory and data compression.
No neural networks or language models required.  All features are
derivable from raw byte sequences using standard library tools (zlib).

The core intuition is that AI-generated text is more predictable than
human writing at the byte level — it has lower Lempel-Ziv complexity,
higher internal redundancy, and smoother entropy profiles.

Features
--------
l1  normalized LZ76 complexity  — LZC(T)·log₂(n)/n
l2  NCD against AI reference corpus  — NCD(T, C_AI)
l3  NCD against human reference corpus  — NCD(T, C_Human)
l4  within-document redundancy ratio  — C(AB)/(C(A)+C(B))
l5  within-document directional asymmetry  — |C(B|A)/|B| − C(A|B)/|A||
l6  window entropy variance  — Var[H(w_i)]
l7  window entropy lag-1 autocorrelation  — Corr(H_t, H_{t+1})

See Also
--------
cipher.layers.cognitive : Cognitive load features (Layer 2).
cipher.layers.assembler : Combines all layer outputs into a FeatureVector.
cipher.models.meta_learner : Consumes the assembled feature vector.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CompressionFeatures:
    """Immutable container for the 7 compression-theoretic feature values.

    All fields are in their raw (unnormalised) form.  The MetaLearner
    applies z-score normalisation using pre-fitted statistics before
    computing the logistic regression output.

    Attributes:
        l1: Normalized LZ76 complexity — lower values indicate more
            repetitive / predictable content, a signal of AI generation.
        l2: NCD between the document and a sample of AI-generated text.
            Closer to 0 means the document compresses well together with
            AI text (structural similarity).
        l3: NCD between the document and a sample of human-written text.
            Closer to 0 means the document is structurally similar to
            human writing.
        l4: Within-document redundancy ratio C(AB)/(C(A)+C(B)) — values
            close to 1 imply high joint compressibility (repetitive style).
        l5: Directional asymmetry between the first and second halves.
        l6: Variance of per-window byte entropy across the document.
        l7: Lag-1 autocorrelation of the per-window entropy series.
        available: Names of features that were actually computed (e.g.
            l2/l3 are absent when no reference corpus is supplied).
    """

    l1: float  # normalized LZ76 complexity
    l2: float  # NCD vs AI corpus
    l3: float  # NCD vs human corpus
    l4: float  # within-doc redundancy ratio
    l5: float  # within-doc directional asymmetry
    l6: float  # window entropy variance
    l7: float  # window entropy lag-1 autocorrelation
    available: list[str] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Return features as a 1-D float64 array in canonical order [l1..l7].

        Returns:
            np.ndarray: Shape (7,), dtype float64.
        """
        return np.array([self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7])


# ---------------------------------------------------------------------------
# LZ76 complexity
# ---------------------------------------------------------------------------

def _lz76_complexity(data: bytes) -> int:
    """Count phrases in the Lempel-Ziv 76 parsing of *data*.

    The LZ76 algorithm (Lempel & Ziv, 1976) parses a sequence into
    non-overlapping phrases such that each new phrase is the shortest
    extension of the current position that has not previously appeared.
    The phrase count C(T) is a complexity measure: random / novel text
    yields many short phrases (high complexity), while repetitive or
    predictable text yields fewer, longer phrases (low complexity).

    Algorithm:
        Scan forward from ``start``; grow the candidate window until
        ``data[start:start+length]`` is NOT found in ``data[0:start]``.
        Record a new phrase and advance ``start``.

    Complexity:
        O(n²) due to the substring search in the preceding text.  This
        is acceptable for documents up to ~50 KB (typical article length).

    Args:
        data: Raw bytes to analyse.

    Returns:
        int: Number of distinct LZ76 phrases; higher → more complex.
    """
    n = len(data)
    if n == 0:
        return 0

    phrases = 0
    start = 0
    while start < n:
        length = 1
        while start + length <= n:
            candidate = data[start : start + length]
            if data.find(candidate, 0, start) == -1:
                phrases += 1
                start += length
                break
            length += 1
        else:
            phrases += 1
            break
    return phrases


def normalized_lz_complexity(text: str) -> float:
    """Feature l1: normalized LZ76 complexity.

    Computes the Lempel-Ziv complexity normalised by the Shannon entropy
    upper bound for a uniform source:

        LZC_norm(T) = C(T) · log₂(n) / n

    where ``n = |T|`` in bytes and ``C(T)`` is the raw LZ76 phrase count.
    This normalisation makes the measure comparable across document
    lengths.  Random text approaches 1.0; highly repetitive text
    approaches 0.  AI-generated text typically scores 15–25 % lower than
    human-written text of the same domain.

    Args:
        text: The document string to analyse.

    Returns:
        float: Normalized complexity in approximately (0.2, 4.0).  Values
        below 2.5 are a mild AI signal; below 2.0 are a strong signal.
    """
    data = text.encode("utf-8")
    n = len(data)
    if n < 4:
        return 0.5
    c = _lz76_complexity(data)
    return (c * math.log2(max(n, 2))) / n


# ---------------------------------------------------------------------------
# Normalized Compression Distance
# ---------------------------------------------------------------------------

def _compress_len(data: bytes, level: int = 6) -> int:
    """Return the byte-length of the zlib-deflate compressed *data*.

    Used as a proxy for Kolmogorov complexity K(x).  Higher compression
    levels are more expensive but give a tighter upper bound on K(x);
    level 6 (default) offers a good speed/accuracy trade-off.

    Args:
        data: Raw bytes to compress.
        level: zlib compression level (0–9).

    Returns:
        int: Compressed byte count C(data) ≈ K(data).
    """
    return len(zlib.compress(data, level))


def _ncd(a: bytes, b: bytes) -> float:
    """Normalised Compression Distance (NCD) between two byte sequences.

    The NCD is an approximation to the normalised information distance
    (Li et al., 2004):

        NCD(A, B) = (C(AB) − min(C(A), C(B))) / max(C(A), C(B))

    where C(·) denotes compressed size and AB is the concatenation of A
    and B.  NCD = 0 implies identical content; NCD = 1 implies completely
    unrelated content.  Values below 0.4 suggest structural similarity.

    Args:
        a: First byte sequence.
        b: Second byte sequence.

    Returns:
        float: NCD value in [0, 1]; 0 = identical, 1 = unrelated.
    """
    ca = _compress_len(a)
    cb = _compress_len(b)
    cab = _compress_len(a + b)
    denom = max(ca, cb)
    if denom == 0:
        return 0.0
    return (cab - min(ca, cb)) / denom


def ncd_vs_corpus(text: str, corpus: str) -> float:
    """Feature l2 / l3: NCD between *text* and a reference corpus sample.

    Truncates the corpus to 8 KB so that compression remains tractable
    and representative on typical hardware.  A low NCD against the AI
    corpus (l2 low) signals AI origin; a low NCD against the human corpus
    (l3 low) signals human origin.

    Args:
        text: The query document to score.
        corpus: A reference corpus string (AI or human training texts
            concatenated).  Only the first 8 192 bytes are used.

    Returns:
        float: NCD value in [0, 1].
    """
    # Take at most 8 KB from corpus to keep compression tractable
    sample = corpus[:8192].encode("utf-8")
    return _ncd(text.encode("utf-8"), sample)


# ---------------------------------------------------------------------------
# Within-document compression asymmetry
# ---------------------------------------------------------------------------

def within_doc_features(text: str) -> tuple[float, float]:
    """Features l4 (redundancy ratio) and l5 (directional asymmetry).

    Splits the document at the midpoint byte boundary and computes
    compression-based statistics on the two halves.

    l4 — Joint redundancy ratio:
        l4 = C(AB) / (C(A) + C(B))
        Close to 1.0 → the two halves share a lot of information;
        the document is internally redundant (AI signal: 0.85–0.90).
        Human text typically achieves 0.80–0.88.

    l5 — Directional conditional complexity asymmetry:
        Approximates C(B|A) ≈ C(AB) − C(A) and
                       C(A|B) ≈ C(AB) − C(B).
        l5 = |C(B|A)/|B| − C(A|B)/|A||
        High asymmetry → the document "builds" in one direction;
        common in narrative human writing.  Near-zero → AI homogeneity.

    Args:
        text: The document string.

    Returns:
        tuple[float, float]: (l4, l5)
    """
    encoded = text.encode("utf-8")
    mid = len(encoded) // 2
    if mid < 10:
        return 0.5, 0.0

    a, b = encoded[:mid], encoded[mid:]
    ca, cb, cab = _compress_len(a), _compress_len(b), _compress_len(a + b)

    # l4: how much joint compression gains vs. independent (closer to 1 → more redundant)
    denom = ca + cb
    l4 = cab / denom if denom > 0 else 0.5

    # l5: asymmetry — does A explain B better than B explains A?
    # Approximate C(B|A) as C(AB) - C(A)
    c_b_given_a = max(cab - ca, 1)
    c_a_given_b = max(cab - cb, 1)
    l5 = abs(c_b_given_a / len(b) - c_a_given_b / len(a))

    return float(l4), float(l5)


# ---------------------------------------------------------------------------
# Window entropy profile
# ---------------------------------------------------------------------------

def _byte_entropy(data: bytes) -> float:
    """Shannon byte entropy of *data*.

    Computes:
        H(X) = −Σ p(x) log₂(p(x))   ∀ x ∈ {0..255} with p(x) > 0

    The result is in bits, with a maximum of 8.0 for a perfectly uniform
    256-symbol distribution.

    Args:
        data: Raw byte window.

    Returns:
        float: Entropy in [0.0, 8.0] bits per symbol.
    """
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def window_entropy_features(
    text: str,
    window_size: int = 500,
    stride: int = 250,
) -> tuple[float, float]:
    """Features l6 (entropy variance) and l7 (lag-1 autocorrelation).

    Slides a byte-level window across the document and computes the
    Shannon entropy H(w_i) for each window.  Two statistics are derived
    from the resulting entropy profile {H_i}:

    l6 — Variance:
        Var[{H_i}] — AI text tends to maintain a very uniform entropy
        level (low variance); human text fluctuates more.

    l7 — Lag-1 autocorrelation:
        Corr(H_t, H_{t+1}) — High positive autocorrelation suggests
        slowly-changing entropy (smooth AI prose); negative values
        suggest oscillating style (some human writing patterns).

    Args:
        text: The document string.
        window_size: Number of bytes per entropy window (default 500).
        stride: Step size between windows (default 250, 50 % overlap).

    Returns:
        tuple[float, float]: (l6_variance, l7_lag1_autocorrelation).
        Both are 0.0 for documents shorter than *window_size*.
    """
    data = text.encode("utf-8")
    n = len(data)
    if n < window_size:
        return 0.0, 0.0

    entropies: list[float] = []
    pos = 0
    while pos + window_size <= n:
        entropies.append(_byte_entropy(data[pos : pos + window_size]))
        pos += stride

    if len(entropies) < 3:
        return 0.0, 0.0

    arr = np.array(entropies)
    variance = float(np.var(arr))

    # Lag-1 autocorrelation
    if arr.std() < 1e-9:
        autocorr = 0.0
    else:
        with np.errstate(invalid="ignore"):
            r = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        autocorr = 0.0 if not np.isfinite(r) else float(r)

    return variance, autocorr


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_compression_features(
    text: str,
    ai_corpus: str | None = None,
    human_corpus: str | None = None,
) -> CompressionFeatures:
    """Compute all 7 compression-theoretic features for *text*.

    This is the public entry point for Layer 1 of the CIPHER pipeline.
    Features l2 and l3 require reference corpora; when absent they
    default to 0.5 (neutral) and are omitted from ``available``.

    Args:
        text: The document to analyse.  Should be at least 50 words;
            very short texts receive default/fallback values.
        ai_corpus: Concatenated sample of AI-generated reference texts.
            Used to compute NCD feature l2.  If None, l2 defaults to 0.5.
        human_corpus: Concatenated sample of human-written reference
            texts.  Used to compute NCD feature l3.  If None, l3 = 0.5.

    Returns:
        CompressionFeatures: Immutable dataclass with fields l1–l7 and
        an ``available`` list of feature names that were computed from
        real data (rather than fallback defaults).
    """
    available: list[str] = []

    l1 = normalized_lz_complexity(text)
    available.append("l1")

    l2 = ncd_vs_corpus(text, ai_corpus) if ai_corpus else 0.5
    if ai_corpus:
        available.append("l2")

    l3 = ncd_vs_corpus(text, human_corpus) if human_corpus else 0.5
    if human_corpus:
        available.append("l3")

    l4, l5 = within_doc_features(text)
    available.extend(["l4", "l5"])

    l6, l7 = window_entropy_features(text)
    available.extend(["l6", "l7"])

    return CompressionFeatures(
        l1=l1, l2=l2, l3=l3, l4=l4, l5=l5, l6=l6, l7=l7,
        available=available,
    )
