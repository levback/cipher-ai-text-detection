"""Layer 6: Sentence-level attribution.

Produces a per-sentence AI probability score by combining:
1. Per-sentence LZ76 complexity (compression signal)
2. Inverse semantic coherence to adjacent sentences (discontinuity signal)
3. SHAP contribution (when a trained meta-learner is available)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from cipher.layers.compression import normalized_lz_complexity


@dataclass
class SentenceScore:
    index: int
    text: str
    score: float  # AI probability [0, 1]
    is_highlighted: bool

    def char_range(self, full_text: str) -> tuple[int, int]:
        start = full_text.find(self.text)
        if start == -1:
            return (0, 0)
        return (start, start + len(self.text))


def _sentence_lzc_scores(sentences: list[str]) -> np.ndarray:
    """Normalised LZC for each sentence; short sentences get median score."""
    scores = np.array([
        normalized_lz_complexity(s) if len(s.split()) >= 5 else np.nan
        for s in sentences
    ])
    median = float(np.nanmedian(scores)) if not np.all(np.isnan(scores)) else 0.5
    scores = np.where(np.isnan(scores), median, scores)
    # Convert to AI signal: low LZC → high AI score
    if scores.max() > scores.min():
        normalised = 1.0 - (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalised = np.full_like(scores, 0.5)
    return normalised


def _sentence_coherence_scores(
    sentences: list[str],
    encoder: Any | None = None,
) -> np.ndarray:
    """Per-sentence "AI coherence": high cosine sim to neighbours → AI signal."""
    n = len(sentences)
    if n < 2:
        return np.full(n, 0.5)

    if encoder is not None:
        embeddings: np.ndarray = encoder.encode(sentences, show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings /= norms
    else:
        # Fallback: BoW cosine
        vocab: dict[str, int] = {}
        for s in sentences:
            for w in s.lower().split():
                if w not in vocab:
                    vocab[w] = len(vocab)
        embeddings = np.zeros((n, max(len(vocab), 1)))
        for i, s in enumerate(sentences):
            for w in s.lower().split():
                if w in vocab:
                    embeddings[i, vocab[w]] += 1
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings /= norms

    # For each sentence, average cosine sim with neighbours
    sims = np.zeros(n)
    for i in range(n):
        neighbours = []
        if i > 0:
            neighbours.append(float(embeddings[i] @ embeddings[i - 1]))
        if i < n - 1:
            neighbours.append(float(embeddings[i] @ embeddings[i + 1]))
        sims[i] = np.mean(neighbours) if neighbours else 0.5

    # High coherence → AI signal
    if sims.max() > sims.min():
        normalised = (sims - sims.min()) / (sims.max() - sims.min())
    else:
        normalised = np.full_like(sims, 0.5)
    return normalised


def compute_sentence_scores(
    text: str,
    sentences: list[str],
    document_score: float,
    encoder: Any | None = None,
    highlight_threshold: float = 0.60,
    weights: tuple[float, float] = (0.45, 0.55),
) -> list[SentenceScore]:
    """Compute per-sentence AI probability scores.

    Parameters
    ----------
    text:
        Full document text (used for character offset lookup).
    sentences:
        Pre-tokenised sentence list.
    document_score:
        Overall document-level AI probability (used to scale sentence scores).
    encoder:
        Optional sentence-transformers encoder for coherence features.
    highlight_threshold:
        Sentences with score ≥ this are marked as highlighted.
    weights:
        (w_compression, w_coherence) — must sum to 1.
    """
    clean = [s.strip() for s in sentences if s.strip()]
    if not clean:
        return []

    w_comp, w_coh = weights

    lzc_scores = _sentence_lzc_scores(clean)
    coh_scores = _sentence_coherence_scores(clean, encoder)

    combined = w_comp * lzc_scores + w_coh * coh_scores

    # Anchor sentence scores so their mean equals the document score
    if combined.std() > 1e-8:
        mean_before = combined.mean()
        if mean_before > 1e-8:
            combined = combined * (document_score / mean_before)
    combined = np.clip(combined, 0.0, 1.0)

    results: list[SentenceScore] = []
    for i, (sent, score) in enumerate(zip(clean, combined)):
        results.append(
            SentenceScore(
                index=i,
                text=sent,
                score=float(score),
                is_highlighted=float(score) >= highlight_threshold,
            )
        )
    return results


@dataclass
class HighlightedSpan:
    start: int
    end: int
    score: float


def extract_highlighted_spans(
    text: str,
    sentence_scores: list[SentenceScore],
) -> list[HighlightedSpan]:
    """Merge consecutive highlighted sentences into contiguous character spans."""
    spans: list[HighlightedSpan] = []
    current_start: int | None = None
    current_end: int | None = None
    current_score = 0.0
    score_count = 0

    for ss in sentence_scores:
        if not ss.is_highlighted:
            if current_start is not None:
                spans.append(
                    HighlightedSpan(
                        start=current_start,
                        end=current_end or 0,
                        score=current_score / max(score_count, 1),
                    )
                )
                current_start = None
                current_end = None
                current_score = 0.0
                score_count = 0
            continue

        start, end = ss.char_range(text)
        if start == 0 and end == 0:
            continue

        if current_start is None:
            current_start = start
        current_end = end
        current_score += ss.score
        score_count += 1

    if current_start is not None:
        spans.append(
            HighlightedSpan(
                start=current_start,
                end=current_end or 0,
                score=current_score / max(score_count, 1),
            )
        )
    return spans
