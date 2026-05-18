"""Tests for the sentence-level attribution highlighter module.

Covers compute_sentence_scores, SentenceScore, HighlightedSpan, and
extract_highlighted_spans.
"""

from __future__ import annotations

import pytest

from cipher.output.highlighter import (
    HighlightedSpan,
    SentenceScore,
    compute_sentence_scores,
    extract_highlighted_spans,
)

# Representative multi-sentence text
MULTI_SENTENCE_TEXT = (
    "The quick brown fox jumps over the lazy dog near the riverbank. "
    "Scientists have long studied the migratory patterns of birds across continents. "
    "The results demonstrated a significant correlation between the two variables measured."
)

THREE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Scientists have long studied the migratory patterns of birds across continents.",
    "The results demonstrated a significant correlation between the two variables measured.",
]


# ---------------------------------------------------------------------------
# SentenceScore
# ---------------------------------------------------------------------------


class TestSentenceScore:
    """Tests for the SentenceScore dataclass."""

    def test_is_highlighted_is_bool(self) -> None:
        """is_highlighted field should be a bool."""
        ss = SentenceScore(index=0, text="Hello world.", score=0.8, is_highlighted=True)
        assert isinstance(ss.is_highlighted, bool)

    def test_score_is_float(self) -> None:
        ss = SentenceScore(index=0, text="Hello world.", score=0.8, is_highlighted=True)
        assert isinstance(ss.score, float)

    def test_char_range_found(self) -> None:
        """char_range returns valid (start, end) when text is in full_text."""
        full_text = "First sentence. Second sentence. Third sentence."
        ss = SentenceScore(index=1, text="Second sentence.", score=0.5, is_highlighted=False)
        start, end = ss.char_range(full_text)
        assert start >= 0
        assert end > start
        assert full_text[start:end] == "Second sentence."

    def test_char_range_not_found_returns_zero(self) -> None:
        """char_range returns (0, 0) when text not found in full_text."""
        ss = SentenceScore(index=0, text="not present", score=0.5, is_highlighted=False)
        assert ss.char_range("some other text") == (0, 0)

    def test_char_range_tuple_type(self) -> None:
        """char_range returns a tuple of two ints."""
        ss = SentenceScore(index=0, text="Hello.", score=0.5, is_highlighted=False)
        result = ss.char_range("Hello. World.")
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# compute_sentence_scores
# ---------------------------------------------------------------------------


class TestComputeSentenceScores:
    """Tests for compute_sentence_scores() without an encoder."""

    def test_returns_list_of_sentence_scores(self) -> None:
        """Should return a list of SentenceScore objects."""
        results = compute_sentence_scores(
            text=MULTI_SENTENCE_TEXT,
            sentences=THREE_SENTENCES,
            document_score=0.6,
            encoder=None,
        )
        assert isinstance(results, list)
        assert len(results) == 3
        for item in results:
            assert isinstance(item, SentenceScore)

    def test_scores_in_unit_range(self) -> None:
        """Every sentence score should be in [0, 1]."""
        results = compute_sentence_scores(
            text=MULTI_SENTENCE_TEXT,
            sentences=THREE_SENTENCES,
            document_score=0.5,
            encoder=None,
        )
        for ss in results:
            assert 0.0 <= ss.score <= 1.0

    def test_is_highlighted_is_bool(self) -> None:
        """is_highlighted attribute should be a Python bool."""
        results = compute_sentence_scores(
            text=MULTI_SENTENCE_TEXT,
            sentences=THREE_SENTENCES,
            document_score=0.5,
            encoder=None,
        )
        for ss in results:
            assert isinstance(ss.is_highlighted, bool)

    def test_indices_are_sequential(self) -> None:
        """Sentence indices should be 0, 1, 2, ..."""
        results = compute_sentence_scores(
            text=MULTI_SENTENCE_TEXT,
            sentences=THREE_SENTENCES,
            document_score=0.5,
            encoder=None,
        )
        for i, ss in enumerate(results):
            assert ss.index == i

    def test_single_sentence_returns_one_score(self) -> None:
        """A single-sentence input should produce exactly one SentenceScore."""
        text = "A single sentence that is fairly long and detailed enough to process well."
        results = compute_sentence_scores(
            text=text,
            sentences=[text],
            document_score=0.4,
            encoder=None,
        )
        assert len(results) == 1

    def test_empty_sentences_returns_empty(self) -> None:
        """No sentences should produce an empty list."""
        results = compute_sentence_scores(
            text="",
            sentences=[],
            document_score=0.5,
            encoder=None,
        )
        assert results == []

    def test_whitespace_only_sentences_filtered(self) -> None:
        """Whitespace-only strings in sentences should be filtered out."""
        results = compute_sentence_scores(
            text=MULTI_SENTENCE_TEXT,
            sentences=["   ", THREE_SENTENCES[0], "  "],
            document_score=0.5,
            encoder=None,
        )
        assert len(results) == 1

    def test_high_threshold_fewer_highlighted(self) -> None:
        """Raising highlight_threshold should decrease highlighted count."""
        results_low = compute_sentence_scores(
            text=MULTI_SENTENCE_TEXT,
            sentences=THREE_SENTENCES,
            document_score=0.8,
            encoder=None,
            highlight_threshold=0.01,
        )
        results_high = compute_sentence_scores(
            text=MULTI_SENTENCE_TEXT,
            sentences=THREE_SENTENCES,
            document_score=0.8,
            encoder=None,
            highlight_threshold=0.99,
        )
        low_count = sum(1 for ss in results_low if ss.is_highlighted)
        high_count = sum(1 for ss in results_high if ss.is_highlighted)
        assert low_count >= high_count

    def test_document_score_anchors_mean(self) -> None:
        """Sentence scores should be anchored near the document score."""
        doc_score = 0.7
        results = compute_sentence_scores(
            text=MULTI_SENTENCE_TEXT,
            sentences=THREE_SENTENCES,
            document_score=doc_score,
            encoder=None,
        )
        scores = [ss.score for ss in results]
        # Mean should be within a reasonable range of doc_score
        mean_score = sum(scores) / len(scores)
        assert abs(mean_score - doc_score) < 0.5  # generous tolerance

    def test_with_encoder_none_does_not_crash(self) -> None:
        """encoder=None branch uses BoW cosine fallback."""
        results = compute_sentence_scores(
            text=MULTI_SENTENCE_TEXT,
            sentences=THREE_SENTENCES,
            document_score=0.5,
            encoder=None,
        )
        assert len(results) == 3


# ---------------------------------------------------------------------------
# HighlightedSpan
# ---------------------------------------------------------------------------


class TestHighlightedSpan:
    """Tests for the HighlightedSpan dataclass."""

    def test_dataclass_fields(self) -> None:
        """HighlightedSpan should expose start, end, score attributes."""
        span = HighlightedSpan(start=10, end=50, score=0.8)
        assert span.start == 10
        assert span.end == 50
        assert span.score == pytest.approx(0.8)

    def test_start_end_ordering(self) -> None:
        span = HighlightedSpan(start=5, end=25, score=0.7)
        assert span.start < span.end


# ---------------------------------------------------------------------------
# extract_highlighted_spans
# ---------------------------------------------------------------------------


class TestExtractHighlightedSpans:
    """Tests for extract_highlighted_spans() merging logic."""

    def test_returns_list(self) -> None:
        results = compute_sentence_scores(
            text=MULTI_SENTENCE_TEXT,
            sentences=THREE_SENTENCES,
            document_score=0.5,
            encoder=None,
        )
        spans = extract_highlighted_spans(MULTI_SENTENCE_TEXT, results)
        assert isinstance(spans, list)

    def test_no_highlighted_gives_empty_spans(self) -> None:
        """All unhighlighted sentences → no spans."""
        sentence_scores = [
            SentenceScore(index=i, text=s, score=0.0, is_highlighted=False)
            for i, s in enumerate(THREE_SENTENCES)
        ]
        spans = extract_highlighted_spans(MULTI_SENTENCE_TEXT, sentence_scores)
        assert spans == []

    def test_all_highlighted_gives_spans(self) -> None:
        """Highlighted sentences produce at least one span."""
        sentence_scores = [
            SentenceScore(index=i, text=s, score=0.9, is_highlighted=True)
            for i, s in enumerate(THREE_SENTENCES)
        ]
        spans = extract_highlighted_spans(MULTI_SENTENCE_TEXT, sentence_scores)
        assert len(spans) >= 1

    def test_span_scores_in_unit_range(self) -> None:
        """All span scores should be in [0, 1]."""
        sentence_scores = [
            SentenceScore(index=i, text=s, score=0.8, is_highlighted=True)
            for i, s in enumerate(THREE_SENTENCES)
        ]
        spans = extract_highlighted_spans(MULTI_SENTENCE_TEXT, sentence_scores)
        for span in spans:
            assert 0.0 <= span.score <= 1.0

    def test_empty_sentence_scores_returns_empty(self) -> None:
        spans = extract_highlighted_spans(MULTI_SENTENCE_TEXT, [])
        assert spans == []
