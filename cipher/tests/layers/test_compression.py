"""Tests for Layer 1: compression-theoretic features."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cipher.layers.compression import (
    _lz76_complexity,
    _ncd,
    normalized_lz_complexity,
    window_entropy_features,
    within_doc_features,
    compute_compression_features,
)


# ---------------------------------------------------------------------------
# LZ76 complexity
# ---------------------------------------------------------------------------

class TestLZ76Complexity:
    def test_empty(self) -> None:
        assert _lz76_complexity(b"") == 0

    def test_single_byte(self) -> None:
        assert _lz76_complexity(b"a") == 1

    def test_all_same_byte(self) -> None:
        # Highly repetitive → very low complexity (implementation-defined ceiling ≤ 8)
        result = _lz76_complexity(b"a" * 100)
        assert result <= 8

    def test_random_bytes_higher_than_repetitive(self) -> None:
        import os
        repetitive = _lz76_complexity(b"ab" * 50)
        random_data = _lz76_complexity(os.urandom(100))
        assert random_data >= repetitive

    def test_monotone_under_repetition(self) -> None:
        # A repeated sequence should have lower or equal complexity than the original
        base = b"the quick brown fox "
        repeated = base * 5
        c_base = _lz76_complexity(base)
        c_rep = _lz76_complexity(repeated)
        # Normalised complexity should be lower for repeated
        assert c_rep / len(repeated) <= c_base / len(base) + 0.01


class TestNormalizedLZComplexity:
    def test_returns_float_in_range(self) -> None:
        score = normalized_lz_complexity("Hello world " * 10)
        assert 0.0 <= score <= 10.0  # large upper bound — not strictly bounded

    def test_repetitive_text_lower_than_diverse(self) -> None:
        repetitive = "the cat sat on the mat. " * 20
        diverse = (
            "Quantum mechanics governs subatomic phenomena. "
            "The GDP of emerging markets fluctuated sharply. "
            "Baroque architecture flourished in seventeenth-century Europe. "
        ) * 5
        assert normalized_lz_complexity(repetitive) <= normalized_lz_complexity(diverse)

    def test_short_text_returns_float(self) -> None:
        result = normalized_lz_complexity("Hi")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# NCD
# ---------------------------------------------------------------------------

class TestNCD:
    def test_identical_texts(self) -> None:
        a = b"the quick brown fox jumps over the lazy dog " * 5
        score = _ncd(a, a)
        assert 0.0 <= score <= 0.5  # identical → low NCD

    def test_range(self) -> None:
        a = b"hello world " * 10
        b = b"goodbye cruel world " * 10
        score = _ncd(a, b)
        assert 0.0 <= score <= 1.5  # NCD is bounded [0, 1] approximately

    def test_empty_inputs(self) -> None:
        score = _ncd(b"", b"test")
        assert 0.0 <= score <= 1.5


# ---------------------------------------------------------------------------
# Within-document features
# ---------------------------------------------------------------------------

class TestWithinDocFeatures:
    def test_returns_two_floats(self) -> None:
        text = "word " * 200
        l4, l5 = within_doc_features(text)
        assert isinstance(l4, float)
        assert isinstance(l5, float)

    def test_l4_in_expected_range(self) -> None:
        text = "the cat sat on the mat. " * 40
        l4, _ = within_doc_features(text)
        # Redundancy ratio should be < 1 (compression helps)
        assert 0.3 <= l4 <= 1.0

    def test_short_text_returns_defaults(self) -> None:
        l4, l5 = within_doc_features("short")
        assert l4 == 0.5
        assert l5 == 0.0


# ---------------------------------------------------------------------------
# Window entropy
# ---------------------------------------------------------------------------

class TestWindowEntropyFeatures:
    def test_short_text_returns_zeros(self) -> None:
        l6, l7 = window_entropy_features("tiny text", window_size=500)
        assert l6 == 0.0
        assert l7 == 0.0

    def test_uniform_text_low_variance(self) -> None:
        text = "a" * 5000
        l6, _ = window_entropy_features(text)
        assert l6 < 0.01  # nearly constant entropy

    def test_mixed_text_higher_variance(self) -> None:
        # Mix of very different content in windows
        import os
        mixed = ("aaa " * 200) + (os.urandom(300).hex() * 2)
        l6_mixed, _ = window_entropy_features(mixed)
        l6_uniform, _ = window_entropy_features("a" * 5000)
        assert l6_mixed >= l6_uniform


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestComputeCompressionFeatures:
    def test_basic(self) -> None:
        text = "This is a test document. " * 30
        feats = compute_compression_features(text)
        assert feats.l1 > 0
        assert "l1" in feats.available

    def test_with_corpora(self) -> None:
        text = "The model generated this text. " * 20
        ai_corpus = "The AI wrote this. " * 100
        human_corpus = "I believe the results show. " * 100
        feats = compute_compression_features(text, ai_corpus, human_corpus)
        assert "l2" in feats.available
        assert "l3" in feats.available
        assert 0.0 <= feats.l2 <= 1.5
        assert 0.0 <= feats.l3 <= 1.5

    def test_to_array_shape(self) -> None:
        text = "word " * 100
        feats = compute_compression_features(text)
        arr = feats.to_array()
        assert arr.shape == (7,)
