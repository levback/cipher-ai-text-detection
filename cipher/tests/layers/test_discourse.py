"""Tests for Layer 3: discourse topology features."""

from __future__ import annotations

import numpy as np
import pytest

from cipher.layers.discourse import (
    DiscourseFeatures,
    _js_divergence,
    _tfidf_paragraph_vectors,
    compute_discourse_features,
    topic_boundary_features,
    stats_skewness,
)


# ---------------------------------------------------------------------------
# JS divergence
# ---------------------------------------------------------------------------

class TestJSDivergence:
    def test_identical_distributions_zero(self) -> None:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert _js_divergence(p, p) < 1e-10

    def test_orthogonal_distributions_high(self) -> None:
        p = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0, 0.0])
        result = _js_divergence(p + 1e-10, q + 1e-10)
        assert result > 0.5

    def test_output_range(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            p = rng.dirichlet(np.ones(8))
            q = rng.dirichlet(np.ones(8))
            js = _js_divergence(p, q)
            assert 0.0 <= js <= 1.0


# ---------------------------------------------------------------------------
# TF-IDF vectors
# ---------------------------------------------------------------------------

class TestTFIDFParagraphVectors:
    def test_output_shape(self) -> None:
        paragraphs = ["hello world", "foo bar baz", "hello foo"]
        vecs = _tfidf_paragraph_vectors(paragraphs)
        assert vecs.shape[0] == 3

    def test_l2_normalised(self) -> None:
        paragraphs = ["the quick brown fox", "jumps over the lazy dog", "fox and dog"]
        vecs = _tfidf_paragraph_vectors(paragraphs)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_different_paragraphs_different_vectors(self) -> None:
        p1 = "quantum physics mechanics wave function"
        p2 = "cooking recipe pasta sauce garlic tomato"
        vecs = _tfidf_paragraph_vectors([p1, p2])
        cosine = float(vecs[0] @ vecs[1])
        assert cosine < 0.8  # should be dissimilar


# ---------------------------------------------------------------------------
# Topic boundary features
# ---------------------------------------------------------------------------

class TestTopicBoundaryFeatures:
    def test_abrupt_transitions_higher_variance(self) -> None:
        # Paragraphs that jump between very different topics
        paragraphs = [
            "quantum mechanics uncertainty principle electron orbital",
            "the french revolution began with storming the bastille",
            "quantum tunneling enables electrons to pass through barriers",
            "napoleon led french forces to victory across europe",
            "wave particle duality is central to quantum physics",
        ]
        var_abrupt, _ = topic_boundary_features(paragraphs)

        # Paragraphs on same topic
        similar = [
            "the cat sat on the mat and looked around",
            "the cat then jumped off the mat onto the floor",
            "the cat walked to the food bowl and ate",
            "the cat returned to the mat after eating",
            "the cat fell asleep on the warm mat again",
        ]
        var_similar, _ = topic_boundary_features(similar)

        assert var_abrupt >= var_similar

    def test_returns_floats(self) -> None:
        paragraphs = ["paragraph one content here", "paragraph two content", "paragraph three"]
        d4, d5 = topic_boundary_features(paragraphs)
        assert isinstance(d4, float)
        assert isinstance(d5, float)

    def test_too_few_paragraphs_returns_defaults(self) -> None:
        d4, d5 = topic_boundary_features(["only one paragraph"])
        assert d4 == 0.0
        assert d5 == 0.0


# ---------------------------------------------------------------------------
# Skewness
# ---------------------------------------------------------------------------

class TestStatsSkewness:
    def test_symmetric_near_zero(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(stats_skewness(arr)) < 0.1

    def test_right_skewed_positive(self) -> None:
        arr = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0])
        assert stats_skewness(arr) > 0

    def test_short_returns_zero(self) -> None:
        assert stats_skewness(np.array([1.0, 2.0])) == 0.0


# ---------------------------------------------------------------------------
# Integration (no spaCy)
# ---------------------------------------------------------------------------

class TestComputeDiscourseFeatures:
    def test_basic_without_spacy(self) -> None:
        text = "\n\n".join([
            "The cat sat on the mat. It looked around the room. The cat was happy.",
            "Quantum mechanics describes how particles behave at subatomic scales.",
            "The French revolution transformed European politics and society.",
            "The cat returned to the mat and fell asleep in the afternoon sun.",
        ])
        feats = compute_discourse_features(text, nlp=None)
        assert isinstance(feats, DiscourseFeatures)
        assert "d4" in feats.available
        assert "d5" in feats.available

    def test_to_array_shape(self) -> None:
        text = "para one. " * 5 + "\n\n" + "para two. " * 5 + "\n\n" + "para three. " * 5
        feats = compute_discourse_features(text, nlp=None)
        arr = feats.to_array()
        assert arr.shape == (7,)

    def test_fallback_values_in_range(self) -> None:
        text = "sentence. " * 30
        feats = compute_discourse_features(text, nlp=None)
        # All features should be finite
        import math
        for val in feats.to_array():
            assert math.isfinite(val)
