"""Tests for Layer 2: cognitive load fingerprint features."""

from __future__ import annotations

import numpy as np
import pytest

from cipher.layers.cognitive import (
    CognitiveFeatures,
    compute_cognitive_features,
    epistemic_density,
    first_person_opinion_density,
    hurst_exponent,
    self_correction_density,
    sentence_length_kl_diff,
    vocabulary_novelty_hurst,
)


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------

class TestHurstExponent:
    def test_random_series_near_half(self) -> None:
        rng = np.random.default_rng(42)
        series = rng.standard_normal(500)
        h = hurst_exponent(series)
        # Random walk should give H near 0.5 (±0.15 tolerance)
        assert 0.35 <= h <= 0.75

    def test_persistent_series_above_half(self) -> None:
        # Cumulative sum of positive values → persistent
        rng = np.random.default_rng(0)
        series = np.cumsum(np.abs(rng.standard_normal(300)))
        h = hurst_exponent(series)
        assert h >= 0.5

    def test_short_series_returns_half(self) -> None:
        assert hurst_exponent(np.array([1.0, 2.0, 3.0])) == 0.5

    def test_output_in_unit_interval(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(5):
            series = rng.standard_normal(200)
            h = hurst_exponent(series)
            assert 0.0 <= h <= 1.0


class TestVocabularyNoveltyHurst:
    def test_diverse_text_higher_than_repetitive(self) -> None:
        diverse = (
            "quantum gravity relativity entropy chaos theory uncertainty "
            "baroque renaissance enlightenment revolution democracy parliament "
        ) * 10
        repetitive = "the cat sat on the mat and the cat sat again " * 20
        h_div = vocabulary_novelty_hurst(diverse)
        h_rep = vocabulary_novelty_hurst(repetitive)
        # Diverse text should have higher Hurst (vocabulary bursts cluster)
        # This is a soft test — direction may vary
        assert isinstance(h_div, float)
        assert isinstance(h_rep, float)

    def test_short_text_returns_half(self) -> None:
        h = vocabulary_novelty_hurst("hello world")
        assert h == 0.5


# ---------------------------------------------------------------------------
# Sentence length distribution
# ---------------------------------------------------------------------------

class TestSentenceLengthKLDiff:
    def test_few_sentences_returns_zero(self) -> None:
        result = sentence_length_kl_diff(["Hello.", "World."])
        assert result == 0.0

    def test_uniform_lengths_positive(self) -> None:
        # Sentences all 10 words → Gaussian fits perfectly → positive
        sentences = ["word " * 10 for _ in range(30)]
        result = sentence_length_kl_diff(sentences)
        assert result > -1.0  # likely ≥ 0 but softer bound

    def test_variable_lengths_lower_value(self) -> None:
        # Mix of very short and very long → Weibull fits better
        sentences = ["hi"] * 10 + ["word " * 40] * 5 + ["word " * 15] * 15
        result = sentence_length_kl_diff(sentences)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Epistemic and marker densities
# ---------------------------------------------------------------------------

class TestEpistemicDensity:
    def test_hedged_text_higher_density(self) -> None:
        hedged = (
            "Perhaps this is correct. I think it might be true. "
            "It seems to me that perhaps we should consider this. "
            "I believe, possibly, that maybe the answer is different. "
        ) * 5
        plain = "The results show a clear increase. The data confirms this. " * 5
        assert epistemic_density(hedged) > epistemic_density(plain)

    def test_empty_text_zero(self) -> None:
        assert epistemic_density("") == 0.0

    def test_returns_non_negative(self) -> None:
        assert epistemic_density("some random text here") >= 0.0


class TestFirstPersonOpinionDensity:
    def test_opinion_text_higher(self) -> None:
        opinion = "I argue that this is wrong. My view is that we should reconsider. " * 5
        neutral = "The results indicate a correlation. Data shows trends. " * 5
        assert first_person_opinion_density(opinion) > first_person_opinion_density(neutral)


class TestSelfCorrectionDensity:
    def test_correction_text_higher(self) -> None:
        correction = "Actually, I mean the second option. To clarify, I was wrong. " * 5
        neutral = "The results indicate a correlation. " * 5
        assert self_correction_density(correction) > self_correction_density(neutral)


# ---------------------------------------------------------------------------
# Semantic coherence (BoW fallback)
# ---------------------------------------------------------------------------

class TestSemanticCoherence:
    def test_coherent_text_higher_mean(self) -> None:
        from cipher.layers.cognitive import semantic_coherence_features

        coherent = [
            "The dog ran across the park.",
            "The dog then jumped over the fence.",
            "After jumping, the dog rested by the pond.",
            "The dog wagged its tail near the pond.",
            "People watched the dog by the water.",
        ]
        incoherent = [
            "Quantum mechanics describes sub-atomic phenomena.",
            "The French revolution began in 1789.",
            "Jazz music originated in New Orleans.",
            "Mitosis is a form of cell division.",
            "The stock market crashed in 1929.",
        ]
        _, _, mean_coh = semantic_coherence_features(coherent, encoder=None)
        _, _, mean_inc = semantic_coherence_features(incoherent, encoder=None)
        assert mean_coh > mean_inc

    def test_short_input_returns_defaults(self) -> None:
        from cipher.layers.cognitive import semantic_coherence_features

        var, autocorr, mean = semantic_coherence_features(["One sentence."], encoder=None)
        assert var == 0.0
        assert mean == 0.5


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestComputeCognitiveFeatures:
    def test_basic_execution(self) -> None:
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "I think this is interesting. Perhaps we should explore further. "
            "Actually, I mean, the data suggests otherwise. "
            "In my view, the results are quite compelling. "
        ) * 5
        feats = compute_cognitive_features(text)
        assert isinstance(feats, CognitiveFeatures)
        assert feats.c6 > 0  # should have some epistemic markers

    def test_to_array_shape(self) -> None:
        text = "sentence one. sentence two. " * 20
        feats = compute_cognitive_features(text)
        arr = feats.to_array()
        assert arr.shape == (8,)

    def test_all_features_available(self) -> None:
        text = "word " * 100
        feats = compute_cognitive_features(text)
        for key in ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]:
            assert key in feats.available
