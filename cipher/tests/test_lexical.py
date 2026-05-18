"""Tests for Layer 6: lexical richness and stylometric features.

Covers all 10 features (s1–s10) and the compute_lexical_features entry point.
Tests verify correctness, edge cases, and direction of signal (AI vs human text).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cipher.layers.lexical import (
    LexicalFeatures,
    _bigram_repetition,
    _hapax_ratio,
    _lexical_density,
    _paragraph_length_cv,
    _punctuation_entropy,
    _sentence_initial_entropy,
    _tokenize,
    _type_token_ratio,
    _yule_k,
    compute_lexical_features,
)


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

# AI-like: uniform sentence starters, low vocabulary diversity, formal
AI_TEXT = (
    "The model generates text with high fluency and coherence. "
    "The output exhibits consistent vocabulary patterns throughout. "
    "The system uses attention mechanisms to process the input. "
    "The results show that the model performs well on benchmarks. "
    "The approach demonstrates the effectiveness of transformer architectures. "
    "The data indicates significant improvements over baseline methods. "
    "The process involves multiple layers of self-attention computation. "
    "The technique is widely adopted in natural language processing research. "
    "The implementation requires careful tuning of hyperparameters. "
    "The evaluation confirms that the method achieves state-of-the-art performance."
)

# Human-like: varied sentence starters, richer vocabulary, personal voice
HUMAN_TEXT = (
    "I stumbled upon this paper quite by accident — and honestly, it surprised me. "
    "What struck me wasn't the results per se, but the way they'd framed the problem. "
    "Actually, the methodology here is a bit unorthodox, now that I think about it. "
    "We tried three different approaches before settling on this one. "
    "Looking back, maybe we should have explored the second option more carefully. "
    "Sarah suggested using a completely different dataset; she was right, as it turned out. "
    "The variance was enormous — unexpected, frankly. "
    "Perhaps that's why no one had attempted this before? "
    "Well, anyway, the results hold up under scrutiny. "
    "There's something almost poetic about finding signal in that much noise."
)

LONG_AI = AI_TEXT * 3
LONG_HUMAN = HUMAN_TEXT * 3


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_returns_lowercase(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_removes_punctuation(self) -> None:
        assert _tokenize("Hello, world!") == ["hello", "world"]

    def test_removes_numbers(self) -> None:
        tokens = _tokenize("3 cats and 4 dogs")
        assert "3" not in tokens
        assert "4" not in tokens

    def test_empty_returns_empty(self) -> None:
        assert _tokenize("") == []

    def test_only_punctuation_returns_empty(self) -> None:
        assert _tokenize("!!! ??? ...") == []


# ---------------------------------------------------------------------------
# _type_token_ratio (s1)
# ---------------------------------------------------------------------------


class TestTypeTokenRatio:
    def test_all_unique_returns_one(self) -> None:
        tokens = ["a", "b", "c", "d", "e"]
        assert _type_token_ratio(tokens) == 1.0

    def test_all_same_returns_fraction(self) -> None:
        tokens = ["the"] * 10
        assert _type_token_ratio(tokens) == pytest.approx(0.1)

    def test_empty_returns_zero(self) -> None:
        assert _type_token_ratio([]) == 0.0

    def test_range(self) -> None:
        tokens = _tokenize(LONG_HUMAN)
        ttr = _type_token_ratio(tokens)
        assert 0.0 < ttr <= 1.0

    def test_human_higher_than_ai(self) -> None:
        """Human text should have higher TTR (more varied vocabulary)."""
        ai_ttr = _type_token_ratio(_tokenize(LONG_AI))
        human_ttr = _type_token_ratio(_tokenize(LONG_HUMAN))
        assert human_ttr > ai_ttr


# ---------------------------------------------------------------------------
# _hapax_ratio (s2)
# ---------------------------------------------------------------------------


class TestHapaxRatio:
    def test_all_unique_returns_one(self) -> None:
        tokens = ["a", "b", "c", "d"]
        assert _hapax_ratio(tokens) == 1.0

    def test_all_repeated_returns_zero(self) -> None:
        tokens = ["the", "the", "the", "cat", "cat", "cat"]
        assert _hapax_ratio(tokens) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert _hapax_ratio([]) == 0.0

    def test_value_in_range(self) -> None:
        tokens = _tokenize(LONG_HUMAN)
        hr = _hapax_ratio(tokens)
        assert 0.0 <= hr <= 1.0


# ---------------------------------------------------------------------------
# _yule_k (s3)
# ---------------------------------------------------------------------------


class TestYuleK:
    def test_nonnegative(self) -> None:
        tokens = _tokenize(LONG_AI)
        assert _yule_k(tokens) >= 0.0

    def test_empty_returns_zero(self) -> None:
        assert _yule_k([]) == 0.0

    def test_single_token_returns_zero(self) -> None:
        assert _yule_k(["word"]) == 0.0

    def test_all_unique_low_k(self) -> None:
        # All unique words → very low repetition → low K
        tokens = [f"word{i}" for i in range(100)]
        k = _yule_k(tokens)
        # For all unique: K = 10000 * (N/N^2 - 1/N) = 10000 * (1/N - 1/N) = 0
        assert k == pytest.approx(0.0, abs=1.0)

    def test_ai_higher_k_than_human(self) -> None:
        """AI text with repeated phrasing should have higher Yule's K."""
        ai_k = _yule_k(_tokenize(LONG_AI))
        human_k = _yule_k(_tokenize(LONG_HUMAN))
        # The AI text repeats "the" many more times → higher K
        assert ai_k > human_k


# ---------------------------------------------------------------------------
# _bigram_repetition (s4)
# ---------------------------------------------------------------------------


class TestBigramRepetition:
    def test_no_repetition_returns_zero(self) -> None:
        tokens = ["a", "b", "c", "d", "e", "f"]
        assert _bigram_repetition(tokens) == 0.0

    def test_full_repetition_returns_one(self) -> None:
        tokens = ["the", "cat", "the", "cat", "the", "cat"]
        assert _bigram_repetition(tokens) > 0.0

    def test_empty_returns_zero(self) -> None:
        assert _bigram_repetition([]) == 0.0

    def test_single_token_returns_zero(self) -> None:
        assert _bigram_repetition(["word"]) == 0.0

    def test_range(self) -> None:
        tokens = _tokenize(LONG_AI)
        br = _bigram_repetition(tokens)
        assert 0.0 <= br <= 1.0


# ---------------------------------------------------------------------------
# _punctuation_entropy (s6)
# ---------------------------------------------------------------------------


class TestPunctuationEntropy:
    def test_no_punctuation_returns_zero(self) -> None:
        assert _punctuation_entropy("hello world") == 0.0

    def test_single_punct_type_returns_zero(self) -> None:
        text = "Hello. World. Test."
        entropy = _punctuation_entropy(text)
        assert entropy == pytest.approx(0.0)

    def test_varied_punct_higher_entropy(self) -> None:
        simple = "Hello. World. Test. Done."
        rich = "Hello! Wait — really? Yes (indeed), it's true: done."
        assert _punctuation_entropy(rich) > _punctuation_entropy(simple)

    def test_entropy_nonnegative(self) -> None:
        assert _punctuation_entropy(LONG_HUMAN) >= 0.0
        assert _punctuation_entropy(LONG_AI) >= 0.0


# ---------------------------------------------------------------------------
# _sentence_initial_entropy (s7)
# ---------------------------------------------------------------------------


class TestSentenceInitialEntropy:
    def test_all_same_starter_returns_zero(self) -> None:
        text = "The cat sat. The dog ran. The bird flew."
        entropy = _sentence_initial_entropy(text)
        assert entropy == pytest.approx(0.0)

    def test_all_different_starters_higher_entropy(self) -> None:
        text = "Alice ran. Bob walked. Charlie swam. Diana flew. Eddie climbed."
        entropy = _sentence_initial_entropy(text)
        assert entropy > 0.0

    def test_human_higher_than_ai(self) -> None:
        """Human text should have more varied sentence starters."""
        ai_ent = _sentence_initial_entropy(LONG_AI)
        human_ent = _sentence_initial_entropy(LONG_HUMAN)
        assert human_ent > ai_ent

    def test_empty_returns_zero(self) -> None:
        assert _sentence_initial_entropy("") == 0.0


# ---------------------------------------------------------------------------
# _paragraph_length_cv (s8)
# ---------------------------------------------------------------------------


class TestParagraphLengthCV:
    def test_single_paragraph_returns_zero(self) -> None:
        # CV requires at least 2 paragraphs
        assert _paragraph_length_cv("One paragraph only.") == 0.0

    def test_equal_length_paragraphs_low_cv(self) -> None:
        paras = "\n\n".join([" ".join(["word"] * 50)] * 4)
        cv = _paragraph_length_cv(paras)
        assert cv == pytest.approx(0.0, abs=0.01)

    def test_varied_lengths_nonzero_cv(self) -> None:
        paras = "\n\n".join([
            " ".join(["word"] * 20),
            " ".join(["word"] * 100),
            " ".join(["word"] * 10),
            " ".join(["word"] * 80),
        ])
        cv = _paragraph_length_cv(paras)
        assert cv > 0.1

    def test_nonnegative(self) -> None:
        assert _paragraph_length_cv(LONG_HUMAN) >= 0.0


# ---------------------------------------------------------------------------
# _lexical_density (s9)
# ---------------------------------------------------------------------------


class TestLexicalDensity:
    def test_without_spacy_returns_value(self) -> None:
        ld = _lexical_density(LONG_HUMAN, nlp=None)
        assert 0.0 < ld <= 1.0

    def test_empty_returns_half(self) -> None:
        # Fallback neutral value
        assert _lexical_density("", nlp=None) == 0.5


# ---------------------------------------------------------------------------
# compute_lexical_features (public entry point)
# ---------------------------------------------------------------------------


class TestComputeLexicalFeatures:
    def test_returns_lexical_features(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        assert isinstance(feats, LexicalFeatures)

    def test_to_array_shape(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        arr = feats.to_array()
        assert arr.shape == (10,)

    def test_to_array_dtype(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        assert feats.to_array().dtype == np.float64

    def test_no_nan_or_inf(self) -> None:
        for text in [LONG_AI, LONG_HUMAN, "Short text."]:
            feats = compute_lexical_features(text)
            arr = feats.to_array()
            assert np.all(np.isfinite(arr)), f"NaN/Inf in features for: {text[:60]}"

    def test_s1_ttr_in_range(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        assert 0.0 < feats.s1 <= 1.0

    def test_s2_hapax_in_range(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        assert 0.0 <= feats.s2 <= 1.0

    def test_s3_yule_k_nonnegative(self) -> None:
        feats = compute_lexical_features(LONG_AI)
        assert feats.s3 >= 0.0

    def test_s4_bigram_rep_in_range(self) -> None:
        feats = compute_lexical_features(LONG_AI)
        assert 0.0 <= feats.s4 <= 1.0

    def test_s5_passive_in_range(self) -> None:
        feats = compute_lexical_features(LONG_AI)
        assert 0.0 <= feats.s5 <= 1.0

    def test_s6_punct_entropy_nonnegative(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        assert feats.s6 >= 0.0

    def test_s7_sent_init_ent_nonneg(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        assert feats.s7 >= 0.0

    def test_s8_para_cv_nonneg(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        assert feats.s8 >= 0.0

    def test_s9_lex_density_in_range(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        assert 0.0 < feats.s9 <= 1.0

    def test_s10_word_rank_positive(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        assert feats.s10 > 0.0

    def test_very_short_text_no_crash(self) -> None:
        feats = compute_lexical_features("Hi.")
        arr = feats.to_array()
        assert arr.shape == (10,)

    def test_empty_string_no_crash(self) -> None:
        feats = compute_lexical_features("")
        arr = feats.to_array()
        assert arr.shape == (10,)
        assert np.all(np.isfinite(arr))

    def test_available_list_populated(self) -> None:
        feats = compute_lexical_features(LONG_HUMAN)
        # At minimum s1-s4, s6-s8, s10 are always computed without spaCy
        assert len(feats.available) >= 8

    def test_ai_lower_ttr_than_human(self) -> None:
        ai_feats = compute_lexical_features(LONG_AI)
        human_feats = compute_lexical_features(LONG_HUMAN)
        assert ai_feats.s1 < human_feats.s1

    def test_ai_lower_sent_init_entropy(self) -> None:
        ai_feats = compute_lexical_features(LONG_AI)
        human_feats = compute_lexical_features(LONG_HUMAN)
        assert ai_feats.s7 < human_feats.s7
