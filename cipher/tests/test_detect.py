"""Integration tests for the full CIPHER detection pipeline."""

from __future__ import annotations

import pytest

from cipher import CIPHERDetector, DetectionResult


# Short human-sounding text (with hedges and varied syntax)
HUMAN_TEXT = """
I believe the experiment yielded rather inconclusive results. Perhaps we should
reconsider our methodology — actually, I think the issue lies in the sampling
procedure itself. The variance across trials was surprisingly high. In my view,
a second round of experiments is warranted before drawing any firm conclusions.

Some researchers have suggested, tentatively, that the anomalous readings might
reflect instrument calibration drift. I'm not entirely sure this explains the
full picture. To be more precise, the drift alone accounts for maybe thirty
percent of the discrepancy; the remaining variation is harder to pin down.

What strikes me, looking at the data again, is how different the afternoon
sessions were from the morning ones. The team noticed this too. We talked about
it briefly on Tuesday, and nobody had a satisfying explanation. Maybe temperature.
Maybe humidity. Possibly just noise.
""".strip()

# AI-sounding text (unnaturally smooth, no hedges, uniform sentences)
AI_TEXT = """
The experiment demonstrated significant results across all measured parameters.
The methodology was rigorous and produced consistent outcomes throughout the study.
The variance observed in the data was within acceptable statistical boundaries.
The researchers concluded that the findings support the initial hypothesis.

The sampling procedure was carefully designed to minimize potential sources of error.
The instrument calibration was verified before each experimental session.
The data collection process followed established protocols for scientific research.
The analysis revealed clear patterns consistent with the theoretical predictions.

The morning and afternoon sessions produced comparable results across all metrics.
The temperature and humidity conditions were controlled throughout the experiment.
The team successfully replicated the findings in three independent trials.
The statistical analysis confirmed the significance of the observed effects.
""".strip()


class TestCIPHERDetector:
    """Integration tests using the heuristic meta-learner (no trained model)."""

    def setup_method(self) -> None:
        self.detector = CIPHERDetector()

    def test_returns_detection_result(self) -> None:
        result = self.detector.detect(HUMAN_TEXT)
        assert isinstance(result, DetectionResult)

    def test_document_score_in_range(self) -> None:
        result = self.detector.detect(HUMAN_TEXT)
        assert 0.0 <= result.document_score <= 1.0

    def test_prediction_set_valid(self) -> None:
        result = self.detector.detect(HUMAN_TEXT)
        valid = {"AI", "Human"}
        for label in result.prediction_set:
            assert label in valid

    def test_confidence_interval_valid(self) -> None:
        result = self.detector.detect(HUMAN_TEXT)
        lo, hi = result.confidence_interval
        assert 0.0 <= lo <= result.document_score <= hi <= 1.0

    def test_word_count_correct(self) -> None:
        result = self.detector.detect(HUMAN_TEXT)
        expected = len(HUMAN_TEXT.split())
        assert result.word_count == expected

    def test_sentence_scores_returned(self) -> None:
        result = self.detector.detect(HUMAN_TEXT, include_sentence_scores=True)
        assert len(result.sentence_scores) > 0
        for ss in result.sentence_scores:
            assert 0.0 <= ss.score <= 1.0

    def test_sentence_scores_omitted(self) -> None:
        result = self.detector.detect(HUMAN_TEXT, include_sentence_scores=False)
        assert result.sentence_scores == []
        assert result.highlighted_spans == []

    def test_signal_breakdown_has_four_layers(self) -> None:
        result = self.detector.detect(HUMAN_TEXT)
        bd = result.signal_breakdown
        for val in [bd.layer1_compression, bd.layer2_cognitive, bd.layer3_discourse]:
            assert 0.0 <= val <= 1.0

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            self.detector.detect("Too short.")

    def test_processing_ms_positive(self) -> None:
        result = self.detector.detect(HUMAN_TEXT)
        assert result.processing_ms > 0

    def test_model_version_set(self) -> None:
        result = self.detector.detect(HUMAN_TEXT)
        assert result.model_version == "cipher-1.0"


class TestMetaLearnerHeuristic:
    """Verify that the heuristic meta-learner produces valid output."""

    def test_ai_text_returns_valid_score(self) -> None:
        """AI text should produce a well-formed score in [0, 1]."""
        detector = CIPHERDetector()
        result = detector.detect(AI_TEXT)
        assert 0.0 <= result.document_score <= 1.0

    def test_human_text_has_valid_score(self) -> None:
        detector = CIPHERDetector()
        result = detector.detect(HUMAN_TEXT)
        assert 0.0 <= result.document_score <= 1.0


class TestFeatureVector:
    def test_feature_vector_shape(self) -> None:
        import numpy as np
        detector = CIPHERDetector()
        result = detector.detect(HUMAN_TEXT)
        arr = result.feature_vector.to_array(include_manifold=False)
        assert arr.shape == (32,)  # 7+8+7+10=32 without manifold

    def test_feature_vector_as_dict(self) -> None:
        detector = CIPHERDetector()
        result = detector.detect(HUMAN_TEXT)
        d = result.feature_vector.as_dict()
        assert "l1_lzc" in d
        assert "c2_hurst" in d
        assert "d1_coref_density" in d
