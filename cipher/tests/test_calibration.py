"""Tests for calibration modules: IsotonicCalibrator and ConformalPredictor.

Covers all branches of fit/transform, save/load, calibrate/predict, and
the ConformalResult label/coverage properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from cipher.calibration.conformal import ConformalPredictor, ConformalResult
from cipher.calibration.isotonic import IsotonicCalibrator


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------


class TestIsotonicCalibratorUnfitted:
    """Tests for IsotonicCalibrator before any call to fit()."""

    def test_unfitted_identity_passthrough(self) -> None:
        """transform() should return clipped input when not fitted."""
        cal = IsotonicCalibrator()
        assert cal.transform(0.7) == pytest.approx(0.7)

    def test_unfitted_clips_below_zero(self) -> None:
        cal = IsotonicCalibrator()
        assert cal.transform(-0.5) == pytest.approx(0.0)

    def test_unfitted_clips_above_one(self) -> None:
        cal = IsotonicCalibrator()
        assert cal.transform(1.8) == pytest.approx(1.0)

    def test_unfitted_transform_array(self) -> None:
        cal = IsotonicCalibrator()
        arr = np.array([-0.5, 0.3, 1.5])
        out = cal.transform_array(arr)
        np.testing.assert_array_almost_equal(out, [0.0, 0.3, 1.0])

    def test_fitted_flag_false_before_fit(self) -> None:
        cal = IsotonicCalibrator()
        assert not cal._fitted

    @pytest.mark.parametrize("score,expected", [
        (-0.5, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.5, 1.0),
    ])
    def test_unfitted_clips_parametrized(self, score: float, expected: float) -> None:
        """Unfitted calibrator clips to [0, 1] for various inputs."""
        cal = IsotonicCalibrator()
        assert cal.transform(score) == pytest.approx(expected)


class TestIsotonicCalibratorFitted:
    """Tests for IsotonicCalibrator after fit()."""

    def _make_fitted(self) -> IsotonicCalibrator:
        cal = IsotonicCalibrator()
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        cal.fit(scores, labels)
        return cal

    def test_fitted_flag_true_after_fit(self) -> None:
        cal = self._make_fitted()
        assert cal._fitted

    def test_transform_returns_float_in_range(self) -> None:
        cal = self._make_fitted()
        result = cal.transform(0.5)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_transform_array_returns_in_range(self) -> None:
        cal = self._make_fitted()
        out = cal.transform_array(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        assert all(0.0 <= v <= 1.0 for v in out)

    def test_fit_returns_self(self) -> None:
        cal = IsotonicCalibrator()
        result = cal.fit(np.array([0.2, 0.8]), np.array([0, 1]))
        assert result is cal

    def test_save_and_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        """Saved and loaded calibrator produces the same transform output."""
        cal = self._make_fitted()
        path = tmp_path / "iso.pkl"
        cal.save(path)
        loaded = IsotonicCalibrator.load(path)
        assert loaded._fitted
        assert cal.transform(0.4) == pytest.approx(loaded.transform(0.4))

    def test_save_creates_parent_dirs(self, tmp_path: pytest.TempPathFactory) -> None:
        cal = self._make_fitted()
        path = tmp_path / "nested" / "dir" / "iso.pkl"
        cal.save(path)
        assert path.exists()

    def test_transform_extreme_values_clipped(self) -> None:
        cal = self._make_fitted()
        assert 0.0 <= cal.transform(-10.0) <= 1.0
        assert 0.0 <= cal.transform(10.0) <= 1.0

    def test_fit_all_same_scores(self) -> None:
        """Fitting on uniform scores should not crash."""
        cal = IsotonicCalibrator()
        cal.fit(np.array([0.5, 0.5, 0.5]), np.array([0, 1, 1]))
        result = cal.transform(0.5)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# ConformalPredictor
# ---------------------------------------------------------------------------


class TestConformalPredictorUncalibrated:
    """Tests for ConformalPredictor predict() with no calibration data."""

    def test_high_score_predicts_ai(self) -> None:
        """Score >= 0.65 should yield prediction_set == ['AI']."""
        cp = ConformalPredictor()
        result = cp.predict(0.80)
        assert result.prediction_set == ["AI"]
        assert result.label == "AI"

    def test_low_score_predicts_human(self) -> None:
        """Score <= 0.35 should yield prediction_set == ['Human']."""
        cp = ConformalPredictor()
        result = cp.predict(0.20)
        assert result.prediction_set == ["Human"]
        assert result.label == "Human"

    def test_boundary_high_score(self) -> None:
        """Score exactly at 0.65 boundary → AI."""
        cp = ConformalPredictor()
        result = cp.predict(0.65)
        assert result.prediction_set == ["AI"]

    def test_boundary_low_score(self) -> None:
        """Score exactly at 0.35 boundary → Human."""
        cp = ConformalPredictor()
        result = cp.predict(0.35)
        assert result.prediction_set == ["Human"]

    def test_middle_score_is_uncertain(self) -> None:
        """Score between thresholds → both labels in prediction_set."""
        cp = ConformalPredictor()
        result = cp.predict(0.50)
        assert result.label == "Uncertain"
        assert "AI" in result.prediction_set
        assert "Human" in result.prediction_set

    def test_uncalibrated_interval_stays_in_bounds(self) -> None:
        """Confidence interval should stay within [0, 1]."""
        cp = ConformalPredictor()
        for score in [0.0, 0.1, 0.5, 0.9, 1.0]:
            result = cp.predict(score)
            lo, hi = result.interval
            assert lo >= 0.0
            assert hi <= 1.0

    def test_quantile_is_none_before_calibrate(self) -> None:
        cp = ConformalPredictor()
        assert cp._quantile is None


class TestConformalPredictorCalibrated:
    """Tests for ConformalPredictor after calibrate()."""

    def _make_calibrated(self, alpha: float = 0.10) -> ConformalPredictor:
        cp = ConformalPredictor(alpha=alpha)
        scores = np.linspace(0.1, 0.9, 20)
        labels = (scores >= 0.5).astype(int)
        cp.calibrate(scores, labels)
        return cp

    def test_calibrate_sets_quantile(self) -> None:
        cp = self._make_calibrated()
        assert cp._quantile is not None

    def test_predict_score_field(self) -> None:
        cp = self._make_calibrated()
        result = cp.predict(0.7)
        assert result.score == pytest.approx(0.7)

    def test_predict_interval_ordered(self) -> None:
        cp = self._make_calibrated()
        result = cp.predict(0.6)
        lo, hi = result.interval
        assert lo <= hi

    def test_predict_interval_in_unit_range(self) -> None:
        cp = self._make_calibrated()
        for score in [0.1, 0.5, 0.9]:
            lo, hi = cp.predict(score).interval
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0

    def test_calibrate_empty_returns_self(self) -> None:
        """Calibrating with empty arrays should not raise."""
        cp = ConformalPredictor()
        result = cp.calibrate(np.array([]), np.array([]))
        assert cp._quantile is None
        assert result is cp

    def test_calibrate_all_ai_labels(self) -> None:
        """All-AI calibration should not crash."""
        cp = ConformalPredictor(alpha=0.1)
        scores = np.array([0.8, 0.85, 0.9, 0.75, 0.7])
        labels = np.array([1, 1, 1, 1, 1])
        cp.calibrate(scores, labels)
        result = cp.predict(0.9)
        assert isinstance(result, ConformalResult)

    def test_calibrate_all_human_labels(self) -> None:
        """All-human calibration should not crash."""
        cp = ConformalPredictor(alpha=0.1)
        scores = np.array([0.1, 0.15, 0.2, 0.05])
        labels = np.array([0, 0, 0, 0])
        cp.calibrate(scores, labels)
        result = cp.predict(0.1)
        assert isinstance(result, ConformalResult)

    def test_calibrate_all_same_scores(self) -> None:
        """Uniform scores should not crash calibration."""
        cp = ConformalPredictor()
        cp.calibrate(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 0, 1, 0]))
        assert isinstance(cp.predict(0.5), ConformalResult)

    def test_save_and_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        """Loaded predictor matches original on alpha and quantile."""
        cp = self._make_calibrated(alpha=0.05)
        path = tmp_path / "conformal.npy"
        cp.save(path)
        loaded = ConformalPredictor.load(path)
        assert loaded.alpha == pytest.approx(0.05)
        assert loaded._quantile == pytest.approx(cp._quantile)

    def test_save_and_load_uncalibrated(self, tmp_path: pytest.TempPathFactory) -> None:
        """Uncalibrated predictor round-trips with _quantile == None."""
        cp = ConformalPredictor(alpha=0.1)
        path = tmp_path / "conformal.npy"
        cp.save(path)
        loaded = ConformalPredictor.load(path)
        assert loaded._quantile is None

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_coverage_property_matches_alpha(self, alpha: float) -> None:
        """coverage == 1 - alpha for every calibrated predictor."""
        cp = self._make_calibrated(alpha=alpha)
        result = cp.predict(0.7)
        assert result.coverage == pytest.approx(1.0 - alpha)

    def test_empty_prediction_set_falls_back(self) -> None:
        """An empty prediction set from calibrated predict should fall back."""
        cp = ConformalPredictor(alpha=0.05)
        cp._quantile = 0.01  # very small quantile → might produce empty set
        result = cp.predict(0.50)
        assert len(result.prediction_set) >= 1


# ---------------------------------------------------------------------------
# ConformalResult
# ---------------------------------------------------------------------------


class TestConformalResult:
    """Tests for ConformalResult label and coverage properties."""

    def test_label_ai(self) -> None:
        r = ConformalResult(score=0.9, prediction_set=["AI"], interval=(0.75, 1.0), alpha=0.05)
        assert r.label == "AI"

    def test_label_human(self) -> None:
        r = ConformalResult(score=0.1, prediction_set=["Human"], interval=(0.0, 0.25), alpha=0.05)
        assert r.label == "Human"

    def test_label_uncertain_both(self) -> None:
        r = ConformalResult(
            score=0.5, prediction_set=["AI", "Human"], interval=(0.35, 0.65), alpha=0.05
        )
        assert r.label == "Uncertain"

    def test_label_uncertain_empty_set(self) -> None:
        """An empty prediction set is treated as Uncertain."""
        r = ConformalResult(score=0.5, prediction_set=[], interval=(0.4, 0.6), alpha=0.05)
        assert r.label == "Uncertain"

    def test_coverage_property(self) -> None:
        r = ConformalResult(score=0.5, prediction_set=["AI"], interval=(0.4, 0.6), alpha=0.10)
        assert r.coverage == pytest.approx(0.90)

    @pytest.mark.parametrize("alpha,expected_coverage", [
        (0.05, 0.95),
        (0.10, 0.90),
        (0.20, 0.80),
    ])
    def test_coverage_parametrized(self, alpha: float, expected_coverage: float) -> None:
        """coverage == 1 - alpha for all alpha values."""
        r = ConformalResult(score=0.5, prediction_set=["AI"], interval=(0.4, 0.6), alpha=alpha)
        assert r.coverage == pytest.approx(expected_coverage)
