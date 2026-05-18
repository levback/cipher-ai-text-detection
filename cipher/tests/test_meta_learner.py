"""Tests for MetaLearner: heuristic and trained modes, persistence, and
NaN/inf robustness.
"""

from __future__ import annotations

import numpy as np
import pytest

from cipher.models.meta_learner import MetaLearner

# Deterministic synthetic training data (20 samples, 32 features)
# 32 = 7 (l) + 8 (c) + 7 (d) + 10 (s) — excludes the m1 manifold feature
_RNG = np.random.default_rng(0)
SYNTHETIC_X = _RNG.random((20, 32))
SYNTHETIC_Y = np.array([1] * 10 + [0] * 10)


# ---------------------------------------------------------------------------
# Heuristic mode
# ---------------------------------------------------------------------------


class TestMetaLearnerHeuristic:
    """Tests for heuristic mode — no training data required."""

    def test_mode_is_heuristic_before_fit(self) -> None:
        """mode property should return 'heuristic' on a fresh MetaLearner."""
        ml = MetaLearner()
        assert ml.mode == "heuristic"

    def test_predict_proba_32_dim_in_range(self) -> None:
        """32-dim feature vector should produce a valid probability."""
        ml = MetaLearner()
        score = ml.predict_proba(np.zeros(32))
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_proba_33_dim_in_range(self) -> None:
        """33-dim feature vector (with manifold) should still work."""
        ml = MetaLearner()
        score = ml.predict_proba(np.zeros(33))
        assert 0.0 <= score <= 1.0

    def test_predict_proba_random_inputs(self) -> None:
        """All random feature vectors should yield scores in [0, 1]."""
        rng = np.random.default_rng(42)
        ml = MetaLearner()
        for _ in range(20):
            x = rng.random(32)
            score = ml.predict_proba(x)
            assert 0.0 <= score <= 1.0

    def test_feature_importances_returns_none_heuristic(self) -> None:
        """feature_importances() is None before training."""
        ml = MetaLearner()
        assert ml.feature_importances() is None

    @pytest.mark.parametrize("dim", [32, 33])
    def test_predict_proba_dims(self, dim: int) -> None:
        """predict_proba handles both 32 and 33-dimensional inputs."""
        ml = MetaLearner()
        score = ml.predict_proba(np.ones(dim) * 0.5)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Trained mode
# ---------------------------------------------------------------------------


class TestMetaLearnerTrained:
    """Tests for trained mode using synthetic data."""

    def _fit(self) -> MetaLearner:
        ml = MetaLearner()
        ml.fit(SYNTHETIC_X.copy(), SYNTHETIC_Y.copy())
        return ml

    def test_mode_is_trained_after_fit(self) -> None:
        """mode should switch to 'trained' after fit()."""
        ml = self._fit()
        assert ml.mode == "trained"

    def test_fit_returns_self(self) -> None:
        """fit() should return the MetaLearner instance for chaining."""
        ml = MetaLearner()
        result = ml.fit(SYNTHETIC_X.copy(), SYNTHETIC_Y.copy())
        assert result is ml

    def test_predict_proba_after_fit_in_range(self) -> None:
        """predict_proba() should remain in [0, 1] after training."""
        ml = self._fit()
        rng = np.random.default_rng(1)
        x = rng.random(32)
        score = ml.predict_proba(x)
        assert 0.0 <= score <= 1.0

    def test_predict_proba_returns_float(self) -> None:
        ml = self._fit()
        score = ml.predict_proba(np.ones(32) * 0.5)
        assert isinstance(score, float)

    def test_feature_importances_after_fit_not_none(self) -> None:
        """feature_importances() should return a dict after training."""
        ml = self._fit()
        importances = ml.feature_importances()
        assert importances is not None

    def test_feature_importances_length(self) -> None:
        """Importances dict should have one entry per feature (32 without manifold, 33 with)."""
        ml = self._fit()
        importances = ml.feature_importances()
        assert importances is not None
        assert len(importances) == 33

    def test_feature_importances_values_are_floats(self) -> None:
        ml = self._fit()
        importances = ml.feature_importances()
        assert importances is not None
        for val in importances.values():
            assert isinstance(val, float)

    def test_save_load_roundtrip_mode(self, tmp_path: pytest.TempPathFactory) -> None:
        """Loaded MetaLearner preserves mode='trained'."""
        ml = self._fit()
        path = tmp_path / "meta.pkl"
        ml.save(path)
        loaded = MetaLearner.load(path)
        assert loaded.mode == "trained"

    def test_save_load_roundtrip_prediction(self, tmp_path: pytest.TempPathFactory) -> None:
        """Loaded MetaLearner produces the same prediction as the original."""
        ml = self._fit()
        path = tmp_path / "meta.pkl"
        ml.save(path)
        loaded = MetaLearner.load(path)
        x = np.random.default_rng(2).random(32)
        assert ml.predict_proba(x) == pytest.approx(loaded.predict_proba(x))

    def test_save_creates_parent_directories(self, tmp_path: pytest.TempPathFactory) -> None:
        ml = self._fit()
        path = tmp_path / "nested" / "models" / "meta.pkl"
        ml.save(path)
        assert path.exists()

    def test_fit_with_nan_values_in_X(self) -> None:
        """NaN values in X should be imputed and not crash fit."""
        X = SYNTHETIC_X.copy()
        X[0, 0] = np.nan
        X[3, 10] = np.nan
        ml = MetaLearner()
        ml.fit(X, SYNTHETIC_Y.copy())
        assert ml.mode == "trained"

    def test_fit_with_inf_values_in_X(self) -> None:
        """Inf values in X should be imputed and not crash fit."""
        X = SYNTHETIC_X.copy()
        X[1, 5] = np.inf
        X[2, 15] = -np.inf
        ml = MetaLearner()
        ml.fit(X, SYNTHETIC_Y.copy())
        score = ml.predict_proba(np.ones(32))
        assert 0.0 <= score <= 1.0

    def test_predict_proba_with_nan_input(self) -> None:
        """NaN in feature vector should be imputed to neutral, not crash."""
        ml = self._fit()
        x = np.ones(32) * 0.5
        x[0] = np.nan
        score = ml.predict_proba(x)
        assert 0.0 <= score <= 1.0

    def test_predict_proba_with_inf_input(self) -> None:
        """Inf in feature vector should be handled gracefully."""
        ml = self._fit()
        x = np.ones(32) * 0.5
        x[2] = np.inf
        score = ml.predict_proba(x)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Heuristic save / load
# ---------------------------------------------------------------------------


class TestMetaLearnerPersistenceHeuristic:
    """Tests for persisting the heuristic (untrained) MetaLearner."""

    def test_save_and_load_heuristic_mode(self, tmp_path: pytest.TempPathFactory) -> None:
        """Heuristic MetaLearner should round-trip as 'heuristic'."""
        ml = MetaLearner()
        path = tmp_path / "meta_heuristic.pkl"
        ml.save(path)
        loaded = MetaLearner.load(path)
        assert loaded.mode == "heuristic"
        assert loaded.feature_importances() is None
