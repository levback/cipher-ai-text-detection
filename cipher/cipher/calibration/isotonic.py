"""Isotonic regression calibration for raw meta-learner scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class IsotonicCalibrator:
    """Post-hoc monotone calibration via isotonic regression.

    Falls back to a no-op identity if scikit-learn is unavailable.
    """

    def __init__(self) -> None:
        self._iso: Any | None = None
        self._fitted = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        try:
            from sklearn.isotonic import IsotonicRegression

            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(raw_scores, labels)
            self._fitted = True
        except ImportError:
            pass
        return self

    def transform(self, score: float) -> float:
        if not self._fitted or self._iso is None:
            return float(np.clip(score, 0.0, 1.0))
        result = self._iso.predict([score])
        return float(np.clip(result[0], 0.0, 1.0))

    def transform_array(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted or self._iso is None:
            return np.clip(scores, 0.0, 1.0)
        return np.clip(self._iso.predict(scores), 0.0, 1.0)

    def save(self, path: Path) -> None:
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"iso": self._iso, "fitted": self._fitted}, fh)

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        import pickle

        instance = cls()
        with path.open("rb") as fh:
            data = pickle.load(fh)  # noqa: S301
        instance._iso = data["iso"]
        instance._fitted = data["fitted"]
        return instance
