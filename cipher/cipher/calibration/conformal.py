"""Conformal prediction: split-conformal prediction sets for AI detection.

Provides coverage-guaranteed prediction sets at a specified miscoverage level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ConformalPredictor:
    """Split-conformal predictor using hinge nonconformity scores.

    Usage
    -----
    # Calibrate on a held-out set of (score, label) pairs
    cp = ConformalPredictor()
    cp.calibrate(scores, labels)

    # At inference
    result = cp.predict(0.82)
    # result.prediction_set → ["AI"]
    # result.interval → (0.73, 0.91)
    """

    alpha: float = 0.05  # target miscoverage level
    _quantile: float | None = None

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> "ConformalPredictor":
        """Fit the conformal threshold on calibration (score, label) pairs.

        Parameters
        ----------
        scores:
            P(AI | x) ∈ [0, 1] for each calibration document.
        labels:
            Binary labels — 1 = AI-generated, 0 = human.
        """
        n = len(scores)
        if n == 0:
            return self

        # Nonconformity score: for true class 1 (AI), s = 1 - score.
        # For true class 0 (human), s = score.
        nonconf = np.where(labels == 1, 1.0 - scores, scores)

        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = float(np.clip(level, 0.0, 1.0))
        self._quantile = float(np.quantile(nonconf, level))
        return self

    def predict(self, score: float) -> "ConformalResult":
        """Return a prediction set and confidence interval for *score*."""
        prediction_set: list[str]
        interval: tuple[float, float]

        if self._quantile is None:
            # Uncalibrated — use fixed thresholds
            if score >= 0.65:
                prediction_set = ["AI"]
            elif score <= 0.35:
                prediction_set = ["Human"]
            else:
                prediction_set = ["AI", "Human"]
            margin = 0.15
            interval = (max(0.0, score - margin), min(1.0, score + margin))
        else:
            q = self._quantile
            prediction_set = []
            # Include AI if nonconformity for AI class ≤ quantile
            if 1.0 - score <= q:
                prediction_set.append("AI")
            # Include Human if nonconformity for Human class ≤ quantile
            if score <= q:
                prediction_set.append("Human")
            if not prediction_set:
                prediction_set = ["AI", "Human"]  # empty set → full set as fallback

            lo = float(np.clip(score - q, 0.0, 1.0))
            hi = float(np.clip(score + q, 0.0, 1.0))
            interval = (lo, hi)

        return ConformalResult(
            score=score,
            prediction_set=prediction_set,
            interval=interval,
            alpha=self.alpha,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array([self._quantile or np.nan, self.alpha]))

    @classmethod
    def load(cls, path: Path) -> "ConformalPredictor":
        data = np.load(path)
        instance = cls(alpha=float(data[1]))
        q = float(data[0])
        instance._quantile = None if np.isnan(q) else q
        return instance


@dataclass
class ConformalResult:
    score: float
    prediction_set: list[str]
    interval: tuple[float, float]
    alpha: float

    @property
    def coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def label(self) -> str:
        if self.prediction_set == ["AI"]:
            return "AI"
        if self.prediction_set == ["Human"]:
            return "Human"
        return "Uncertain"
