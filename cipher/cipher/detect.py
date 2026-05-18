"""Main detection pipeline.

CIPHERDetector orchestrates all six layers and returns a DetectionResult.

Quick start
-----------
>>> from cipher import CIPHERDetector
>>> detector = CIPHERDetector()
>>> result = detector.detect("The quick brown fox...")
>>> print(result.document_score)
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cipher.calibration.conformal import ConformalPredictor, ConformalResult
from cipher.calibration.isotonic import IsotonicCalibrator
from cipher.layers.assembler import FeatureVector
from cipher.layers.cognitive import compute_cognitive_features
from cipher.layers.compression import compute_compression_features
from cipher.layers.discourse import compute_discourse_features
from cipher.layers.lexical import compute_lexical_features
from cipher.models.meta_learner import MetaLearner
from cipher.output.highlighter import (
    HighlightedSpan,
    SentenceScore,
    compute_sentence_scores,
    extract_highlighted_spans,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SignalBreakdown:
    layer1_compression: float
    layer2_cognitive: float
    layer3_discourse: float
    layer4_manifold: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "layer1_compression": self.layer1_compression,
            "layer2_cognitive": self.layer2_cognitive,
            "layer3_discourse": self.layer3_discourse,
            "layer4_manifold": self.layer4_manifold,
        }


@dataclass
class DetectionResult:
    document_score: float
    prediction_set: list[str]
    confidence_interval: tuple[float, float]
    coverage: float
    sentence_scores: list[SentenceScore]
    highlighted_spans: list[HighlightedSpan]
    signal_breakdown: SignalBreakdown
    feature_vector: FeatureVector
    word_count: int
    processing_ms: float
    model_version: str = "cipher-1.0"

    @property
    def label(self) -> str:
        if self.prediction_set == ["AI"]:
            return "AI"
        if self.prediction_set == ["Human"]:
            return "Human"
        return "Uncertain"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CIPHERDetector:
    """Full CIPHER detection pipeline.

    Parameters
    ----------
    model_dir:
        Directory containing trained model weights.  When None, the detector
        operates in heuristic mode (no training required, lower accuracy).
    nlp:
        Pre-loaded spaCy language model.  When None, discourse topology
        features fall back to a reduced feature set.
    encoder:
        Pre-loaded sentence-transformers model for semantic coherence.
        When None, a bag-of-words cosine fallback is used.
    ai_corpus / human_corpus:
        Reference texts for NCD features (Layers 1 l2, l3).
    coverage:
        Target conformal coverage level (0.90–0.99).
    highlight_threshold:
        Sentence score threshold for highlighting (default 0.60).
    """

    MIN_WORDS = 50
    MODEL_VERSION = "cipher-1.0"

    def __init__(
        self,
        model_dir: Path | None = None,
        nlp: Any | None = None,
        encoder: Any | None = None,
        ai_corpus: str | None = None,
        human_corpus: str | None = None,
        coverage: float = 0.95,
        highlight_threshold: float = 0.60,
    ) -> None:
        self._nlp = nlp
        self._encoder = encoder
        self._ai_corpus = ai_corpus
        self._human_corpus = human_corpus
        self._coverage = coverage
        self._highlight_threshold = highlight_threshold

        # Load models
        self._meta = MetaLearner()
        self._calibrator = IsotonicCalibrator()
        self._conformal = ConformalPredictor(alpha=1.0 - coverage)
        self._manifold_encoder: Any | None = None

        if model_dir and model_dir.exists():
            self._load_models(model_dir)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self, model_dir: Path) -> None:
        meta_path = model_dir / "meta_learner.pkl"
        if meta_path.exists():
            self._meta = MetaLearner.load(meta_path)

        cal_path = model_dir / "isotonic.pkl"
        if cal_path.exists():
            self._calibrator = IsotonicCalibrator.load(cal_path)

        conf_path = model_dir / "conformal.npy"
        if conf_path.exists():
            self._conformal = ConformalPredictor.load(conf_path)

        # Optional neural manifold encoder
        try:
            from cipher.models.contrastive_encoder import ManifoldEncoder

            encoder_weights = model_dir / "contrastive_encoder.pt"
            scorer_weights = model_dir / "manifold_scorer.pkl"
            if encoder_weights.exists() and scorer_weights.exists():
                self._manifold_encoder = ManifoldEncoder(
                    model_path=encoder_weights,
                )
                self._manifold_encoder.load_scorer(scorer_weights)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        if self._nlp is not None:
            doc = self._nlp(text[:50_000])
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        # Regex fallback
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    # ------------------------------------------------------------------
    # Per-layer partial scores (for signal breakdown)
    # ------------------------------------------------------------------

    @staticmethod
    def _partial_score(vec: np.ndarray, indices: list[int]) -> float:
        """Sigmoid of mean of z-scored selected features (rough partial score)."""
        sub = vec[indices]
        if sub.std() < 1e-8:
            return 0.5
        s = float(sub.mean())
        return float(1.0 / (1.0 + np.exp(-s)))

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def detect(
        self,
        text: str,
        include_sentence_scores: bool = True,
    ) -> DetectionResult:
        start = time.perf_counter()

        words = text.split()
        if len(words) < self.MIN_WORDS:
            raise ValueError(
                f"Text too short: {len(words)} words. Minimum is {self.MIN_WORDS}."
            )

        sentences = self._split_sentences(text)

        # Layer 1
        comp_feats = compute_compression_features(
            text,
            ai_corpus=self._ai_corpus,
            human_corpus=self._human_corpus,
        )

        # Layer 2
        cog_feats = compute_cognitive_features(text, sentences, self._encoder)

        # Layer 3
        disc_feats = compute_discourse_features(text, self._nlp)

        # Layer 6: lexical richness
        lex_feats = compute_lexical_features(text, self._nlp)

        # Layer 4 (optional)
        manifold_score: float | None = None
        if self._manifold_encoder is not None:
            manifold_score = self._manifold_encoder.manifold_score(text)

        fv = FeatureVector(
            compression=comp_feats,
            cognitive=cog_feats,
            discourse=disc_feats,
            lexical=lex_feats,
            manifold_score=manifold_score,
        )

        # Layer 5: meta-learner
        include_manifold = manifold_score is not None
        raw_vec = fv.to_array(include_manifold=include_manifold)
        raw_score = self._meta.predict_proba(raw_vec)
        # Skip isotonic calibration when a trained sklearn model is loaded —
        # sklearn's predict_proba already applies Platt scaling.  The isotonic
        # calibrator is only beneficial on top of the heuristic meta-learner.
        if self._meta.mode == "trained":
            calibrated_score = float(np.clip(raw_score, 0.0, 1.0))
        else:
            calibrated_score = self._calibrator.transform(raw_score)

        # Conformal prediction
        conf_result: ConformalResult = self._conformal.predict(calibrated_score)

        # Build signal breakdown using raw feature directions
        full_vec = fv.to_array(include_manifold=False)
        breakdown = SignalBreakdown(
            layer1_compression=self._partial_score(full_vec, list(range(7))),
            layer2_cognitive=self._partial_score(full_vec, list(range(7, 15))),
            layer3_discourse=self._partial_score(full_vec, list(range(15, 22))),
            layer4_manifold=manifold_score,
        )

        # Layer 6: sentence attribution
        sentence_scores: list[SentenceScore] = []
        highlighted_spans: list[HighlightedSpan] = []
        if include_sentence_scores:
            sentence_scores = compute_sentence_scores(
                text=text,
                sentences=sentences,
                document_score=calibrated_score,
                encoder=self._encoder,
                highlight_threshold=self._highlight_threshold,
            )
            highlighted_spans = extract_highlighted_spans(text, sentence_scores)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return DetectionResult(
            document_score=calibrated_score,
            prediction_set=conf_result.prediction_set,
            confidence_interval=conf_result.interval,
            coverage=conf_result.coverage,
            sentence_scores=sentence_scores,
            highlighted_spans=highlighted_spans,
            signal_breakdown=breakdown,
            feature_vector=fv,
            word_count=len(words),
            processing_ms=elapsed_ms,
            model_version=self.MODEL_VERSION,
        )
