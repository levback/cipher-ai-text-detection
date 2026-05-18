"""Structural and import tests for the CIPHER package.

Verifies that all public symbols are importable, modules are structurally
sound, and there are no circular import errors.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Top-level package exports
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    """Tests for symbols exported from the cipher top-level package."""

    def test_cipher_detector_importable(self) -> None:
        """CIPHERDetector should be importable from the top-level package."""
        from cipher import CIPHERDetector  # noqa: F401

    def test_detection_result_importable(self) -> None:
        """DetectionResult should be importable from the top-level package."""
        from cipher import DetectionResult  # noqa: F401

    def test_signal_breakdown_importable(self) -> None:
        """SignalBreakdown should be importable from the top-level package."""
        from cipher import SignalBreakdown  # noqa: F401

    def test_cipher_detector_is_class(self) -> None:
        from cipher import CIPHERDetector
        assert isinstance(CIPHERDetector, type)

    def test_detection_result_is_class(self) -> None:
        from cipher import DetectionResult
        assert isinstance(DetectionResult, type)

    def test_version_attribute_exists(self) -> None:
        import cipher
        assert hasattr(cipher, "__version__")
        assert isinstance(cipher.__version__, str)


# ---------------------------------------------------------------------------
# Layer modules
# ---------------------------------------------------------------------------


class TestLayerModuleImports:
    """Tests that all layer modules are importable."""

    def test_compression_module_importable(self) -> None:
        import cipher.layers.compression  # noqa: F401

    def test_cognitive_module_importable(self) -> None:
        import cipher.layers.cognitive  # noqa: F401

    def test_discourse_module_importable(self) -> None:
        import cipher.layers.discourse  # noqa: F401

    def test_assembler_module_importable(self) -> None:
        import cipher.layers.assembler  # noqa: F401

    def test_feature_vector_importable(self) -> None:
        from cipher.layers.assembler import FeatureVector  # noqa: F401

    def test_compression_features_importable(self) -> None:
        from cipher.layers.compression import CompressionFeatures  # noqa: F401

    def test_discourse_features_importable(self) -> None:
        from cipher.layers.discourse import DiscourseFeatures  # noqa: F401


# ---------------------------------------------------------------------------
# Model modules
# ---------------------------------------------------------------------------


class TestModelModuleImports:
    """Tests that model modules are importable."""

    def test_meta_learner_importable(self) -> None:
        from cipher.models.meta_learner import MetaLearner  # noqa: F401

    def test_contrastive_encoder_module_importable(self) -> None:
        import cipher.models.contrastive_encoder  # noqa: F401

    def test_manifold_scorer_importable(self) -> None:
        from cipher.models.contrastive_encoder import ManifoldScorer  # noqa: F401

    def test_torch_available_flag_exists(self) -> None:
        from cipher.models.contrastive_encoder import _TORCH_AVAILABLE
        assert isinstance(_TORCH_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# Calibration modules
# ---------------------------------------------------------------------------


class TestCalibrationModuleImports:
    """Tests that calibration modules are importable."""

    def test_isotonic_module_importable(self) -> None:
        import cipher.calibration.isotonic  # noqa: F401

    def test_conformal_module_importable(self) -> None:
        import cipher.calibration.conformal  # noqa: F401

    def test_isotonic_calibrator_importable(self) -> None:
        from cipher.calibration.isotonic import IsotonicCalibrator  # noqa: F401

    def test_conformal_predictor_importable(self) -> None:
        from cipher.calibration.conformal import ConformalPredictor  # noqa: F401

    def test_conformal_result_importable(self) -> None:
        from cipher.calibration.conformal import ConformalResult  # noqa: F401


# ---------------------------------------------------------------------------
# Output module
# ---------------------------------------------------------------------------


class TestOutputModuleImports:
    """Tests that output/highlighter module is importable."""

    def test_highlighter_module_importable(self) -> None:
        import cipher.output.highlighter  # noqa: F401

    def test_sentence_score_importable(self) -> None:
        from cipher.output.highlighter import SentenceScore  # noqa: F401

    def test_highlighted_span_importable(self) -> None:
        from cipher.output.highlighter import HighlightedSpan  # noqa: F401

    def test_compute_sentence_scores_importable(self) -> None:
        from cipher.output.highlighter import compute_sentence_scores  # noqa: F401


# ---------------------------------------------------------------------------
# CLI module
# ---------------------------------------------------------------------------


class TestCLIModuleImports:
    """Tests that the CLI module is importable and app is a Typer instance."""

    def test_cli_module_importable(self) -> None:
        import cipher.cli  # noqa: F401

    def test_app_is_typer_instance(self) -> None:
        import typer
        from cipher.cli import app
        assert isinstance(app, typer.Typer)

    def test_train_app_is_typer_instance(self) -> None:
        import typer
        from cipher.training.train_meta import train_app
        assert isinstance(train_app, typer.Typer)


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------


class TestFeatureNames:
    """Tests for the FEATURE_NAMES tuple on FeatureVector."""

    def test_feature_names_has_33_elements(self) -> None:
        """FEATURE_NAMES should have exactly 33 canonical feature names.

        Breakdown: 7 (l1-l7) + 8 (c1-c8) + 7 (d1-d7) + 1 (m1) + 10 (s1-s10) = 33.
        """
        from cipher.layers.assembler import FeatureVector
        assert len(FeatureVector.FEATURE_NAMES) == 33

    def test_feature_names_is_tuple(self) -> None:
        from cipher.layers.assembler import FeatureVector
        assert isinstance(FeatureVector.FEATURE_NAMES, tuple)

    def test_feature_names_all_strings(self) -> None:
        from cipher.layers.assembler import FeatureVector
        for name in FeatureVector.FEATURE_NAMES:
            assert isinstance(name, str)

    def test_feature_names_starts_with_l1(self) -> None:
        from cipher.layers.assembler import FeatureVector
        assert FeatureVector.FEATURE_NAMES[0] == "l1_lzc"

    def test_feature_names_ends_with_s10(self) -> None:
        from cipher.layers.assembler import FeatureVector
        assert FeatureVector.FEATURE_NAMES[-1] == "s10_word_rank"

    def test_feature_names_contains_m1(self) -> None:
        from cipher.layers.assembler import FeatureVector
        assert "m1_manifold" in FeatureVector.FEATURE_NAMES

    def test_feature_names_contains_lexical_features(self) -> None:
        from cipher.layers.assembler import FeatureVector
        lexical = [n for n in FeatureVector.FEATURE_NAMES if n.startswith("s")]
        assert len(lexical) == 10


# ---------------------------------------------------------------------------
# MetaLearner interface
# ---------------------------------------------------------------------------


class TestMetaLearnerInterface:
    """Tests that MetaLearner exposes the expected public interface."""

    def test_has_fit_method(self) -> None:
        from cipher.models.meta_learner import MetaLearner
        assert callable(getattr(MetaLearner, "fit", None))

    def test_has_predict_proba_method(self) -> None:
        from cipher.models.meta_learner import MetaLearner
        assert callable(getattr(MetaLearner, "predict_proba", None))

    def test_has_save_method(self) -> None:
        from cipher.models.meta_learner import MetaLearner
        assert callable(getattr(MetaLearner, "save", None))

    def test_has_load_classmethod(self) -> None:
        from cipher.models.meta_learner import MetaLearner
        assert callable(getattr(MetaLearner, "load", None))

    def test_has_feature_importances_method(self) -> None:
        from cipher.models.meta_learner import MetaLearner
        assert callable(getattr(MetaLearner, "feature_importances", None))

    def test_has_mode_property(self) -> None:
        from cipher.models.meta_learner import MetaLearner
        ml = MetaLearner()
        assert hasattr(ml, "mode")


# ---------------------------------------------------------------------------
# No circular imports
# ---------------------------------------------------------------------------


class TestNoCircularImports:
    """Smoke-tests that importing the package does not cause circular imports."""

    def test_import_cipher_no_error(self) -> None:
        """Importing the cipher package should not raise any ImportError."""
        import cipher  # noqa: F401

    def test_import_detect_no_error(self) -> None:
        import cipher.detect  # noqa: F401

    def test_import_all_layers_no_error(self) -> None:
        import cipher.layers.assembler  # noqa: F401
        import cipher.layers.cognitive  # noqa: F401
        import cipher.layers.compression  # noqa: F401
        import cipher.layers.discourse  # noqa: F401

    def test_import_all_models_no_error(self) -> None:
        import cipher.models.contrastive_encoder  # noqa: F401
        import cipher.models.meta_learner  # noqa: F401

    def test_import_all_calibration_no_error(self) -> None:
        import cipher.calibration.conformal  # noqa: F401
        import cipher.calibration.isotonic  # noqa: F401
