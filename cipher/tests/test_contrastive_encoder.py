"""Tests for the contrastive encoder module (Layer 4).

Tests cover the _TORCH_AVAILABLE flag, ManifoldScorer fit/score/save/load,
and the conditional availability of torch-dependent classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cipher.models.contrastive_encoder import ManifoldScorer, _TORCH_AVAILABLE


# ---------------------------------------------------------------------------
# Torch-dependent class availability
# ---------------------------------------------------------------------------


class TestTorchAvailability:
    """Tests confirming that torch-dependent classes are conditionally defined."""

    def test_torch_available_flag_is_bool(self) -> None:
        """_TORCH_AVAILABLE should be a boolean."""
        assert isinstance(_TORCH_AVAILABLE, bool)

    def test_contrastive_encoder_importable_iff_torch_available(self) -> None:
        """ContrastiveEncoder should only be importable when torch is present."""
        if _TORCH_AVAILABLE:
            from cipher.models.contrastive_encoder import ContrastiveEncoder  # noqa: F401
        else:
            with pytest.raises(ImportError):
                from cipher.models.contrastive_encoder import ContrastiveEncoder  # type: ignore[assignment] # noqa: F401

    def test_projection_head_importable_iff_torch_available(self) -> None:
        """ProjectionHead should only be importable when torch is present."""
        if _TORCH_AVAILABLE:
            from cipher.models.contrastive_encoder import ProjectionHead  # noqa: F401
        else:
            with pytest.raises(ImportError):
                from cipher.models.contrastive_encoder import ProjectionHead  # type: ignore[assignment] # noqa: F401

    def test_triplet_loss_importable_iff_torch_available(self) -> None:
        """TripletLoss should only be importable when torch is present."""
        if _TORCH_AVAILABLE:
            from cipher.models.contrastive_encoder import TripletLoss  # noqa: F401
        else:
            with pytest.raises(ImportError):
                from cipher.models.contrastive_encoder import TripletLoss  # type: ignore[assignment] # noqa: F401


# ---------------------------------------------------------------------------
# ManifoldScorer (does not require torch)
# ---------------------------------------------------------------------------


class TestManifoldScorerUnfitted:
    """Tests for ManifoldScorer before fit() is called."""

    def test_score_before_fit_returns_half(self) -> None:
        """score() should return 0.5 when GMMs are not fitted."""
        scorer = ManifoldScorer()
        embedding = np.random.default_rng(0).random(256)
        result = scorer.score(embedding)
        assert result == pytest.approx(0.5)

    def test_gmm_attributes_none_before_fit(self) -> None:
        """_gmm_ai and _gmm_human should be None before fitting."""
        scorer = ManifoldScorer()
        assert scorer._gmm_ai is None
        assert scorer._gmm_human is None


class TestManifoldScorerFitted:
    """Tests for ManifoldScorer after fit() with synthetic embeddings."""

    DIM = 16  # small embedding dimension for speed

    def _make_fitted(self, n_components: int = 2) -> ManifoldScorer:
        rng = np.random.default_rng(42)
        ai_emb = rng.random((20, self.DIM))
        human_emb = rng.random((20, self.DIM)) + 0.5  # shift to make distinguishable
        scorer = ManifoldScorer(n_components=n_components)
        scorer.fit(ai_emb, human_emb)
        return scorer

    def test_fit_returns_self(self) -> None:
        """fit() should return the ManifoldScorer instance."""
        rng = np.random.default_rng(0)
        ai_emb = rng.random((10, self.DIM))
        human_emb = rng.random((10, self.DIM))
        scorer = ManifoldScorer(n_components=2)
        result = scorer.fit(ai_emb, human_emb)
        assert result is scorer

    def test_gmm_attributes_set_after_fit(self) -> None:
        """_gmm_ai and _gmm_human should be set after fit()."""
        scorer = self._make_fitted()
        assert scorer._gmm_ai is not None
        assert scorer._gmm_human is not None

    def test_score_returns_float(self) -> None:
        """score() should return a Python float."""
        scorer = self._make_fitted()
        embedding = np.random.default_rng(1).random(self.DIM)
        result = scorer.score(embedding)
        assert isinstance(result, float)

    def test_score_in_unit_range(self) -> None:
        """score() result should be in [0, 1]."""
        scorer = self._make_fitted()
        rng = np.random.default_rng(10)
        for _ in range(10):
            embedding = rng.random(self.DIM)
            result = scorer.score(embedding)
            assert 0.0 <= result <= 1.0

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """save() / load() should preserve GMM state."""
        scorer = self._make_fitted()
        path = tmp_path / "manifold.pkl"
        scorer.save(path)
        loaded = ManifoldScorer.load(path)
        assert loaded._gmm_ai is not None
        assert loaded._gmm_human is not None
        embedding = np.random.default_rng(5).random(self.DIM)
        assert scorer.score(embedding) == pytest.approx(loaded.score(embedding))

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save() should create parent directories if they don't exist."""
        scorer = self._make_fitted()
        path = tmp_path / "nested" / "dir" / "manifold.pkl"
        scorer.save(path)
        assert path.exists()

    def test_n_components_capped_to_sample_size(self) -> None:
        """n_components is capped to the number of embeddings."""
        rng = np.random.default_rng(0)
        ai_emb = rng.random((3, self.DIM))
        human_emb = rng.random((3, self.DIM))
        scorer = ManifoldScorer(n_components=16)  # more than samples
        scorer.fit(ai_emb, human_emb)
        # Should fit without error
        result = scorer.score(rng.random(self.DIM))
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("dim", [8, 16, 32])
    def test_score_various_embedding_dims(self, dim: int) -> None:
        """ManifoldScorer should work with different embedding dimensions."""
        rng = np.random.default_rng(99)
        ai_emb = rng.random((10, dim))
        human_emb = rng.random((10, dim)) + 0.5
        scorer = ManifoldScorer(n_components=2)
        scorer.fit(ai_emb, human_emb)
        result = scorer.score(rng.random(dim))
        assert 0.0 <= result <= 1.0
