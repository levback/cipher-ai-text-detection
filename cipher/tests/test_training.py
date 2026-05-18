"""Tests for training pipeline helpers: _load_labeled_csv, _extract_features,
and the train_meta / train_contrastive typer commands.
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from typer.testing import CliRunner

from cipher.cli import app
from cipher.training.train_meta import _extract_features, _load_labeled_csv

runner = CliRunner()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

LONG_AI = (
    "The model generated highly coherent text with uniform sentence length distributions. "
    "The output demonstrated consistent vocabulary and syntactic patterns throughout. "
    "Statistical analysis confirmed minimal variance in lexical complexity metrics. "
    "All paragraphs followed identical structural templates without deviation. "
    "The generated content exhibited smooth topic transitions at every boundary. "
    "No epistemic markers or hedging language appeared in the produced text. "
)

LONG_HUMAN = (
    "I think the results were, honestly, a bit surprising — maybe even unexpected. "
    "We hadn't anticipated such wide variance across the different trial conditions. "
    "Actually, looking back at the data now, the third trial stands out significantly. "
    "Perhaps the ambient temperature affected the readings more than we realised. "
    "It's worth noting that our calibration procedure might have introduced noise. "
    "The team debated this for a while; no consensus was reached by end of day. "
)


def _write_labeled_csv(
    path: Path,
    texts: list[str],
    labels: list[int],
    doc_ids: list[str] | None = None,
) -> None:
    fields = ["text", "label"] if doc_ids is None else ["text", "label", "doc_id"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, (text, label) in enumerate(zip(texts, labels)):
            row: dict[str, object] = {"text": text, "label": label}
            if doc_ids is not None:
                row["doc_id"] = doc_ids[i]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# _load_labeled_csv
# ---------------------------------------------------------------------------


class TestLoadLabeledCSV:
    """Tests for _load_labeled_csv helper."""

    def test_returns_texts_labels_doc_ids(self, tmp_path: Path) -> None:
        """Should return lists of texts, integer labels, and doc_ids."""
        csv_path = tmp_path / "data.csv"
        _write_labeled_csv(csv_path, [LONG_AI, LONG_HUMAN], [1, 0])
        texts, labels, doc_ids = _load_labeled_csv(csv_path)
        assert texts == [LONG_AI, LONG_HUMAN]
        assert labels == [1, 0]
        assert len(doc_ids) == 2

    def test_explicit_doc_ids_preserved(self, tmp_path: Path) -> None:
        """Explicit doc_id column should be returned verbatim."""
        csv_path = tmp_path / "data.csv"
        _write_labeled_csv(csv_path, [LONG_AI, LONG_HUMAN], [1, 0], ["docA", "docB"])
        _, _, doc_ids = _load_labeled_csv(csv_path)
        assert doc_ids == ["docA", "docB"]

    def test_auto_doc_id_when_column_missing(self, tmp_path: Path) -> None:
        """When no doc_id column, auto-generated IDs should be returned."""
        csv_path = tmp_path / "data.csv"
        _write_labeled_csv(csv_path, [LONG_AI, LONG_HUMAN], [1, 0])
        _, _, doc_ids = _load_labeled_csv(csv_path)
        assert doc_ids[0].startswith("doc_")

    def test_label_types_are_int(self, tmp_path: Path) -> None:
        """Labels should be Python ints, not strings."""
        csv_path = tmp_path / "data.csv"
        _write_labeled_csv(csv_path, [LONG_AI, LONG_HUMAN], [1, 0])
        _, labels, _ = _load_labeled_csv(csv_path)
        for label in labels:
            assert isinstance(label, int)

    def test_multiple_rows(self, tmp_path: Path) -> None:
        """Should handle multiple rows correctly."""
        csv_path = tmp_path / "data.csv"
        texts = [LONG_AI] * 3 + [LONG_HUMAN] * 3
        labels = [1, 1, 1, 0, 0, 0]
        _write_labeled_csv(csv_path, texts, labels)
        out_texts, out_labels, out_doc_ids = _load_labeled_csv(csv_path)
        assert len(out_texts) == 6
        assert len(out_labels) == 6
        assert len(out_doc_ids) == 6
        assert sum(out_labels) == 3


# ---------------------------------------------------------------------------
# _extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Tests for _extract_features with no NLP model and no encoder."""

    def test_shape_two_texts(self) -> None:
        """Two texts without nlp/encoder should yield shape (2, 32).

        32 = 7 (compression) + 8 (cognitive) + 7 (discourse) + 10 (lexical).
        The manifold feature (m1) is excluded when include_manifold=False.
        """
        X = _extract_features(
            texts=[LONG_AI, LONG_HUMAN],
            nlp=None,
            encoder=None,
            ai_corpus=None,
            human_corpus=None,
        )
        assert X.shape == (2, 32)

    def test_shape_five_texts(self) -> None:
        """Five texts should yield shape (5, 32)."""
        texts = [LONG_AI, LONG_HUMAN, LONG_AI, LONG_HUMAN, LONG_AI]
        X = _extract_features(
            texts=texts,
            nlp=None,
            encoder=None,
            ai_corpus=None,
            human_corpus=None,
        )
        assert X.shape == (5, 32)

    def test_all_finite_values(self) -> None:
        """Feature matrix should contain only finite numbers."""
        X = _extract_features(
            texts=[LONG_AI, LONG_HUMAN],
            nlp=None,
            encoder=None,
            ai_corpus=None,
            human_corpus=None,
        )
        assert np.all(np.isfinite(X)), "Feature matrix contains non-finite values"

    def test_dtype_is_float(self) -> None:
        """Feature matrix should be a float array."""
        X = _extract_features(
            texts=[LONG_AI],
            nlp=None,
            encoder=None,
            ai_corpus=None,
            human_corpus=None,
        )
        assert np.issubdtype(X.dtype, np.floating)


# ---------------------------------------------------------------------------
# train meta command (via CLI runner with mocked _extract_features)
# ---------------------------------------------------------------------------


class TestTrainMetaCommand:
    """Tests for the `cipher train meta` CLI command."""

    def _make_train_csv(self, tmp_path: Path) -> Path:
        """Create a small labelled CSV for training (5 AI + 5 human)."""
        csv_path = tmp_path / "train.csv"
        texts = [LONG_AI] * 5 + [LONG_HUMAN] * 5
        labels = [1] * 5 + [0] * 5
        _write_labeled_csv(csv_path, texts, labels)
        return csv_path

    def test_train_meta_saves_models(self, tmp_path: Path) -> None:
        """train meta should save meta_learner.pkl, isotonic.pkl, conformal.npy."""
        train_csv = self._make_train_csv(tmp_path)
        output_dir = tmp_path / "models"

        # Mock _extract_features to return deterministic pre-computed features
        fake_X = np.random.default_rng(42).random((8, 32))
        fake_X_cal = np.random.default_rng(43).random((2, 32))

        call_count = {"n": 0}

        def mock_extract(texts, nlp, encoder, ai_corpus, human_corpus):  # type: ignore[no-untyped-def]
            n = len(texts)
            call_count["n"] += 1
            rng = np.random.default_rng(call_count["n"])
            return rng.random((n, 32))

        with mock.patch("cipher.training.train_meta._extract_features", side_effect=mock_extract):
            result = runner.invoke(
                app,
                [
                    "train",
                    "meta",
                    "--data",
                    str(train_csv),
                    "--output-dir",
                    str(output_dir),
                    "--no-use-encoder",
                ],
            )

        assert result.exit_code == 0, result.output
        assert (output_dir / "meta_learner.pkl").exists()
        assert (output_dir / "isotonic.pkl").exists()
        assert (output_dir / "conformal.npy").exists()

    def test_train_meta_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["train", "meta", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# train contrastive command
# ---------------------------------------------------------------------------


class TestTrainContrastiveCommand:
    """Tests for the `cipher train contrastive` CLI command."""

    def test_exits_one_when_torch_missing(self, tmp_path: Path) -> None:
        """train contrastive should exit 1 when torch is unavailable."""
        dummy_csv = tmp_path / "data.csv"
        dummy_csv.write_text("text,label\nhello world,1\n")

        import builtins
        real_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            result = runner.invoke(
                app,
                [
                    "train",
                    "contrastive",
                    "--data",
                    str(dummy_csv),
                    "--output-dir",
                    str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 1
