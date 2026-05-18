"""Tests for the CIPHER CLI using typer.testing.CliRunner.

Covers detect, detect-file, batch, evaluate, train sub-commands, and
the --help flag for each subcommand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cipher.cli import app

runner = CliRunner()

# Text with well over 50 words for valid detection
LONG_TEXT = (
    "The experiment demonstrated significant results across all measured parameters. "
    "The methodology was rigorous and produced consistent outcomes throughout the study. "
    "The variance observed in the data was within acceptable statistical boundaries. "
    "The researchers concluded that the findings support the initial hypothesis. "
    "The sampling procedure was carefully designed to minimize potential sources of error. "
    "The instrument calibration was verified before each experimental session. "
    "The data collection process followed established protocols for scientific research. "
    "The analysis revealed clear patterns consistent with the theoretical predictions."
)

SHORT_TEXT = "Too short."


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Helper to write a CSV file from a list of dicts."""
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# detect command
# ---------------------------------------------------------------------------


class TestDetectCommand:
    """Tests for the `cipher detect` command."""

    def test_detect_valid_text_exits_zero(self) -> None:
        """Valid long text should produce exit code 0."""
        result = runner.invoke(app, ["detect", LONG_TEXT, "--no-encoder"])
        assert result.exit_code == 0, result.output

    def test_detect_short_text_exits_one(self) -> None:
        """Text below MIN_WORDS should produce exit code 1."""
        result = runner.invoke(app, ["detect", SHORT_TEXT, "--no-encoder"])
        assert result.exit_code == 1

    def test_detect_json_flag_exits_zero(self) -> None:
        """--json flag should emit JSON output and exit 0."""
        result = runner.invoke(app, ["detect", LONG_TEXT, "--no-encoder", "--json"])
        assert result.exit_code == 0, result.output

    def test_detect_json_output_has_document_score(self) -> None:
        """JSON output must contain a 'document_score' key."""
        result = runner.invoke(app, ["detect", LONG_TEXT, "--no-encoder", "--json"])
        assert result.exit_code == 0
        # Find the JSON portion of the output
        output = result.output
        # The output contains the JSON as a formatted block; extract it
        start = output.find("{")
        end = output.rfind("}") + 1
        assert start != -1, "No JSON found in output"
        data = json.loads(output[start:end])
        assert "document_score" in data

    def test_detect_json_output_valid_score_range(self) -> None:
        """document_score in JSON output should be in [0, 1]."""
        result = runner.invoke(app, ["detect", LONG_TEXT, "--no-encoder", "--json"])
        assert result.exit_code == 0
        output = result.output
        start = output.find("{")
        end = output.rfind("}") + 1
        data = json.loads(output[start:end])
        assert 0.0 <= data["document_score"] <= 1.0

    def test_detect_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["detect", "--help"])
        assert result.exit_code == 0

    def test_detect_help_contains_usage(self) -> None:
        result = runner.invoke(app, ["detect", "--help"])
        assert "Usage" in result.output or "usage" in result.output.lower()


# ---------------------------------------------------------------------------
# detect-file command
# ---------------------------------------------------------------------------


class TestDetectFileCommand:
    """Tests for the `cipher detect-file` command."""

    def test_detect_file_existing_file_exits_zero(self, tmp_path: Path) -> None:
        """Existing file with enough text should exit 0."""
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text(LONG_TEXT)
        result = runner.invoke(
            app, ["detect-file", str(txt_file), "--no-encoder"]
        )
        assert result.exit_code == 0, result.output

    def test_detect_file_nonexistent_exits_one(self, tmp_path: Path) -> None:
        """Non-existent file should exit 1."""
        missing = tmp_path / "does_not_exist.txt"
        result = runner.invoke(app, ["detect-file", str(missing), "--no-encoder"])
        assert result.exit_code == 1

    def test_detect_file_short_content_exits_one(self, tmp_path: Path) -> None:
        """File with fewer than 50 words should exit 1."""
        short_file = tmp_path / "short.txt"
        short_file.write_text(SHORT_TEXT)
        result = runner.invoke(
            app, ["detect-file", str(short_file), "--no-encoder"]
        )
        assert result.exit_code == 1

    def test_detect_file_json_output(self, tmp_path: Path) -> None:
        """--json flag on detect-file should produce valid JSON."""
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text(LONG_TEXT)
        result = runner.invoke(
            app, ["detect-file", str(txt_file), "--no-encoder", "--json"]
        )
        assert result.exit_code == 0
        output = result.output
        start = output.find("{")
        end = output.rfind("}") + 1
        assert start != -1
        data = json.loads(output[start:end])
        assert "document_score" in data

    def test_detect_file_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["detect-file", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# batch command
# ---------------------------------------------------------------------------


class TestBatchCommand:
    """Tests for the `cipher batch` command."""

    def _make_batch_csv(self, tmp_path: Path, n: int = 3) -> Path:
        """Create a small batch CSV with long texts."""
        csv_path = tmp_path / "batch_input.csv"
        rows = [{"id": str(i), "text": LONG_TEXT} for i in range(n)]
        _write_csv(csv_path, rows)
        return csv_path

    def test_batch_exits_zero(self, tmp_path: Path) -> None:
        """Batch command on valid CSV should exit 0."""
        input_csv = self._make_batch_csv(tmp_path)
        output_jsonl = tmp_path / "out.jsonl"
        result = runner.invoke(
            app,
            ["batch", str(input_csv), str(output_jsonl), "--no-encoder"],
        )
        assert result.exit_code == 0, result.output

    def test_batch_creates_output_file(self, tmp_path: Path) -> None:
        """Batch command should create the output JSONL file."""
        input_csv = self._make_batch_csv(tmp_path)
        output_jsonl = tmp_path / "out.jsonl"
        runner.invoke(
            app,
            ["batch", str(input_csv), str(output_jsonl), "--no-encoder"],
        )
        assert output_jsonl.exists()

    def test_batch_output_has_entries(self, tmp_path: Path) -> None:
        """Each input row should produce one JSONL entry."""
        n = 3
        input_csv = self._make_batch_csv(tmp_path, n=n)
        output_jsonl = tmp_path / "out.jsonl"
        runner.invoke(
            app,
            ["batch", str(input_csv), str(output_jsonl), "--no-encoder"],
        )
        lines = output_jsonl.read_text().strip().splitlines()
        assert len(lines) == n

    def test_batch_missing_input_exits_one(self, tmp_path: Path) -> None:
        """Non-existent input CSV should exit 1."""
        result = runner.invoke(
            app,
            ["batch", str(tmp_path / "missing.csv"), str(tmp_path / "out.jsonl")],
        )
        assert result.exit_code == 1

    def test_batch_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["batch", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# evaluate command
# ---------------------------------------------------------------------------


class TestEvaluateCommand:
    """Tests for the `cipher evaluate` command."""

    def _make_eval_csv(self, tmp_path: Path) -> Path:
        """Create a small labelled CSV for evaluation."""
        csv_path = tmp_path / "eval.csv"
        rows = [
            {"text": LONG_TEXT, "label": "1"},
            {"text": LONG_TEXT, "label": "0"},
            {"text": LONG_TEXT, "label": "1"},
            {"text": LONG_TEXT, "label": "0"},
        ]
        _write_csv(csv_path, rows)
        return csv_path

    def test_evaluate_exits_zero(self, tmp_path: Path) -> None:
        """Evaluate on valid labelled CSV should exit 0."""
        eval_csv = self._make_eval_csv(tmp_path)
        result = runner.invoke(
            app, ["evaluate", str(eval_csv), "--no-encoder"]
        )
        assert result.exit_code == 0, result.output

    def test_evaluate_prints_auroc(self, tmp_path: Path) -> None:
        """Output should contain 'AUROC' metric."""
        eval_csv = self._make_eval_csv(tmp_path)
        result = runner.invoke(
            app, ["evaluate", str(eval_csv), "--no-encoder"]
        )
        assert "AUROC" in result.output

    def test_evaluate_missing_file_exits_one(self, tmp_path: Path) -> None:
        """Non-existent data file should exit 1."""
        result = runner.invoke(
            app, ["evaluate", str(tmp_path / "missing.csv"), "--no-encoder"]
        )
        assert result.exit_code == 1

    def test_evaluate_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# train sub-commands
# ---------------------------------------------------------------------------


class TestTrainSubCommands:
    """Tests for the `cipher train` sub-command group."""

    def test_train_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0

    def test_train_meta_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["train", "meta", "--help"])
        assert result.exit_code == 0

    def test_train_contrastive_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["train", "contrastive", "--help"])
        assert result.exit_code == 0

    def test_train_contrastive_without_torch_exits_one(self, tmp_path: Path) -> None:
        """train contrastive should exit 1 when torch is unavailable."""
        import unittest.mock as mock

        dummy_csv = tmp_path / "data.csv"
        dummy_csv.write_text("text,label\nhello,1\n")

        # Patch the import inside the command function to simulate missing torch
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


# ---------------------------------------------------------------------------
# top-level help
# ---------------------------------------------------------------------------


class TestTopLevelHelp:
    """Tests for top-level --help."""

    def test_top_level_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_top_level_help_contains_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        # Should at least mention the main commands
        assert "detect" in result.output
