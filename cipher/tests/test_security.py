"""Security and robustness tests for the CIPHER detection pipeline.

Verifies that adversarial or edge-case inputs are handled safely without
crashes, injection, or path traversal vulnerabilities.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from cipher.cli import app
from cipher.detect import CIPHERDetector

runner = CliRunner()

# A valid long text (≥50 words) used as baseline
VALID_LONG_TEXT = (
    "The experiment demonstrated significant results across all measured parameters. "
    "The methodology was rigorous and produced consistent outcomes throughout the study. "
    "The variance observed in the data was within acceptable statistical boundaries. "
    "The researchers concluded that the findings support the initial hypothesis. "
    "The sampling procedure was carefully designed to minimize potential sources of error. "
    "The instrument calibration was verified before each experimental session. "
    "The data collection process followed established protocols for scientific research. "
    "The analysis revealed clear patterns consistent with the theoretical predictions."
)


@pytest.fixture()
def detector() -> CIPHERDetector:
    """Fresh CIPHERDetector in heuristic mode (no heavy models)."""
    return CIPHERDetector()


# ---------------------------------------------------------------------------
# Text too short / whitespace
# ---------------------------------------------------------------------------


class TestTextTooShort:
    """Tests that short or whitespace inputs are rejected safely."""

    def test_pure_whitespace_raises_value_error(self, detector: CIPHERDetector) -> None:
        """Pure whitespace has zero meaningful words → ValueError."""
        with pytest.raises(ValueError):
            detector.detect("     \t\n   ")

    def test_single_word_raises_value_error(self, detector: CIPHERDetector) -> None:
        """Single word is below MIN_WORDS threshold."""
        with pytest.raises(ValueError):
            detector.detect("hello")

    def test_empty_string_raises_value_error(self, detector: CIPHERDetector) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            detector.detect("")

    def test_cli_short_text_exits_one(self) -> None:
        """CLI should exit 1 for text below MIN_WORDS."""
        result = runner.invoke(app, ["detect", "Too short.", "--no-encoder"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Null bytes
# ---------------------------------------------------------------------------


class TestNullBytes:
    """Tests for null byte handling in input text."""

    def test_null_bytes_in_short_text_raises_or_handles(
        self, detector: CIPHERDetector
    ) -> None:
        """Null bytes in a short text should not crash — raises ValueError (too short)."""
        text_with_nulls = "\x00" * 50
        # This likely has < 50 words, so expect ValueError
        with pytest.raises((ValueError, Exception)):
            detector.detect(text_with_nulls)

    def test_null_bytes_in_long_text_no_crash(self, detector: CIPHERDetector) -> None:
        """Long text with embedded null bytes should not crash."""
        text = VALID_LONG_TEXT + "\x00" * 10
        try:
            result = detector.detect(text, include_sentence_scores=False)
            # If it succeeds, score must be valid
            assert 0.0 <= result.document_score <= 1.0
        except ValueError:
            # ValueError (too short) is acceptable
            pass


# ---------------------------------------------------------------------------
# Extremely long text
# ---------------------------------------------------------------------------


class TestExtremelyLongText:
    """Tests that the pipeline handles very large inputs without crashing."""

    def test_100k_char_text_no_crash(self, detector: CIPHERDetector) -> None:
        """Text of ~100k characters should complete without an uncaught exception."""
        long_text = (VALID_LONG_TEXT + " ") * 100  # ~8k words
        result = detector.detect(long_text, include_sentence_scores=False)
        assert 0.0 <= result.document_score <= 1.0

    def test_large_text_processing_time_reasonable(
        self, detector: CIPHERDetector
    ) -> None:
        """Large text should finish processing (no infinite loop)."""
        import time

        large_text = (VALID_LONG_TEXT + " ") * 50
        start = time.perf_counter()
        detector.detect(large_text, include_sentence_scores=False)
        elapsed = time.perf_counter() - start
        assert elapsed < 60.0  # should finish in under 60 seconds


# ---------------------------------------------------------------------------
# HTML injection
# ---------------------------------------------------------------------------


class TestHTMLInjection:
    """Tests that HTML/script tags in text are treated as plain text."""

    def test_html_script_tag_treated_as_plain_text(
        self, detector: CIPHERDetector
    ) -> None:
        """HTML injection in text should not cause code execution or crash."""
        html_text = VALID_LONG_TEXT + ' <script>alert("xss")</script> ' * 3
        result = detector.detect(html_text, include_sentence_scores=False)
        assert 0.0 <= result.document_score <= 1.0

    def test_html_tags_in_text_no_crash(self, detector: CIPHERDetector) -> None:
        """Various HTML tags should be treated as plain text."""
        html_text = (
            VALID_LONG_TEXT
            + " <b>bold</b> <i>italic</i> <a href='http://example.com'>link</a> " * 5
        )
        result = detector.detect(html_text, include_sentence_scores=False)
        assert 0.0 <= result.document_score <= 1.0


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------


class TestPathTraversal:
    """Tests that path traversal inputs to detect-file exit safely."""

    def test_path_traversal_etc_passwd_exits_one(self) -> None:
        """Path traversal to /etc/passwd should exit 1 (file not found)."""
        result = runner.invoke(
            app, ["detect-file", "../../etc/passwd", "--no-encoder"]
        )
        assert result.exit_code == 1

    def test_path_traversal_absolute_missing_exits_one(self) -> None:
        """Absolute path to a non-existent file should exit 1."""
        result = runner.invoke(
            app, ["detect-file", "/nonexistent/path/to/file.txt", "--no-encoder"]
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# SQL injection patterns
# ---------------------------------------------------------------------------


class TestSQLInjection:
    """Tests that SQL injection patterns in text are treated as plain text."""

    def test_sql_injection_in_text_no_crash(self, detector: CIPHERDetector) -> None:
        """SQL injection patterns should be treated as plain text, not executed."""
        sql_text = (
            VALID_LONG_TEXT
            + " ' OR '1'='1'; DROP TABLE users; -- " * 5
        )
        result = detector.detect(sql_text, include_sentence_scores=False)
        assert 0.0 <= result.document_score <= 1.0

    def test_sql_union_pattern_no_crash(self, detector: CIPHERDetector) -> None:
        """UNION-based SQL injection in text should be harmless."""
        sql_text = (
            VALID_LONG_TEXT
            + " UNION SELECT username, password FROM users WHERE 1=1-- " * 3
        )
        result = detector.detect(sql_text, include_sentence_scores=False)
        assert 0.0 <= result.document_score <= 1.0


# ---------------------------------------------------------------------------
# Unicode edge cases
# ---------------------------------------------------------------------------


class TestUnicodeEdgeCases:
    """Tests for Unicode edge cases in input text."""

    def test_emoji_only_text_too_short(self, detector: CIPHERDetector) -> None:
        """A few emojis are below MIN_WORDS → ValueError."""
        with pytest.raises(ValueError):
            detector.detect("😀🎉🚀✨💫")

    def test_right_to_left_text_no_crash(self, detector: CIPHERDetector) -> None:
        """Arabic/Hebrew RTL text should not crash the pipeline."""
        # Arabic text repeated to meet MIN_WORDS requirement
        rtl_text = "هذا النص باللغة العربية ويجب أن يعمل بشكل صحيح دون أي مشاكل. " * 10
        try:
            result = detector.detect(rtl_text, include_sentence_scores=False)
            assert 0.0 <= result.document_score <= 1.0
        except ValueError:
            # Too short is acceptable
            pass

    def test_zero_width_chars_no_crash(self, detector: CIPHERDetector) -> None:
        """Zero-width characters embedded in text should not crash."""
        zwc_text = VALID_LONG_TEXT.replace(" ", "\u200b \u200c ")  # zero-width spaces
        try:
            result = detector.detect(zwc_text, include_sentence_scores=False)
            assert 0.0 <= result.document_score <= 1.0
        except ValueError:
            pass

    def test_mixed_scripts_no_crash(self, detector: CIPHERDetector) -> None:
        """Mixed scripts (Latin + CJK) should not crash."""
        mixed = (
            VALID_LONG_TEXT
            + " 这是一些中文文字。これは日本語のテキストです。한국어 텍스트입니다. " * 3
        )
        result = detector.detect(mixed, include_sentence_scores=False)
        assert 0.0 <= result.document_score <= 1.0
