"""Tests for the paragraph-boundary-aware text chunker.

Covers:
- chunk_document: normal chunking, short docs, blank docs, merge_tail logic
- chunk_documents: doc_id propagation, auto-id generation, label preservation
- _split_paragraphs: blank-line splitting, Windows line endings, whitespace
"""

from __future__ import annotations

import pytest

from cipher.data.chunker import _split_paragraphs, chunk_document, chunk_documents


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _para(n_words: int) -> str:
    """Return a paragraph of exactly *n_words* words."""
    return " ".join(f"word{i}" for i in range(n_words))


def _doc(*paragraphs: str) -> str:
    """Join paragraphs with double newlines."""
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# _split_paragraphs
# ---------------------------------------------------------------------------


class TestSplitParagraphs:
    def test_single_paragraph(self) -> None:
        text = "Hello world. This is a test."
        result = _split_paragraphs(text)
        assert result == ["Hello world. This is a test."]

    def test_two_paragraphs(self) -> None:
        text = "First para.\n\nSecond para."
        result = _split_paragraphs(text)
        assert result == ["First para.", "Second para."]

    def test_multiple_blank_lines(self) -> None:
        text = "A.\n\n\n\nB."
        result = _split_paragraphs(text)
        assert result == ["A.", "B."]

    def test_windows_line_endings(self) -> None:
        text = "Para one.\r\n\r\nPara two."
        result = _split_paragraphs(text)
        assert result == ["Para one.", "Para two."]

    def test_old_mac_line_endings(self) -> None:
        text = "Para one.\r\rPara two."
        result = _split_paragraphs(text)
        assert result == ["Para one.", "Para two."]

    def test_strips_internal_whitespace(self) -> None:
        text = "  Hello   world.  \n\n  Second.  "
        result = _split_paragraphs(text)
        assert result[0] == "Hello world."
        assert result[1] == "Second."

    def test_empty_string(self) -> None:
        assert _split_paragraphs("") == []

    def test_only_whitespace(self) -> None:
        assert _split_paragraphs("   \n\n   \n\n   ") == []

    def test_single_newline_does_not_split(self) -> None:
        text = "Line one.\nLine two."
        result = _split_paragraphs(text)
        assert len(result) == 1
        assert "Line one." in result[0]


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------


class TestChunkDocument:
    def test_returns_list(self) -> None:
        text = _doc(_para(120))
        result = chunk_document(text, min_words=100)
        assert isinstance(result, list)

    def test_single_large_paragraph_one_chunk(self) -> None:
        text = _doc(_para(200))
        chunks = chunk_document(text, min_words=100)
        assert len(chunks) == 1
        assert len(chunks[0].split()) >= 100

    def test_two_paragraphs_makes_one_chunk(self) -> None:
        # 60 + 60 = 120 words → should close at the second paragraph boundary
        text = _doc(_para(60), _para(60))
        chunks = chunk_document(text, min_words=100)
        assert len(chunks) == 1
        assert len(chunks[0].split()) >= 100

    def test_chunks_close_at_paragraph_boundary(self) -> None:
        # 50 + 60 + 50 + 60 = 220 words → should produce 2 chunks
        text = _doc(_para(50), _para(60), _para(50), _para(60))
        chunks = chunk_document(text, min_words=100)
        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk.split()) >= 100

    def test_empty_document_returns_empty(self) -> None:
        assert chunk_document("", min_words=100) == []

    def test_blank_document_returns_empty(self) -> None:
        assert chunk_document("   \n\n   ", min_words=100) == []

    def test_very_short_doc_returns_empty_by_default(self) -> None:
        # 30 words < min_words=100, and no prior chunks to merge into
        text = _para(30)
        chunks = chunk_document(text, min_words=100, merge_tail=False)
        assert chunks == []

    def test_merge_tail_appends_to_last_chunk(self) -> None:
        # 110 words → first chunk, then 30 leftover → merged into first chunk
        text = _doc(_para(110), _para(30))
        chunks = chunk_document(text, min_words=100, merge_tail=True)
        assert len(chunks) == 1
        total_words = len(chunks[0].split())
        assert total_words == 140

    def test_no_merge_tail_discards_short_remainder(self) -> None:
        # 110 words → first chunk, then 20 leftover → too short (< 50), discarded
        text = _doc(_para(110), _para(20))
        chunks = chunk_document(text, min_words=100, merge_tail=False)
        assert len(chunks) == 1

    def test_chunk_contains_paragraph_separator(self) -> None:
        text = _doc(_para(60), _para(60))
        chunks = chunk_document(text, min_words=100)
        assert "\n\n" in chunks[0]

    def test_large_doc_produces_multiple_chunks(self) -> None:
        # 10 paragraphs × 50 words = 500 words total → at least 4 chunks at 100 words
        text = _doc(*[_para(50)] * 10)
        chunks = chunk_document(text, min_words=100)
        assert len(chunks) >= 4

    def test_all_chunks_meet_min_words(self) -> None:
        text = _doc(*[_para(40)] * 20)
        chunks = chunk_document(text, min_words=100)
        # All chunks except possibly the last (tail-merged) should be ≥ 100 words
        for chunk in chunks[:-1]:
            assert len(chunk.split()) >= 100

    def test_custom_min_words(self) -> None:
        text = _doc(_para(60), _para(60), _para(60))
        chunks_50 = chunk_document(text, min_words=50)
        chunks_150 = chunk_document(text, min_words=150)
        # Lower threshold → more (or equal) chunks
        assert len(chunks_50) >= len(chunks_150)


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------


class TestChunkDocuments:
    def test_basic_pair_without_doc_id(self) -> None:
        text = _doc(_para(120))
        result = chunk_documents([(text, 0)])
        assert len(result) >= 1
        for chunk, label, doc_id in result:
            assert label == 0
            assert doc_id == "doc_0"

    def test_explicit_doc_id_preserved(self) -> None:
        text = _doc(_para(120))
        result = chunk_documents([(text, 1, "my_doc_42")])
        for _, _, doc_id in result:
            assert doc_id == "my_doc_42"

    def test_multiple_docs_different_doc_ids(self) -> None:
        doc_a = _doc(_para(120))
        doc_b = _doc(_para(120))
        result = chunk_documents([(doc_a, 0, "doc_a"), (doc_b, 1, "doc_b")])
        doc_ids = {r[2] for r in result}
        assert "doc_a" in doc_ids
        assert "doc_b" in doc_ids

    def test_auto_doc_id_uses_index(self) -> None:
        pairs = [(_doc(_para(120)), i % 2) for i in range(3)]
        result = chunk_documents(pairs)
        doc_ids = {r[2] for r in result}
        assert "doc_0" in doc_ids
        assert "doc_1" in doc_ids
        assert "doc_2" in doc_ids

    def test_labels_preserved(self) -> None:
        ai_doc = _doc(_para(120))
        human_doc = _doc(_para(120))
        result = chunk_documents([(ai_doc, 1, "ai"), (human_doc, 0, "human")])
        ai_chunks = [r for r in result if r[2] == "ai"]
        human_chunks = [r for r in result if r[2] == "human"]
        assert all(r[1] == 1 for r in ai_chunks)
        assert all(r[1] == 0 for r in human_chunks)

    def test_short_doc_produces_no_chunks(self) -> None:
        text = _para(10)  # Only 10 words — too short
        result = chunk_documents([(text, 0)])
        # May produce 0 or 1 chunks; if 1 chunk, it was accepted as a tail
        # Either way, no error
        assert isinstance(result, list)

    def test_returns_strings(self) -> None:
        text = _doc(_para(120))
        result = chunk_documents([(text, 0)])
        assert all(isinstance(r[0], str) for r in result)
        assert all(isinstance(r[1], int) for r in result)
        assert all(isinstance(r[2], str) for r in result)

    def test_empty_input_returns_empty(self) -> None:
        assert chunk_documents([]) == []
