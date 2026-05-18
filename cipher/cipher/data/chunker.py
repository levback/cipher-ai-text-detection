"""Paragraph-boundary-aware text chunker for CIPHER training data.

Strategy
--------
Rather than using hard word-count windows, this chunker accumulates whole
paragraphs until the running word count reaches *min_words*, then closes
the chunk at the **next** paragraph boundary.  This preserves the natural
rhetorical structure of the source document in every training sample.

Algorithm:
    1. Split the document on blank lines to get raw paragraphs.
    2. Walk paragraphs in order, appending them to a pending buffer.
    3. When the buffer word-count reaches or exceeds *min_words*, emit
       the buffer as one chunk and start a new one.
    4. Any remaining paragraphs that are below *min_words* are either
       merged into the last chunk (if ``merge_tail=True``) or discarded.

Advantages over a sliding-window approach:
    - Chunk boundaries always fall at natural paragraph ends.
    - Chunks are self-contained rhetorical units (≥ *min_words* words).
    - Feature estimates (Hurst, discourse topology, etc.) are less noisy
      because each chunk contains at least one full paragraph of context.

Exported symbols
----------------
chunk_document(text, min_words, merge_tail) -> list[str]
    Main entry point: chunk a single document string.

chunk_documents(pairs, min_words, merge_tail) -> list[tuple[str, int, str]]
    Chunk a sequence of (text, label) or (text, label, doc_id) pairs and
    return (chunk_text, label, doc_id) triples with doc-level identity
    preserved, enabling group-aware train/val/test splits.
"""

from __future__ import annotations

import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    text: str,
    min_words: int = 100,
    merge_tail: bool = True,
) -> list[str]:
    """Chunk *text* into paragraph-boundary-aligned segments.

    Splits the document on blank lines, accumulates paragraphs until the
    running word count reaches *min_words*, then emits a chunk.

    Args:
        text:       Raw document text.
        min_words:  Minimum word count to close a chunk at the next
                    paragraph boundary (default 100).
        merge_tail: If True (default), any trailing paragraphs below
                    *min_words* are appended to the last chunk.  If False
                    they are dropped.

    Returns:
        list[str]: List of chunk strings, each containing whole paragraphs
        and at least *min_words* words (except possibly the last chunk when
        *merge_tail* is True and the document is short).  An empty list is
        returned for blank or very-short documents.

    Examples::

        >>> chunks = chunk_document(long_article, min_words=100)
        >>> all(len(c.split()) >= 90 for c in chunks)  # approx lower bound
        True
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if para_words == 0:
            continue
        buffer.append(para)
        buffer_words += para_words

        if buffer_words >= min_words:
            chunks.append("\n\n".join(buffer))
            buffer = []
            buffer_words = 0

    # Handle remaining buffer
    if buffer:
        if merge_tail and chunks:
            # Merge leftovers into the last chunk
            last = chunks.pop()
            chunks.append(last + "\n\n" + "\n\n".join(buffer))
        elif buffer_words >= min_words // 2:
            # Keep the tail only if it is at least half the minimum
            chunks.append("\n\n".join(buffer))
        # Otherwise discard — too short for reliable feature extraction

    return chunks


def chunk_documents(
    pairs: Sequence[tuple[str, int] | tuple[str, int, str]],
    min_words: int = 100,
    merge_tail: bool = True,
) -> list[tuple[str, int, str]]:
    """Chunk a sequence of labelled documents and return (chunk, label, doc_id).

    Each element of *pairs* may be either:
        - (text, label)          — doc_id is auto-generated as "doc_{index}"
        - (text, label, doc_id)  — doc_id is taken verbatim

    The returned triples always carry the source document's doc_id so that
    downstream train/val/test splitting can group all chunks from the same
    document into the same split (no cross-document leakage).

    Args:
        pairs:      Sequence of (text, label) or (text, label, doc_id) tuples.
        min_words:  Minimum words per chunk, forwarded to :func:`chunk_document`.
        merge_tail: Whether to merge sub-minimum tails, forwarded to
                    :func:`chunk_document`.

    Returns:
        list[tuple[str, int, str]]: (chunk_text, label, doc_id) triples.

    Example::

        rows = [(article_text, 0), (ai_text, 1)]
        chunks = chunk_documents(rows, min_words=100)
        # Each chunk knows its source document:
        # ("The quick ...", 0, "doc_0"), ("The quick ...", 0, "doc_0"), ...
    """
    result: list[tuple[str, int, str]] = []
    for idx, item in enumerate(pairs):
        if len(item) == 3:
            text, label, doc_id = item  # type: ignore[misc]
        else:
            text, label = item  # type: ignore[misc]
            doc_id = f"doc_{idx}"

        for chunk in chunk_document(text, min_words=min_words, merge_tail=merge_tail):
            result.append((chunk, int(label), str(doc_id)))

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[str]:
    """Split *text* into non-empty paragraphs on one or more blank lines.

    Normalises Windows line endings before splitting.  Each returned
    paragraph has leading/trailing whitespace stripped.

    Args:
        text: Raw document text.

    Returns:
        list[str]: Non-empty, stripped paragraph strings.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on one or more blank lines
    raw_paragraphs = re.split(r"\n{2,}", text)
    # Clean each paragraph (collapse internal whitespace, strip edges)
    cleaned: list[str] = []
    for para in raw_paragraphs:
        para = re.sub(r"[ \t]+", " ", para).strip()
        if para:
            cleaned.append(para)
    return cleaned
