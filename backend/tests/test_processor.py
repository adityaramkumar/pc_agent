"""Pure-function tests for the chunker."""

from __future__ import annotations

from app.processor import chunk_text, normalize_whitespace


def test_normalize_whitespace_collapses_runs() -> None:
    assert normalize_whitespace("hello   \n\n\tworld  ") == "hello world"


def test_chunk_short_text_below_min_drops() -> None:
    assert chunk_text("hi") == []


def test_chunk_short_text_at_threshold_returns_one() -> None:
    text = "x" * 200
    chunks = chunk_text(text)
    assert chunks == [text]


def test_chunk_long_text_overlaps_and_covers() -> None:
    text = "abcdefghij " * 600  # 6600 chars
    chunks = chunk_text(text, chunk_size=2000, overlap=200, min_size=80)
    assert len(chunks) >= 3
    # Joined chunks (sans overlap accounting) should at least cover the source.
    joined = "".join(chunks)
    assert len(joined) >= len(text.strip())
    # Each chunk respects the size cap.
    for chunk in chunks:
        assert len(chunk) <= 2000


def test_chunk_prefers_sentence_boundary() -> None:
    sentence = "This is a complete sentence. " * 100  # ~3000 chars
    chunks = chunk_text(sentence, chunk_size=800, overlap=80, min_size=80)
    # We expect most chunks to end on a period since the sentence pattern is dense.
    ending_on_period = sum(1 for c in chunks[:-1] if c.rstrip().endswith("."))
    assert ending_on_period >= max(1, len(chunks) - 2)
