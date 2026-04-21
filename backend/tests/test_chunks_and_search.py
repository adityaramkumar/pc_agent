"""Round-trip tests for chunk insertion + FTS5 + vector search."""

from __future__ import annotations

import struct

from app.store import IngestEvent, Store


def _ingest_with_chunks(store: Store, *, text: str, ts: int = 1) -> int:
    [event_id] = store.insert_events(
        [IngestEvent(type="page_visit", url="https://x", title="t", text=text, ts=ts, meta={})]
    )
    chunk_ids = store.insert_chunks(event_id=event_id, texts=[text], ts=ts)
    return chunk_ids[0]


def test_fts_search_finds_chunk_by_token(store: Store) -> None:
    _ingest_with_chunks(store, text="The quick brown fox jumps over the lazy dog")
    matches = store.fts_search("fox jumps", k=5)
    assert len(matches) == 1
    chunk_id, score = matches[0]
    assert isinstance(chunk_id, int)
    assert isinstance(score, float)


def test_fts_search_short_or_empty_query_returns_empty(store: Store) -> None:
    _ingest_with_chunks(store, text="hello world from python")
    assert store.fts_search("") == []
    # `a` has length < 2 so the sanitizer drops it; query becomes empty.
    assert store.fts_search("a") == []


def test_vector_search_round_trip(store: Store) -> None:
    chunk_id = _ingest_with_chunks(store, text="vector world")
    # Insert a deterministic 768-dim vector and query with a perfect match.
    vector = [0.0] * 768
    vector[0] = 1.0
    store.insert_chunk_vectors([(chunk_id, vector)])

    results = store.vector_search(vector, k=3)
    assert len(results) == 1
    assert results[0][0] == chunk_id


def test_fetch_chunks_hydrates_with_event_metadata(store: Store) -> None:
    chunk_id = _ingest_with_chunks(store, text="the answer to everything")
    [chunk] = store.fetch_chunks([chunk_id])
    assert chunk.url == "https://x"
    assert chunk.title == "t"
    assert chunk.text == "the answer to everything"


def test_chunk_vectors_blob_format() -> None:
    """Sanity check the struct format we use for vector blobs."""
    vec = [1.0, 2.0, 3.0]
    blob = struct.pack(f"{len(vec)}f", *vec)
    out = list(struct.unpack(f"{len(vec)}f", blob))
    assert out == vec
