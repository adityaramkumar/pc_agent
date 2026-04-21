"""Storage-layer tests: schema initializes, inserts/reads/deletes work."""

from __future__ import annotations

from app.store import IngestEvent, Store


def _event(url: str = "https://example.com", ts: int = 1) -> IngestEvent:
    return IngestEvent(
        type="page_visit",
        url=url,
        title="Example",
        text="hello world",
        ts=ts,
        meta={"k": "v"},
    )


def test_schema_creates_expected_tables(store: Store) -> None:
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
        ).fetchall()
    names = {row["name"] for row in rows}
    for required in ("events", "chunks", "chunk_vectors", "chunks_fts"):
        assert required in names, f"missing table: {required}"


def test_insert_and_list_events(store: Store) -> None:
    ids = store.insert_events([_event(ts=10), _event(ts=20), _event(ts=30)])
    assert len(ids) == 3
    assert all(isinstance(i, int) for i in ids)

    rows = store.list_events()
    assert [r.ts for r in rows] == [30, 20, 10]
    assert rows[0].meta == {"k": "v"}


def test_get_and_delete_event(store: Store) -> None:
    [event_id] = store.insert_events([_event()])
    fetched = store.get_event(event_id)
    assert fetched is not None
    assert fetched.url == "https://example.com"

    assert store.delete_event(event_id) is True
    assert store.get_event(event_id) is None
    assert store.delete_event(event_id) is False


def test_count_events(store: Store) -> None:
    assert store.count_events() == 0
    store.insert_events([_event(ts=1), _event(ts=2)])
    assert store.count_events() == 2


def test_chunk_cascade_on_event_delete(store: Store) -> None:
    [event_id] = store.insert_events([_event()])
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO chunks (event_id, text, ts) VALUES (?, ?, ?)",
            (event_id, "chunk text", 1),
        )
        before = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    assert before == 1

    store.delete_event(event_id)
    with store.connect() as conn:
        after = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        fts_after = conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"]
    assert after == 0
    assert fts_after == 0
