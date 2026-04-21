"""SQLite + sqlite-vec storage layer.

A single-writer, many-reader SQLite DB. Tables:

  events        — raw captures from the extension
  chunks        — derived text chunks (populated by app/processor.py later)
  chunk_vectors — vec0 virtual table holding 768-dim embeddings keyed by chunk_id
  chunks_fts    — FTS5 virtual table mirroring chunks.text for keyword search

Connections are created per-call via a small context manager so each request
gets its own; SQLite handles concurrent readers fine and our writes are
infrequent enough that a simple locking story works.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Prefer the pysqlite3 wheel when present: it bundles a recent sqlite compiled
# with extension loading enabled, which the stdlib `sqlite3` is missing on some
# distributions (notably macOS Python.org builds). Fall back to the stdlib
# module when the wheel isn't available — Linux Python ships with extensions
# enabled so CI is fine.
try:
    from pysqlite3 import dbapi2 as sqlite3  # type: ignore[import-not-found, unused-ignore]
except ImportError:  # pragma: no cover - exercised on systems without pysqlite3
    import sqlite3  # type: ignore[no-redef, unused-ignore]

import sqlite_vec

from app.config import Settings


@dataclass(slots=True)
class EventRow:
    id: int
    type: str
    url: str
    title: str | None
    text: str | None
    ts: int
    meta: dict[str, Any]


@dataclass(slots=True)
class IngestEvent:
    type: str
    url: str
    title: str | None
    text: str | None
    ts: int
    meta: dict[str, Any]


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: int
    event_id: int
    text: str
    ts: int
    url: str
    title: str | None
    score: float
    sources: tuple[str, ...]  # which retrievers contributed (e.g. ("vec", "fts"))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    type      TEXT    NOT NULL,
    url       TEXT    NOT NULL,
    title     TEXT,
    text      TEXT,
    ts        INTEGER NOT NULL,
    meta_json TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS events_ts_idx  ON events(ts DESC);
CREATE INDEX IF NOT EXISTS events_url_idx ON events(url);

CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    text     TEXT    NOT NULL,
    ts       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_event_idx ON chunks(event_id);
CREATE INDEX IF NOT EXISTS chunks_ts_idx    ON chunks(ts DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


class Store:
    """Thin wrapper that owns the DB path and hands out connections."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._init_lock = threading.Lock()
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self._ensure_initialized()
        conn = self._open_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            # Double-checked locking: another thread may have raced us into the
            # critical section. mypy can't see the concurrent mutation so we
            # silence its unreachable warning here.
            if self._initialized:
                return  # type: ignore[unreachable]
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._open_connection()
            try:
                conn.executescript(_SCHEMA)
            finally:
                conn.close()
            self._initialized = True

    # --- writes ------------------------------------------------------------

    def insert_events(self, events: list[IngestEvent]) -> list[int]:
        if not events:
            return []
        ids: list[int] = []
        with self.connect() as conn:
            cur = conn.cursor()
            for ev in events:
                cur.execute(
                    """
                    INSERT INTO events (type, url, title, text, ts, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ev.type,
                        ev.url,
                        ev.title,
                        ev.text,
                        ev.ts,
                        json.dumps(ev.meta or {}),
                    ),
                )
                row_id = cur.lastrowid
                if row_id is None:
                    raise RuntimeError("sqlite did not return a lastrowid for INSERT")
                ids.append(int(row_id))
        return ids

    def delete_event(self, event_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            return bool(cur.rowcount and cur.rowcount > 0)

    # --- reads -------------------------------------------------------------

    def list_events(self, limit: int = 100, offset: int = 0) -> list[EventRow]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, type, url, title, text, ts, meta_json
                FROM events
                ORDER BY ts DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [_row_to_event(row) for row in rows]

    def get_event(self, event_id: int) -> EventRow | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, type, url, title, text, ts, meta_json
                FROM events WHERE id = ?
                """,
                (event_id,),
            ).fetchone()
        return _row_to_event(row) if row else None

    def count_events(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
        return int(row["n"])

    # --- chunks + vectors --------------------------------------------------

    def insert_chunks(self, *, event_id: int, texts: list[str], ts: int) -> list[int]:
        if not texts:
            return []
        ids: list[int] = []
        with self.connect() as conn:
            cur = conn.cursor()
            for text in texts:
                cur.execute(
                    "INSERT INTO chunks (event_id, text, ts) VALUES (?, ?, ?)",
                    (event_id, text, ts),
                )
                row_id = cur.lastrowid
                if row_id is None:
                    raise RuntimeError("sqlite did not return a lastrowid for chunks insert")
                ids.append(int(row_id))
        return ids

    def insert_chunk_vectors(self, items: list[tuple[int, list[float]]]) -> None:
        if not items:
            return
        import struct

        with self.connect() as conn:
            cur = conn.cursor()
            for chunk_id, vector in items:
                blob = struct.pack(f"{len(vector)}f", *vector)
                cur.execute(
                    "INSERT OR REPLACE INTO chunk_vectors(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, blob),
                )

    def vector_search(self, query_vector: list[float], *, k: int = 20) -> list[tuple[int, float]]:
        """Returns (chunk_id, distance) pairs ordered by ascending distance."""
        if not query_vector:
            return []
        import struct

        blob = struct.pack(f"{len(query_vector)}f", *query_vector)
        with self.connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT chunk_id, distance
                    FROM chunk_vectors
                    WHERE embedding MATCH ?
                      AND k = ?
                    ORDER BY distance
                    """,
                    (blob, k),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                # Empty vec0 tables sometimes raise on KNN queries with 0 rows;
                # treat as no results rather than a hard failure.
                if "no such" in str(exc).lower() or "match" in str(exc).lower():
                    return []
                raise
        return [(int(r["chunk_id"]), float(r["distance"])) for r in rows]

    def fts_search(self, query: str, *, k: int = 20) -> list[tuple[int, float]]:
        """Returns (chunk_id, bm25_score) pairs. Lower bm25 = more relevant."""
        cleaned = _sanitize_fts_query(query)
        if not cleaned:
            return []
        with self.connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT rowid, bm25(chunks_fts) AS score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (cleaned, k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [(int(r["rowid"]), float(r["score"])) for r in rows]

    def fetch_chunks(self, chunk_ids: list[int]) -> list[RetrievedChunk]:
        """Hydrate a list of chunk IDs with chunk + event metadata."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.event_id, c.text, c.ts,
                       e.url, e.title
                FROM chunks c JOIN events e ON e.id = c.event_id
                WHERE c.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        return [
            RetrievedChunk(
                chunk_id=int(r["chunk_id"]),
                event_id=int(r["event_id"]),
                text=r["text"],
                ts=int(r["ts"]),
                url=r["url"],
                title=r["title"],
                score=0.0,
                sources=(),
            )
            for r in rows
        ]


_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _sanitize_fts_query(query: str) -> str:
    """FTS5's MATCH grammar is picky; reduce to a safe OR of tokens."""
    tokens = _FTS_TOKEN_RE.findall(query)
    if not tokens:
        return ""
    quoted = [f'"{t}"' for t in tokens if len(t) >= 2]
    return " OR ".join(quoted)


def _row_to_event(row: sqlite3.Row) -> EventRow:
    raw_meta = row["meta_json"]
    try:
        meta = json.loads(raw_meta) if raw_meta else {}
    except json.JSONDecodeError:
        meta = {}
    return EventRow(
        id=int(row["id"]),
        type=row["type"],
        url=row["url"],
        title=row["title"],
        text=row["text"],
        ts=int(row["ts"]),
        meta=meta,
    )


_cached_store: Store | None = None
_cached_store_lock = threading.Lock()


def get_store(settings: Settings | None = None) -> Store:
    global _cached_store
    if _cached_store is not None:
        return _cached_store
    with _cached_store_lock:
        if _cached_store is None:
            from app.config import get_settings

            cfg = settings or get_settings()
            _cached_store = Store(cfg.db_path)
    return _cached_store


def reset_store_for_tests() -> None:
    global _cached_store
    _cached_store = None
