"""Tests for the in-memory session store."""

from __future__ import annotations

import time

from app.sessions import Session, SessionStore


def test_create_returns_unique_sessions() -> None:
    store = SessionStore()
    a = store.create("question a")
    b = store.create("question b")
    assert a.id != b.id
    assert store.get(a.id) is a
    assert store.get(b.id) is b


def test_get_nonexistent_returns_none() -> None:
    store = SessionStore()
    assert store.get("nope") is None


def test_drop_removes_session() -> None:
    store = SessionStore()
    sess = store.create("q")
    assert store.get(sess.id) is not None
    store.drop(sess.id)
    assert store.get(sess.id) is None


def test_drop_missing_session_is_noop() -> None:
    store = SessionStore()
    store.drop("never-existed")  # should not raise


def test_get_touches_last_accessed() -> None:
    store = SessionStore(idle_timeout=10.0)
    sess = store.create("q")
    initial = sess.last_touched
    time.sleep(0.01)
    retrieved = store.get(sess.id)
    assert retrieved is sess
    assert sess.last_touched > initial


def test_idle_eviction() -> None:
    store = SessionStore(idle_timeout=0.05)
    sess = store.create("q")
    assert store.get(sess.id) is sess
    time.sleep(0.1)
    # Creating another session triggers the sweep.
    store.create("trigger sweep")
    assert store.get(sess.id) is None


def test_session_history_starts_empty() -> None:
    sess = Session(id="x", question="q")
    assert sess.history == []
    assert sess.chunks_seen == {}
    assert sess.turn == 0
