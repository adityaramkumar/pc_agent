"""In-memory session store for the agentic loop.

A `Session` is a short-lived (≤5 min idle) container holding the chat
history with Gemini, the chunks the LLM has seen via `search_memory`
(needed for citations on the final answer), and a turn counter that
caps the loop to keep cost bounded.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.store import RetrievedChunk

SESSION_IDLE_TIMEOUT_SEC = 300
_EVICTION_INTERVAL_SEC = 60
MAX_TURNS = 5


@dataclass(slots=True)
class Session:
    id: str
    question: str
    history: list[Any] = field(default_factory=list)  # google.genai Content list
    chunks_seen: dict[int, RetrievedChunk] = field(default_factory=dict)
    turn: int = 0
    created_at: float = field(default_factory=time.time)
    last_touched: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_touched = time.time()


class SessionStore:
    def __init__(
        self,
        idle_timeout: float = SESSION_IDLE_TIMEOUT_SEC,
        eviction_interval: float = _EVICTION_INTERVAL_SEC,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._idle_timeout = idle_timeout
        self._start_eviction_thread(eviction_interval)

    def _start_eviction_thread(self, interval: float) -> None:
        def _loop() -> None:
            while True:
                time.sleep(interval)
                with self._lock:
                    self._evict_expired_locked()

        t = threading.Thread(target=_loop, daemon=True, name="session-eviction")
        t.start()

    def create(self, question: str) -> Session:
        sess = Session(id=uuid.uuid4().hex, question=question)
        with self._lock:
            self._evict_expired_locked()
            self._sessions[sess.id] = sess
        return sess

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._evict_expired_locked()
            sess = self._sessions.get(session_id)
            if sess is not None:
                sess.touch()
            return sess

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired_locked(self) -> None:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items() if now - s.last_touched > self._idle_timeout
        ]
        for sid in expired:
            self._sessions.pop(sid, None)


_cached: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _cached
    if _cached is None:
        _cached = SessionStore()
    return _cached


def reset_session_store_for_tests() -> None:
    global _cached
    _cached = None
