"""Turns ingested events into chunks + embeddings.

Chunks are pure-function character-window splits; embeddings come from
Gemini's `gemini-embedding-001` with task_type=RETRIEVAL_DOCUMENT, batched
in groups of 100 (the API's hard limit).

Embedding errors are swallowed: the chunks are still inserted (so FTS5
keyword search keeps working), they just have no vector entry. Callers
shouldn't depend on every chunk being embedded.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import TypeVar

from app.llm import GeminiClient, get_gemini_client
from app.store import Store, get_store

logger = logging.getLogger(__name__)


CHUNK_SIZE_CHARS = 2_000
CHUNK_OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 80
EMBED_BATCH_SIZE = 100

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
    min_size: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """Split a string into overlapping windows.

    Boundaries are nudged to the nearest sentence-ish break (period, newline,
    or whitespace) within the last 10% of the window so we avoid cleaving
    sentences in the middle.
    """
    text = normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_size else []

    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        # Try to nudge `end` back to a sentence boundary in the last 10% of
        # the window. If we can't find a good break, use the hard cut.
        if end < len(text):
            search_start = max(pos + int(chunk_size * 0.9), pos + min_size)
            window = text[search_start:end]
            # Prefer ". " then any whitespace.
            best = window.rfind(". ")
            if best == -1:
                best = window.rfind(" ")
            if best != -1:
                end = search_start + best + 1
        chunk = text[pos:end].strip()
        if len(chunk) >= min_size:
            chunks.append(chunk)
        if end >= len(text):
            break
        pos = max(end - overlap, pos + 1)
    return chunks


_T = TypeVar("_T")


def _batched(items: list[_T], n: int) -> Iterator[list[_T]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


class Processor:
    """Stateless helper that wires Store + GeminiClient together for ingest."""

    def __init__(
        self,
        store: Store | None = None,
        client: GeminiClient | None = None,
    ) -> None:
        self._store = store or get_store()
        self._client = client or get_gemini_client()

    async def process_event(self, event_id: int) -> int:
        """Chunk + embed + persist. Returns the number of chunks written."""
        row = self._store.get_event(event_id)
        if row is None or not row.text:
            return 0

        chunks = chunk_text(row.text)
        if not chunks:
            return 0

        chunk_ids = self._store.insert_chunks(event_id=event_id, texts=chunks, ts=row.ts)

        try:
            for batch_ids, batch_texts in zip(
                _batched(chunk_ids, EMBED_BATCH_SIZE),
                _batched(chunks, EMBED_BATCH_SIZE),
                strict=True,
            ):
                result = await self._client.embed_documents(batch_texts)
                self._store.insert_chunk_vectors(list(zip(batch_ids, result.vectors, strict=True)))
        except Exception as exc:  # pragma: no cover - depends on live API
            logger.warning(
                "embedding failed for event %s; chunks stored without vectors: %s",
                event_id,
                exc,
            )

        return len(chunk_ids)


_cached: Processor | None = None


def get_processor() -> Processor:
    global _cached
    if _cached is None:
        _cached = Processor()
    return _cached


def reset_processor_for_tests() -> None:
    global _cached
    _cached = None
