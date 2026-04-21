"""Thin wrapper around the google-genai SDK.

Two distinct call modes (Gemini does *not* allow JSON-mode and tools together):

* `agentic_turn(history, tools)` — used inside the action loop. Tools enabled,
  no JSON mode. Returns either a `function_call` or free-text continuation.
* `final_answer(history, schema)` — called once after the agentic loop
  completes. No tools, response_mime_type=application/json with a Pydantic
  schema describing `{answer, citations}`. This produces the side-panel output.

Both modes share the same underlying client, model, and generation config.
The full implementation lands in the memory_indexing / action_loop steps;
for the backend skeleton this exposes only the client factory + an
`embed_documents` helper used by the ingest pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


@dataclass(slots=True)
class EmbedResult:
    vectors: list[list[float]]


class GeminiClient:
    """Wraps `google.genai.Client` with the project's defaults applied."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.google_api_key:
            logger.warning("GOOGLE_API_KEY is not set; LLM/embedding calls will fail at runtime.")
        self._client = genai.Client(api_key=self._settings.google_api_key or "missing")

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def raw(self) -> genai.Client:
        return self._client

    async def embed_documents(self, texts: Sequence[str]) -> EmbedResult:
        """Embed a batch of texts as RETRIEVAL_DOCUMENT (asymmetric)."""
        return await self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        result = await self._embed([text], task_type="RETRIEVAL_QUERY")
        return result.vectors[0]

    async def _embed(self, texts: Sequence[str], *, task_type: str) -> EmbedResult:
        if not texts:
            return EmbedResult(vectors=[])

        config = genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._settings.embedding_dim,
        )

        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                response = await asyncio.to_thread(
                    self._client.models.embed_content,
                    model=self._settings.embedding_model,
                    contents=list(texts),
                    config=config,
                )
                vectors = [list(emb.values or []) for emb in (response.embeddings or [])]
                return EmbedResult(vectors=vectors)
            except Exception as exc:  # pragma: no cover - exercised in integration
                status = _extract_status(exc)
                if status not in _RETRYABLE_STATUSES or attempt == 4:
                    raise
                logger.warning("embed retry %d after %.1fs (status=%s)", attempt + 1, delay, status)
                last_exc = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 32.0)

        raise RuntimeError("embed retries exhausted") from last_exc


def _extract_status(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


_cached_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    global _cached_client
    if _cached_client is None:
        _cached_client = GeminiClient()
    return _cached_client


def reset_client_for_tests() -> None:
    global _cached_client
    _cached_client = None


# Surface the genai types module for callers that need to construct
# `FunctionDeclaration` etc. without importing google.genai directly.
__all__ = [
    "EmbedResult",
    "GeminiClient",
    "genai_types",
    "get_gemini_client",
    "reset_client_for_tests",
]


_ = Any  # keep typing import used; placeholder for forthcoming agentic types
