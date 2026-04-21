"""Wrapper around the google-genai SDK.

Three capabilities exposed:

1. Embeddings (`embed_documents`, `embed_query`) with task_type asymmetry
   and exponential backoff on 429/5xx.
2. `agentic_turn(history)`, the tool-calling round-trip used by the action
   loop. Returns either a function call or free text.
3. `final_answer(question, retrieved)`, a JSON-mode single-shot that turns
   a question + retrieved memories into a structured `{answer, citations}`.

Gemini doesn't allow `response_mime_type=application/json` together with
`tools`, which is why the two call modes are separate methods.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.config import Settings, get_settings
from app.prompts import (
    AGENTIC_SYSTEM_INSTRUCTION,
    FINAL_ANSWER_SCHEMA,
    FINAL_ANSWER_SYSTEM_INSTRUCTION,
    build_final_answer_prompt,
)
from app.store import RetrievedChunk
from app.tools import all_tools

logger = logging.getLogger(__name__)


_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_EMBED_RETRIES = 5
_EMBED_INITIAL_DELAY = 1.0
_EMBED_MAX_DELAY = 32.0


@dataclass(slots=True)
class EmbedResult:
    vectors: list[list[float]]


@dataclass(slots=True)
class Citation:
    url: str
    ts: int
    snippet: str


@dataclass(slots=True)
class FinalAnswer:
    answer: str
    citations: list[Citation]


@dataclass(slots=True)
class FunctionCall:
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class AgenticTurn:
    """One round-trip with the model in tool-calling mode."""

    function_call: FunctionCall | None
    text: str | None
    raw_content: Any  # google.genai Content for the model's reply


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

    # --- Embeddings ------------------------------------------------------

    async def embed_documents(self, texts: Sequence[str]) -> EmbedResult:
        """Embed a batch of texts as RETRIEVAL_DOCUMENT (asymmetric)."""
        return await self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string as RETRIEVAL_QUERY."""
        result = await self._embed([text], task_type="RETRIEVAL_QUERY")
        return result.vectors[0]

    async def _embed(self, texts: Sequence[str], *, task_type: str) -> EmbedResult:
        if not texts:
            return EmbedResult(vectors=[])

        config = genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._settings.embedding_dim,
        )

        delay = _EMBED_INITIAL_DELAY
        last_exc: Exception | None = None
        for attempt in range(_MAX_EMBED_RETRIES):
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
                status_code = _extract_status(exc)
                if status_code not in _RETRYABLE_STATUSES or attempt == _MAX_EMBED_RETRIES - 1:
                    raise
                logger.warning(
                    "embed retry %d after %.1fs (status=%s)", attempt + 1, delay, status_code
                )
                last_exc = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, _EMBED_MAX_DELAY)

        raise RuntimeError("embed retries exhausted") from last_exc

    # --- Agentic turn (tools, no JSON) -----------------------------------

    async def agentic_turn(self, history: Sequence[Any]) -> AgenticTurn:
        # mypy can't reconcile the SDK's covariant `list[Tool|...]` with our
        # plain `list[Tool]`, so cast at the boundary.
        tools: Any = all_tools()
        config = genai_types.GenerateContentConfig(
            system_instruction=AGENTIC_SYSTEM_INSTRUCTION,
            tools=tools,
            temperature=0.2,
        )
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._settings.llm_model,
            contents=list(history),
            config=config,
        )
        return parse_agentic_response(response)

    # --- Final answer (JSON, no tools) -----------------------------------

    async def final_answer(
        self,
        *,
        question: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> FinalAnswer:
        prompt = build_final_answer_prompt(question, retrieved)

        config = genai_types.GenerateContentConfig(
            system_instruction=FINAL_ANSWER_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=FINAL_ANSWER_SCHEMA,
            temperature=0.2,
        )

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._settings.llm_model,
            contents=prompt,
            config=config,
        )

        return parse_final_answer(response)


# --- Response parsers (pure, easy to unit-test) -------------------------


def parse_agentic_response(response: Any) -> AgenticTurn:
    """Turn a google-genai Content response into our AgenticTurn dataclass."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return AgenticTurn(function_call=None, text=None, raw_content=None)
    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []

    text_pieces: list[str] = []
    fn_call: FunctionCall | None = None
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            args = getattr(function_call, "args", {}) or {}
            try:
                args_dict = dict(args)
            except Exception:
                args_dict = {}
            fn_call = FunctionCall(name=function_call.name, args=args_dict)
            break  # one function call per turn is all we handle
        text = getattr(part, "text", None)
        if text:
            text_pieces.append(text)

    return AgenticTurn(
        function_call=fn_call,
        text="".join(text_pieces) if text_pieces else None,
        raw_content=content,
    )


def parse_final_answer(response: Any) -> FinalAnswer:
    """Turn the JSON-mode response into a FinalAnswer. Tolerates malformed JSON."""
    raw = getattr(response, "text", None)
    if not raw:
        return FinalAnswer(answer="(no response from model)", citations=[])
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return FinalAnswer(answer=raw, citations=[])
    citations_raw = payload.get("citations") or []
    citations = [
        Citation(
            url=str(c.get("url", "")),
            ts=int(c.get("ts", 0)),
            snippet=str(c.get("snippet", "")),
        )
        for c in citations_raw
        if isinstance(c, dict)
    ]
    return FinalAnswer(answer=str(payload.get("answer", "")), citations=citations)


def _extract_status(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


# --- Cached singleton -----------------------------------------------------


_cached_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    global _cached_client
    if _cached_client is None:
        _cached_client = GeminiClient()
    return _cached_client


def reset_client_for_tests() -> None:
    global _cached_client
    _cached_client = None


__all__ = [
    "AgenticTurn",
    "Citation",
    "EmbedResult",
    "FinalAnswer",
    "FunctionCall",
    "GeminiClient",
    "get_gemini_client",
    "parse_agentic_response",
    "parse_final_answer",
    "reset_client_for_tests",
]
