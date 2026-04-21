"""Thin wrapper around the google-genai SDK.

Two distinct call modes (Gemini does *not* allow JSON-mode and tools together):

* `agentic_turn(history, tools)` — used inside the action loop. Tools enabled,
  no JSON mode. Returns either a `function_call` or free-text continuation.
* `final_answer(question, retrieved)` — no tools, response_mime_type=
  application/json with a schema describing `{answer, citations}`. This is the
  call that produces the side-panel-rendered output.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.config import Settings, get_settings
from app.store import RetrievedChunk

logger = logging.getLogger(__name__)


_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


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


_FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "ts": {"type": "integer"},
                    "snippet": {"type": "string"},
                },
                "required": ["url", "ts", "snippet"],
            },
        },
    },
    "required": ["answer", "citations"],
}


_SYSTEM_INSTRUCTION = (
    "You are pc_agent, the user's personal browser-history assistant. "
    "Answer the user's question using ONLY the provided memories from their "
    "own browsing. If the memories don't contain enough information, say so "
    "honestly rather than guessing. Always include citations: copy the "
    "URL/ts/snippet of every memory you actually used. Be concise."
)


_AGENTIC_SYSTEM_INSTRUCTION = (
    "You are pc_agent, the user's personal browser-history assistant. "
    "You have three tools:\n"
    "  - search_memory(query): search the user's local browsing history.\n"
    "  - visit_page(url, wait_for_selector?): open a tab in the background\n"
    "    and read the page. Use ONLY when the user asks you to go check\n"
    "    something live (e.g. 'check what X replied').\n"
    "  - extract_from_page(url, what, css_hint?): read a known page with a\n"
    "    targeted extraction. Use for SPAs (Gmail, LinkedIn) where Readability\n"
    "    returns garbage; pass a CSS selector hint for the relevant region.\n\n"
    "Guidance:\n"
    "  - Always start with search_memory to see what the user already has.\n"
    "  - Only use visit_page / extract_from_page when memory alone is\n"
    "    insufficient and the user is clearly asking you to fetch.\n"
    "  - When you have enough information, write a short final answer\n"
    "    naming the URLs and timestamps you used. The system will then\n"
    "    convert your answer into a structured response with citations."
)


# --- Tool declarations ---------------------------------------------------

SEARCH_MEMORY_TOOL = genai_types.FunctionDeclaration(
    name="search_memory",
    description=(
        "Search the user's local browsing history (pages, selections, form "
        "inputs they've sent). Returns the top relevant snippets."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "query": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="A natural-language search query.",
            ),
        },
        required=["query"],
    ),
)

VISIT_PAGE_TOOL = genai_types.FunctionDeclaration(
    name="visit_page",
    description=(
        "Open a URL in a background tab and return the page's main content. "
        "Uses Readability to extract the article-style text. Best for "
        "article/blog/docs pages."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "url": genai_types.Schema(type=genai_types.Type.STRING),
            "wait_for_selector": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Optional CSS selector to wait for before extracting.",
            ),
        },
        required=["url"],
    ),
)

EXTRACT_FROM_PAGE_TOOL = genai_types.FunctionDeclaration(
    name="extract_from_page",
    description=(
        "Open a URL and extract content from a specific CSS-targeted region. "
        "Use this for SPAs (Gmail, LinkedIn, Slack web) where the main "
        "content lives in dynamic regions and Readability returns junk."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "url": genai_types.Schema(type=genai_types.Type.STRING),
            "what": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="What to look for (free-text, included in the result).",
            ),
            "css_hint": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="A CSS selector for the region of interest.",
            ),
        },
        required=["url", "what"],
    ),
)


def all_tools() -> list[genai_types.Tool]:
    return [
        genai_types.Tool(
            function_declarations=[
                SEARCH_MEMORY_TOOL,
                VISIT_PAGE_TOOL,
                EXTRACT_FROM_PAGE_TOOL,
            ]
        )
    ]


# --- Agentic-turn output -------------------------------------------------


@dataclass(slots=True)
class FunctionCall:
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class AgenticTurn:
    """Result of one round-trip with the model in tool-calling mode."""

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

    async def agentic_turn(self, history: Sequence[Any]) -> AgenticTurn:
        """One round-trip with the model in tool-calling mode."""
        # mypy can't reconcile the SDK's covariant `list[Tool|...]` with our
        # plain `list[Tool]`, so cast at the boundary.
        tools: Any = all_tools()
        config = genai_types.GenerateContentConfig(
            system_instruction=_AGENTIC_SYSTEM_INSTRUCTION,
            tools=tools,
            temperature=0.2,
        )
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._settings.llm_model,
            contents=list(history),
            config=config,
        )
        return _parse_agentic_response(response)

    async def final_answer(
        self,
        *,
        question: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> FinalAnswer:
        """Single-shot answer-with-citations call. No tools, JSON output."""
        prompt = _build_final_prompt(question, retrieved)

        config = genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_FINAL_ANSWER_SCHEMA,
            temperature=0.2,
        )

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._settings.llm_model,
            contents=prompt,
            config=config,
        )

        return _parse_final_answer(response)


def _parse_agentic_response(response: Any) -> AgenticTurn:
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
            # google.genai gives us a dict-like; coerce to plain dict.
            try:
                args = dict(args)
            except Exception:
                args = {}
            fn_call = FunctionCall(name=function_call.name, args=args)
            break  # only one function call per turn for our purposes
        text = getattr(part, "text", None)
        if text:
            text_pieces.append(text)

    return AgenticTurn(
        function_call=fn_call,
        text="".join(text_pieces) if text_pieces else None,
        raw_content=content,
    )


def _build_final_prompt(question: str, retrieved: Sequence[RetrievedChunk]) -> str:
    if not retrieved:
        body = "(no memories matched the question)"
    else:
        body = "\n\n".join(_format_chunk(i, c) for i, c in enumerate(retrieved, start=1))
    return (
        f"# Question\n{question}\n\n"
        f"# Memories from the user's browsing\n{body}\n\n"
        "# Task\nAnswer the question using only the memories above. "
        "Include citations for the memories you used."
    )


def _format_chunk(idx: int, chunk: RetrievedChunk) -> str:
    iso = datetime.fromtimestamp(chunk.ts / 1000, tz=UTC).isoformat()
    title = chunk.title or chunk.url
    return (
        f"[{idx}] {title}\n"
        f"    url: {chunk.url}\n"
        f"    ts:  {chunk.ts}  ({iso})\n"
        f"    text: {chunk.text}"
    )


def _parse_final_answer(response: Any) -> FinalAnswer:
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
    "AgenticTurn",
    "Citation",
    "EmbedResult",
    "FinalAnswer",
    "FunctionCall",
    "GeminiClient",
    "all_tools",
    "genai_types",
    "get_gemini_client",
    "reset_client_for_tests",
]
