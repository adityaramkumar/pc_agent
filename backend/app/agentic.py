"""Agentic loop orchestration.

The side panel drives the loop: it POSTs a question to `/query/start`, and
we keep calling `agentic_turn` until one of three things happens:

1. The model returns free text. We call `final_answer` with the chunks
   we've seen and return the structured answer.
2. The model calls `search_memory`. We execute it locally and keep looping.
3. The model calls `visit_page` / `extract_from_page`. We hand the call
   back to the side panel to run in a background tab, and close the loop
   when it POSTs /query/continue with the result.

Splitting this out of `main.py` keeps the HTTP layer thin and makes the
loop unit-testable with a mocked LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from google.genai import types as genai_types

from app.llm import AgenticTurn, FunctionCall, GeminiClient
from app.prompts import format_search_results
from app.rag import Retriever
from app.sessions import MAX_TURNS, Session, SessionStore
from app.tools import ToolName

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoopOutcome:
    """Result of driving the loop forward until it blocks or finishes."""

    pending: FunctionCall | None  # browser-side tool the panel must run
    text: str | None  # model's final text reply (if no tool requested)
    turn_cap_hit: bool  # true if we stopped because of MAX_TURNS


# --- Content helpers -----------------------------------------------------


def user_content(text: str) -> Any:
    """A `role=user` Content carrying plain text."""
    return genai_types.Content(role="user", parts=[genai_types.Part(text=text)])


def function_response_content(name: str, payload: dict[str, Any]) -> Any:
    """A `role=user` Content carrying a function_response part."""
    return genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_function_response(name=name, response=payload)],
    )


def last_function_call_name(history: list[Any]) -> str | None:
    """Walk the history backward looking for the most recent function_call part."""
    for content in reversed(history):
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                return str(fc.name)
    return None


# --- The loop ------------------------------------------------------------


async def run_until_blocked_or_done(
    sess: Session,
    *,
    llm: GeminiClient,
    retriever: Retriever,
) -> LoopOutcome:
    """Drive the agentic loop until one of:

    - the model requests a browser-side tool (returned as LoopOutcome.pending)
    - the model gives a final text reply (returned as LoopOutcome.text)
    - we hit MAX_TURNS (turn_cap_hit=True; caller should finalize anyway)
    """
    while sess.turn < MAX_TURNS:
        sess.turn += 1
        turn: AgenticTurn = await llm.agentic_turn(sess.history)
        if turn.raw_content is not None:
            sess.history.append(turn.raw_content)

        fn = turn.function_call
        if fn is None:
            return LoopOutcome(pending=None, text=turn.text, turn_cap_hit=False)

        if fn.name == ToolName.SEARCH_MEMORY.value:
            result_text = await _run_local_search(sess, retriever, fn.args)
            sess.history.append(
                function_response_content(ToolName.SEARCH_MEMORY.value, {"results": result_text})
            )
            continue

        # visit_page / extract_from_page must be executed by the side panel.
        return LoopOutcome(pending=fn, text=None, turn_cap_hit=False)

    return LoopOutcome(pending=None, text=None, turn_cap_hit=True)


async def _run_local_search(sess: Session, retriever: Retriever, args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "(empty query)"
    chunks = await retriever.search(query)
    for c in chunks:
        sess.chunks_seen[c.chunk_id] = c
    return format_search_results(chunks)


# --- Session + response coordination -------------------------------------


@dataclass(slots=True)
class AgenticResult:
    """What the HTTP layer needs to build a QueryResponse.

    Either `answer` is set (loop is done, session dropped) OR `pending` +
    `session_id` are set (side panel must run the tool and come back).
    """

    answer: str | None = None
    citations: list[Any] = None  # type: ignore[assignment]
    session_id: str | None = None
    pending: FunctionCall | None = None


async def drive_and_finalize(
    sess: Session,
    *,
    llm: GeminiClient,
    retriever: Retriever,
    sessions: SessionStore,
) -> AgenticResult:
    """Run the loop and either return a pending tool or the final answer.

    On any exception the session is dropped so we don't leak state.
    """
    try:
        outcome = await run_until_blocked_or_done(sess, llm=llm, retriever=retriever)
    except Exception:
        sessions.drop(sess.id)
        raise

    if outcome.pending is not None:
        return AgenticResult(
            session_id=sess.id,
            pending=outcome.pending,
        )

    # Either the model gave a final text OR the turn cap fired. Either way,
    # call final_answer with the chunks we've accumulated.
    try:
        final = await llm.final_answer(
            question=sess.question,
            retrieved=list(sess.chunks_seen.values()),
        )
    finally:
        sessions.drop(sess.id)

    # Deduplicate citations by URL, preserving first occurrence.
    seen_urls: set[str] = set()
    unique_citations = []
    for c in final.citations:
        if c.url not in seen_urls:
            seen_urls.add(c.url)
            unique_citations.append(c)

    return AgenticResult(
        answer=final.answer,
        citations=unique_citations,
    )
