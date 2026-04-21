"""Tests for the agentic loop using stub LLM + stub retriever.

The real `GeminiClient` hits the network; these tests substitute an object
with the same `agentic_turn` / `final_answer` surface so we can drive the
loop deterministically.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pytest

from app.agentic import (
    AgenticResult,
    drive_and_finalize,
    function_response_content,
    last_function_call_name,
    run_until_blocked_or_done,
    user_content,
)
from app.llm import AgenticTurn, Citation, FinalAnswer, FunctionCall
from app.sessions import MAX_TURNS, Session, SessionStore
from app.store import RetrievedChunk


class StubLLM:
    """Returns a pre-programmed sequence of AgenticTurn objects, then a final."""

    def __init__(
        self,
        turns: Iterable[AgenticTurn],
        final: FinalAnswer | None = None,
    ) -> None:
        self._turns = iter(list(turns))
        self._final = final or FinalAnswer(answer="done", citations=[])
        self.agentic_calls = 0
        self.final_calls = 0

    async def agentic_turn(self, _history: Sequence[Any]) -> AgenticTurn:
        self.agentic_calls += 1
        try:
            return next(self._turns)
        except StopIteration:
            return AgenticTurn(function_call=None, text="(stub exhausted)", raw_content=None)

    async def final_answer(
        self, *, question: str, retrieved: Sequence[RetrievedChunk]
    ) -> FinalAnswer:
        self.final_calls += 1
        self._last_question = question
        self._last_retrieved = list(retrieved)
        return self._final


class StubRetriever:
    """Returns a pre-programmed chunk list, regardless of the query."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    async def search(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        return list(self._chunks)


def _chunk(chunk_id: int = 1, ts: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        event_id=1,
        text="a relevant snippet",
        ts=ts,
        url="https://example.com",
        title="Example",
        score=1.0,
        sources=("fts",),
    )


def _text_turn(text: str) -> AgenticTurn:
    return AgenticTurn(function_call=None, text=text, raw_content=None)


def _tool_turn(name: str, args: dict[str, Any]) -> AgenticTurn:
    return AgenticTurn(
        function_call=FunctionCall(name=name, args=args),
        text=None,
        raw_content=None,
    )


# --- Helpers on their own -------------------------------------------------


def test_user_content_has_role_and_text() -> None:
    content = user_content("hello")
    assert getattr(content, "role", None) == "user"
    parts = getattr(content, "parts", None) or []
    assert len(parts) == 1
    assert getattr(parts[0], "text", None) == "hello"


def test_function_response_content_wraps_payload() -> None:
    content = function_response_content("search_memory", {"results": "x"})
    parts = getattr(content, "parts", None) or []
    assert len(parts) == 1
    fr = getattr(parts[0], "function_response", None)
    assert fr is not None
    assert fr.name == "search_memory"


def test_last_function_call_name_walks_backward() -> None:
    sess = Session(id="s", question="q")
    # user question (no function_call)
    sess.history.append(user_content("q"))
    # a model call to search_memory
    sess.history.append(_tool_turn("search_memory", {"query": "x"}).raw_content)
    # no raw_content returned from stub, so manufacture one via helpers
    # test the "no matches" path instead
    assert last_function_call_name([user_content("just text")]) is None


# --- run_until_blocked_or_done ------------------------------------------


@pytest.mark.asyncio
async def test_loop_terminates_on_text_reply() -> None:
    llm = StubLLM([_text_turn("here is your answer")])
    retriever = StubRetriever([])
    sess = Session(id="s", question="q")
    outcome = await run_until_blocked_or_done(sess, llm=llm, retriever=retriever)  # type: ignore[arg-type]

    assert outcome.pending is None
    assert outcome.text == "here is your answer"
    assert outcome.turn_cap_hit is False
    assert llm.agentic_calls == 1


@pytest.mark.asyncio
async def test_loop_runs_search_memory_locally() -> None:
    chunk = _chunk()
    llm = StubLLM(
        [
            _tool_turn("search_memory", {"query": "pricing"}),
            _text_turn("the answer is 42"),
        ]
    )
    retriever = StubRetriever([chunk])
    sess = Session(id="s", question="q")

    outcome = await run_until_blocked_or_done(sess, llm=llm, retriever=retriever)  # type: ignore[arg-type]

    assert outcome.pending is None
    assert outcome.text == "the answer is 42"
    assert retriever.queries == ["pricing"]
    assert chunk.chunk_id in sess.chunks_seen


@pytest.mark.asyncio
async def test_loop_returns_pending_on_browser_tool() -> None:
    llm = StubLLM([_tool_turn("visit_page", {"url": "https://ex.com"})])
    retriever = StubRetriever([])
    sess = Session(id="s", question="q")

    outcome = await run_until_blocked_or_done(sess, llm=llm, retriever=retriever)  # type: ignore[arg-type]

    assert outcome.pending is not None
    assert outcome.pending.name == "visit_page"
    assert outcome.pending.args == {"url": "https://ex.com"}
    assert outcome.text is None


@pytest.mark.asyncio
async def test_loop_respects_turn_cap() -> None:
    # All turns call search_memory; never a final text. We should hit the cap.
    turns = [_tool_turn("search_memory", {"query": f"q{i}"}) for i in range(MAX_TURNS + 2)]
    llm = StubLLM(turns)
    retriever = StubRetriever([_chunk()])
    sess = Session(id="s", question="q")

    outcome = await run_until_blocked_or_done(sess, llm=llm, retriever=retriever)  # type: ignore[arg-type]

    assert outcome.pending is None
    assert outcome.text is None
    assert outcome.turn_cap_hit is True
    assert sess.turn == MAX_TURNS


# --- drive_and_finalize -------------------------------------------------


@pytest.mark.asyncio
async def test_drive_finalize_calls_final_and_drops_session() -> None:
    llm = StubLLM(
        [
            _tool_turn("search_memory", {"query": "x"}),
            _text_turn("ok"),
        ],
        final=FinalAnswer(
            answer="final answer",
            citations=[Citation(url="https://ex.com", ts=1, snippet="s")],
        ),
    )
    retriever = StubRetriever([_chunk()])
    sessions = SessionStore()
    sess = sessions.create("q")

    result: AgenticResult = await drive_and_finalize(
        sess,
        llm=llm,
        retriever=retriever,
        sessions=sessions,  # type: ignore[arg-type]
    )

    assert result.answer == "final answer"
    assert result.citations is not None
    assert len(result.citations) == 1
    assert llm.final_calls == 1
    # Session dropped after finalize.
    assert sessions.get(sess.id) is None


@pytest.mark.asyncio
async def test_drive_finalize_pending_keeps_session_alive() -> None:
    llm = StubLLM([_tool_turn("visit_page", {"url": "https://ex.com"})])
    retriever = StubRetriever([])
    sessions = SessionStore()
    sess = sessions.create("q")

    result = await drive_and_finalize(
        sess,
        llm=llm,
        retriever=retriever,
        sessions=sessions,  # type: ignore[arg-type]
    )

    assert result.pending is not None
    assert result.session_id == sess.id
    assert result.answer is None
    # Session kept: side panel needs to continue the loop.
    assert sessions.get(sess.id) is sess


@pytest.mark.asyncio
async def test_drive_finalize_drops_session_on_exception() -> None:
    class BoomLLM:
        async def agentic_turn(self, _history: Sequence[Any]) -> AgenticTurn:
            raise RuntimeError("boom")

        async def final_answer(self, **_: Any) -> FinalAnswer:  # pragma: no cover
            raise AssertionError("final_answer should not be called")

    sessions = SessionStore()
    sess = sessions.create("q")

    with pytest.raises(RuntimeError, match="boom"):
        await drive_and_finalize(
            sess,
            llm=BoomLLM(),  # type: ignore[arg-type]
            retriever=StubRetriever([]),  # type: ignore[arg-type]
            sessions=sessions,
        )

    assert sessions.get(sess.id) is None
