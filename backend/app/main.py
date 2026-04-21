"""FastAPI entrypoint for the pc_agent backend.

Currently exposes:
- GET    /health          — readiness check
- POST   /ingest          — batch capture from the extension
- GET    /memories        — list recent events for the Activity tab
- GET    /memories/{id}   — fetch a single event
- DELETE /memories/{id}   — forget a single event (cascades to chunks)

The `/query/start` and `/query/continue` endpoints land in the
memory_indexing / action_loop steps.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from app import __version__
from app.config import Settings, get_settings
from app.llm import (
    AgenticTurn,
    Citation,
    FinalAnswer,
    FunctionCall,
    GeminiClient,
    get_gemini_client,
)
from app.processor import Processor, get_processor
from app.rag import Retriever, get_retriever
from app.sessions import MAX_TURNS, Session, SessionStore, get_session_store
from app.store import EventRow, IngestEvent, RetrievedChunk, Store, get_store

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    version: str


class EventIn(BaseModel):
    type: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=1, max_length=4096)
    title: str | None = Field(default=None, max_length=2048)
    text: str | None = None
    ts: int = Field(ge=0)
    meta: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    events: list[EventIn] = Field(default_factory=list)


class IngestResponse(BaseModel):
    ingested: int
    ids: list[int]


class EventOut(BaseModel):
    id: int
    type: str
    url: str
    title: str | None
    text: str | None
    ts: int
    meta: dict[str, Any]

    @classmethod
    def from_row(cls, row: EventRow) -> EventOut:
        return cls(
            id=row.id,
            type=row.type,
            url=row.url,
            title=row.title,
            text=row.text,
            ts=row.ts,
            meta=row.meta,
        )


class MemoriesResponse(BaseModel):
    total: int
    events: list[EventOut]


class CitationOut(BaseModel):
    url: str
    ts: int
    snippet: str

    @classmethod
    def from_model(cls, c: Citation) -> CitationOut:
        return cls(url=c.url, ts=c.ts, snippet=c.snippet)


class QueryStartRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class QueryContinueRequest(BaseModel):
    session_id: str = Field(min_length=1)
    tool_result: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Side-panel response shape.

    Either:
      - `answer` + `citations` populated when the agentic loop is done; or
      - `session_id` + `pending_tool` + `args` populated when the side panel
        needs to execute a browser-side tool (visit_page / extract_from_page).
    """

    answer: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)
    session_id: str | None = None
    pending_tool: str | None = None
    args: dict[str, Any] | None = None


def _store_dep(settings: Annotated[Settings, Depends(get_settings)]) -> Store:
    return get_store(settings)


def _processor_dep() -> Processor:
    return get_processor()


def _retriever_dep() -> Retriever:
    return get_retriever()


def _llm_dep() -> GeminiClient:
    return get_gemini_client()


def _sessions_dep() -> SessionStore:
    return get_session_store()


SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[Store, Depends(_store_dep)]
ProcessorDep = Annotated[Processor, Depends(_processor_dep)]
RetrieverDep = Annotated[Retriever, Depends(_retriever_dep)]
LLMDep = Annotated[GeminiClient, Depends(_llm_dep)]
SessionsDep = Annotated[SessionStore, Depends(_sessions_dep)]


async def _process_event_safely(processor: Processor, event_id: int) -> None:
    """Background task wrapper that never lets ingest fail because of LLM errors."""
    try:
        await processor.process_event(event_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("background processing failed for event %d: %s", event_id, exc)


# --- Agentic loop helpers ------------------------------------------------


def _format_search_results(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no matches)"
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        title = c.title or c.url
        text = c.text[:600] + ("..." if len(c.text) > 600 else "")
        lines.append(
            f"[{i}] chunk_id={c.chunk_id} url={c.url} ts={c.ts}\n"
            f"    title: {title}\n"
            f"    text:  {text}"
        )
    return "\n\n".join(lines)


async def _run_local_search(sess: Session, retriever: Retriever, args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "(empty query)"
    chunks = await retriever.search(query)
    for c in chunks:
        sess.chunks_seen[c.chunk_id] = c
    return _format_search_results(chunks)


def _user_content(text: str) -> Any:
    return genai_types.Content(role="user", parts=[genai_types.Part(text=text)])


def _function_response_content(name: str, payload: dict[str, Any]) -> Any:
    return genai_types.Content(
        role="user",
        parts=[
            genai_types.Part.from_function_response(name=name, response=payload),
        ],
    )


async def _step_loop(
    sess: Session,
    *,
    llm: GeminiClient,
    retriever: Retriever,
) -> tuple[FunctionCall | None, str | None]:
    """Drive the agentic loop until we either need a browser-side tool
    (returns the FunctionCall) or the model gives a final text response."""
    while sess.turn < MAX_TURNS:
        sess.turn += 1
        turn: AgenticTurn = await llm.agentic_turn(sess.history)
        if turn.raw_content is not None:
            sess.history.append(turn.raw_content)

        fn = turn.function_call
        if fn is None:
            return None, turn.text

        if fn.name == "search_memory":
            result_text = await _run_local_search(sess, retriever, fn.args)
            sess.history.append(
                _function_response_content("search_memory", {"results": result_text})
            )
            continue

        # visit_page / extract_from_page must be executed by the side panel.
        return fn, None

    # Cap hit; force a final text from whatever's been said.
    return None, None


async def _finalize(sess: Session, *, llm: GeminiClient) -> QueryResponse:
    chunks = list(sess.chunks_seen.values())
    final: FinalAnswer = await llm.final_answer(question=sess.question, retrieved=chunks)
    return QueryResponse(
        answer=final.answer,
        citations=[CitationOut.from_model(c) for c in final.citations],
    )


async def _step_and_respond(
    sess: Session,
    *,
    llm: GeminiClient,
    retriever: Retriever,
    sessions: SessionStore,
) -> QueryResponse:
    """Drive the loop, then either return a pending tool or the final answer."""
    try:
        pending, _ = await _step_loop(sess, llm=llm, retriever=retriever)
    except Exception as exc:
        sessions.drop(sess.id)
        logger.exception("agentic loop failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM call failed; check backend logs",
        ) from exc

    if pending is not None:
        return QueryResponse(session_id=sess.id, pending_tool=pending.name, args=pending.args)

    try:
        return await _finalize(sess, llm=llm)
    finally:
        sessions.drop(sess.id)


def _last_function_call_name(history: list[Any]) -> str | None:
    for content in reversed(history):
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                return str(fc.name)
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="pc_agent", version=__version__)

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^chrome-extension://.*$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        body: IngestRequest,
        background: BackgroundTasks,
        store: StoreDep,
        processor: ProcessorDep,
    ) -> IngestResponse:
        events = [
            IngestEvent(
                type=e.type,
                url=e.url,
                title=e.title,
                text=e.text,
                ts=e.ts,
                meta=e.meta,
            )
            for e in body.events
        ]
        ids = store.insert_events(events)
        # Queue chunking + embedding so the request returns immediately;
        # only events that actually carry text are worth processing.
        for ev_id, ev in zip(ids, events, strict=True):
            if ev.text:
                background.add_task(_process_event_safely, processor, ev_id)
        return IngestResponse(ingested=len(ids), ids=ids)

    @app.post("/query/start", response_model=QueryResponse)
    async def query_start(
        body: QueryStartRequest,
        retriever: RetrieverDep,
        llm: LLMDep,
        sessions: SessionsDep,
    ) -> QueryResponse:
        sess = sessions.create(body.question)
        sess.history.append(_user_content(body.question))
        return await _step_and_respond(sess, llm=llm, retriever=retriever, sessions=sessions)

    @app.post("/query/continue", response_model=QueryResponse)
    async def query_continue(
        body: QueryContinueRequest,
        retriever: RetrieverDep,
        llm: LLMDep,
        sessions: SessionsDep,
    ) -> QueryResponse:
        sess = sessions.get(body.session_id)
        if sess is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="session not found or expired",
            )
        last_tool_name = _last_function_call_name(sess.history) or "unknown_tool"
        sess.history.append(_function_response_content(last_tool_name, body.tool_result))
        return await _step_and_respond(sess, llm=llm, retriever=retriever, sessions=sessions)

    @app.get("/memories", response_model=MemoriesResponse)
    async def list_memories(
        store: StoreDep,
        limit: int = 100,
        offset: int = 0,
    ) -> MemoriesResponse:
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit must be in [1, 1000]",
            )
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="offset must be >= 0",
            )
        rows = store.list_events(limit=limit, offset=offset)
        total = store.count_events()
        return MemoriesResponse(
            total=total,
            events=[EventOut.from_row(r) for r in rows],
        )

    @app.get("/memories/{event_id}", response_model=EventOut)
    async def get_memory(event_id: int, store: StoreDep) -> EventOut:
        row = store.get_event(event_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="event not found")
        return EventOut.from_row(row)

    @app.delete("/memories/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_memory(event_id: int, store: StoreDep) -> None:
        deleted = store.delete_event(event_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="event not found")
        return None

    return app


app = create_app()
