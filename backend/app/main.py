"""FastAPI entrypoint: thin HTTP layer on top of the other modules.

Routes:
- GET    /health            readiness check
- POST   /ingest            batch capture from the extension
- GET    /memories          list recent events
- GET    /memories/{id}     fetch a single event
- DELETE /memories/{id}     forget a single event (cascades to chunks)
- POST   /query/start       kick off an agentic session
- POST   /query/continue    resume an agentic session with a tool result

Loop orchestration lives in `agentic.py`; storage in `store.py`; chunking
and embedding in `processor.py`; retrieval in `rag.py`; the Gemini client
in `llm.py`; prompts in `prompts.py`; tool declarations in `tools.py`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.agentic import (
    AgenticResult,
    drive_and_finalize,
    function_response_content,
    last_function_call_name,
    user_content,
)
from app.config import get_settings
from app.deps import LLMDep, ProcessorDep, RetrieverDep, SessionsDep, StoreDep
from app.processor import Processor
from app.schemas import (
    CitationOut,
    EventOut,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MemoriesResponse,
    QueryContinueRequest,
    QueryResponse,
    QueryStartRequest,
)
from app.store import IngestEvent

logger = logging.getLogger(__name__)


async def _process_event_safely(processor: Processor, event_id: int) -> None:
    """Background-task wrapper that never lets ingest fail because of LLM errors."""
    try:
        await processor.process_event(event_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("background processing failed for event %d: %s", event_id, exc)


def _result_to_response(result: AgenticResult) -> QueryResponse:
    if result.pending is not None:
        return QueryResponse(
            session_id=result.session_id,
            pending_tool=result.pending.name,
            args=result.pending.args,
        )
    return QueryResponse(
        answer=result.answer,
        citations=[CitationOut.from_model(c) for c in (result.citations or [])],
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if not settings.google_api_key:
        logger.error(
            "GOOGLE_API_KEY is not set. Set it in backend/.env or as an environment variable. "
            "All LLM and embedding calls will fail until it is provided."
        )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="pc_agent", version=__version__, lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^chrome-extension://.*$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    # --- Health ----------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    # --- Ingest ----------------------------------------------------------

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
        # Queue chunking + embedding so the request returns immediately.
        # Only events that actually carry text are worth processing.
        for ev_id, ev in zip(ids, events, strict=True):
            if ev.text:
                background.add_task(_process_event_safely, processor, ev_id)
        return IngestResponse(ingested=len(ids), ids=ids)

    # --- Query -----------------------------------------------------------

    @app.post("/query/start", response_model=QueryResponse)
    async def query_start(
        body: QueryStartRequest,
        retriever: RetrieverDep,
        llm: LLMDep,
        sessions: SessionsDep,
    ) -> QueryResponse:
        sess = sessions.create(body.question)
        sess.history.append(user_content(body.question))
        try:
            result = await drive_and_finalize(sess, llm=llm, retriever=retriever, sessions=sessions)
        except Exception as exc:
            logger.exception("agentic loop failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM call failed; check backend logs",
            ) from exc
        return _result_to_response(result)

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
        tool_name = last_function_call_name(sess.history) or "unknown_tool"
        sess.history.append(function_response_content(tool_name, body.tool_result))
        try:
            result = await drive_and_finalize(sess, llm=llm, retriever=retriever, sessions=sessions)
        except Exception as exc:
            logger.exception("agentic loop failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM call failed; check backend logs",
            ) from exc
        return _result_to_response(result)

    # --- Memories --------------------------------------------------------

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
