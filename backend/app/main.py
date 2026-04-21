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
from pydantic import BaseModel, Field

from app import __version__
from app.config import Settings, get_settings
from app.llm import Citation, FinalAnswer
from app.processor import Processor, get_processor
from app.rag import Retriever, get_retriever
from app.store import EventRow, IngestEvent, Store, get_store

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


class QueryResponse(BaseModel):
    """Side-panel response shape.

    `answer` and `citations` are populated when the agentic loop is done.
    `pending_tool` / `args` / `session_id` are reserved for the action_loop
    step (memory_indexing always returns a final answer in one shot).
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


SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[Store, Depends(_store_dep)]
ProcessorDep = Annotated[Processor, Depends(_processor_dep)]
RetrieverDep = Annotated[Retriever, Depends(_retriever_dep)]


async def _process_event_safely(processor: Processor, event_id: int) -> None:
    """Background task wrapper that never lets ingest fail because of LLM errors."""
    try:
        await processor.process_event(event_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("background processing failed for event %d: %s", event_id, exc)


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
    async def query_start(body: QueryStartRequest, retriever: RetrieverDep) -> QueryResponse:
        retrieved = await retriever.search(body.question)
        client = retriever._client
        try:
            final: FinalAnswer = await client.final_answer(
                question=body.question, retrieved=retrieved
            )
        except Exception as exc:
            logger.exception("LLM final_answer failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM call failed; check backend logs",
            ) from exc
        return QueryResponse(
            answer=final.answer,
            citations=[CitationOut.from_model(c) for c in final.citations],
        )

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
