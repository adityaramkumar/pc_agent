"""Pydantic request/response models for the HTTP layer.

Separating these from `main.py` keeps the routing file short and lets
tests import the models directly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.llm import Citation
from app.store import EventRow


class HealthResponse(BaseModel):
    status: str
    version: str


# --- Ingest ---------------------------------------------------------------


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


# --- Memories -------------------------------------------------------------


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


# --- Query ----------------------------------------------------------------


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
    """Either a final answer OR a pending browser-side tool request.

    If `answer` is set, the loop is done and `citations` is populated.
    If `session_id` + `pending_tool` + `args` are set, the side panel
    needs to execute the tool and POST the result back to /query/continue.
    """

    answer: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)
    session_id: str | None = None
    pending_tool: str | None = None
    args: dict[str, Any] | None = None
