"""FastAPI entrypoint for the pc_agent backend.

Real endpoints (`/ingest`, `/query/start`, `/query/continue`, `/memories`) are
added in subsequent commits. For now this is just a health check so CI has
something to import and test.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


app = FastAPI(title="pc_agent", version="0.0.1")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from app import __version__

    return HealthResponse(status="ok", version=__version__)
