"""FastAPI dependency factories and type aliases.

Keeping `Annotated[X, Depends(...)]` aliases in one place makes route
signatures short and readable.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.llm import GeminiClient, get_gemini_client
from app.processor import Processor, get_processor
from app.rag import Retriever, get_retriever
from app.sessions import SessionStore, get_session_store
from app.store import Store, get_store


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

# Export the underlying factory functions so tests can override them.
__all__ = [
    "LLMDep",
    "ProcessorDep",
    "RetrieverDep",
    "SessionsDep",
    "SettingsDep",
    "StoreDep",
    "_llm_dep",
    "_processor_dep",
    "_retriever_dep",
    "_sessions_dep",
    "_store_dep",
]
