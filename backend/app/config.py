"""Settings loaded from environment / .env file."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_db_path() -> Path:
    return Path("~/.pc_agent/memory.db").expanduser()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("PC_AGENT_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    llm_model: str = Field(default="gemini-2.5-flash", alias="LLM_MODEL")
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    db_path: Path = Field(default_factory=_default_db_path, alias="DB_PATH")
    backend_host: str = Field(default="127.0.0.1", alias="BACKEND_HOST")
    backend_port: int = Field(default=8765, alias="BACKEND_PORT")

    embedding_dim: int = 768


_cached: Settings | None = None


def get_settings() -> Settings:
    global _cached
    if _cached is None:
        _cached = Settings()
    return _cached


def reset_settings_for_tests() -> None:
    """Used by tests to force a re-read of the environment."""
    global _cached
    _cached = None
