"""Shared pytest fixtures: per-test SQLite DB and isolated FastAPI app."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import _store_dep, create_app
from app.store import Store


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        GOOGLE_API_KEY="test-key",
        DB_PATH=tmp_path / "memory.db",
    )


@pytest.fixture()
def store(settings: Settings) -> Store:
    return Store(settings.db_path)


@pytest.fixture()
def app(store: Store) -> Iterator[FastAPI]:
    app = create_app()
    app.dependency_overrides[_store_dep] = lambda: store
    yield app
    app.dependency_overrides.clear()


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
