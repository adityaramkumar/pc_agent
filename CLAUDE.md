# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pc_agent` is a personal browser memory system: a Chrome MV3 extension captures pages/selections/form inputs and sends them to a local FastAPI backend, which stores them in SQLite, embeds them with Gemini, and answers natural-language questions via an agentic RAG loop. All data stays local; only question text and short snippets are sent to Gemini.

## Repository Structure

Two independent sub-projects:
- `backend/` — Python/FastAPI server (runs on `localhost:8765`)
- `extension/` — Chrome MV3 extension (TypeScript/React, built with Vite)

## Commands

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

ruff check .              # lint
ruff format --check .     # format check
mypy app                  # type check
pytest -q                 # all tests
pytest -q tests/test_rag.py                    # single file
pytest -q tests/test_rag.py::test_rrf_fusion   # single test
```

Copy `backend/.env.example` to `backend/.env` and set `GOOGLE_API_KEY` before running the server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload
```

### Extension

```bash
cd extension
npm install
npm run lint        # ESLint
npm run typecheck   # tsc
npm run build       # production bundle → dist/
npm run dev         # Vite dev mode (HMR)
```

Load `extension/dist/` as an unpacked extension in Chrome.

## Architecture

### Data Flow

```
Chrome Extension
  content/capture.ts   → extracts page via Readability, tracks selections/forms
  background/sw.ts     → batches events, POSTs to /ingest, executes browser tools
  panel/               → React UI, runs query loop, handles pending tool callbacks
        ↕ HTTP localhost:8765
FastAPI Backend
  /ingest              → stores raw events, queues embedding
  /query/start         → starts agentic session
  /query/continue      → resumes after browser tool completes
  /memory              → CRUD on stored events
```

### Backend Module Responsibilities

- **`agentic.py`** — Core agentic loop. Drives Gemini turns until one of three outcomes: a browser tool is requested (returns `LoopOutcome.pending` to the client for execution), a free-text answer is produced, or `MAX_TURNS=5` is reached. `search_memory` runs server-side; `visit_page`/`extract_from_page` require a client round-trip via `/query/continue`.
- **`rag.py`** — Hybrid retrieval: vector ANN (sqlite-vec) + FTS5 keyword search, merged with Reciprocal Rank Fusion and a 14-day recency half-life boost. Top-8 chunks go to the LLM.
- **`processor.py`** — Chunks events into 2000-char overlapping windows (200-char overlap, nudged to sentence boundaries), then batch-embeds (100 at a time) with exponential backoff.
- **`store.py`** — Thread-safe SQLite wrapper. Schema: `events`, `chunks`, `chunk_vectors` (sqlite-vec virtual table, 768-dim), `chunks_fts` (FTS5 mirror). Prefers `pysqlite3` over stdlib for extension loading on macOS.
- **`llm.py`** — `google-genai` SDK wrapper. Asymmetric embeddings (RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for queries). Agentic turns use function-calling mode; final answer uses JSON mode with a structured citation schema. Retries on 429 and 5xx.
- **`sessions.py`** — In-memory session store keyed by UUID. Holds Gemini chat history, `chunks_seen` for citations, turn counter, 5-min idle eviction.
- **`deps.py`** — All FastAPI `Depends()` singletons live here. Override via `app.dependency_overrides` in tests.
- **`tools.py`** — Three Gemini function declarations: `search_memory`, `visit_page`, `extract_from_page`.
- **`prompts.py`** — System prompt and per-turn prompt builders.

### Extension Module Responsibilities

- **`content/capture.ts`** — Runs at `document_idle` on every page. Extracts main content via `@mozilla/readability`. Skips incognito, password/CC/OTP fields, and blocklisted domains (banks, password managers, Google auth).
- **`background/sw.ts`** — Service worker. Receives capture events, batches to `/ingest`, and executes `visit_page`/`extract_from_page` tools by opening background tabs and scraping.
- **`lib/queryLoop.ts`** — State machine for the query flow: idle → loading → pending_tool → done/error. Handles the two-part round-trip when the agentic loop needs a browser tool.
- **`panel/`** — React side panel. `AskTab` for queries, `ActivityTab` for memory management and settings.

### Testing Approach

Tests use a per-test temporary SQLite DB and an isolated FastAPI `TestClient`. Dependency injection is overridden via `app.dependency_overrides`. No mocking of Gemini by default — integration tests that call the real API are skipped unless `GOOGLE_API_KEY` is set.

## Key Conventions

- **Embeddings are asymmetric**: always use `RETRIEVAL_DOCUMENT` task type when storing chunks, `RETRIEVAL_QUERY` when embedding a user question.
- **Browser tools are two-phase**: the backend returns `pending_tool` + `args`; the extension executes the tool and POSTs the result to `/query/continue`. Never try to execute `visit_page`/`extract_from_page` server-side.
- **Session state is ephemeral**: sessions live only in memory and expire after 5 minutes of inactivity. The client must re-start if the session is gone.
- **SQLite writes are single-writer**: use separate connections per request; don't share connections across threads.
- **Ruff double-quote strings**: the formatter enforces double quotes — don't use single quotes in Python.
- **mypy strict on `app/`**: imports of `sqlite_vec` and `google.genai` have `# type: ignore` by convention since they lack stubs.
