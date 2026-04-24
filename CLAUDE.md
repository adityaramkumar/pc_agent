# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pc_agent` is a personal browser memory system: a Chrome MV3 extension captures pages/selections/form inputs and sends them to a local FastAPI backend, which stores them in SQLite, embeds them with Gemini, and answers natural-language questions via an agentic RAG loop. All data stays local; only question text and short snippets are sent to Gemini.

## Repository Structure

Two independent sub-projects:
- `backend/` — Python/FastAPI server (runs on `localhost:8765`)
- `extension/` — Chrome MV3 extension (TypeScript/React, built with Vite)

```
pc_agent/
├── backend/
│   ├── app/
│   │   ├── main.py        # HTTP routes + FastAPI app factory
│   │   ├── agentic.py     # Agentic loop (run_until_blocked_or_done, drive_and_finalize)
│   │   ├── llm.py         # GeminiClient: embeddings, agentic_turn, final_answer
│   │   ├── rag.py         # Hybrid retrieval: RRF + recency boost
│   │   ├── store.py       # SQLite + sqlite-vec storage layer
│   │   ├── processor.py   # Chunking + batch embedding
│   │   ├── sessions.py    # In-memory session management
│   │   ├── tools.py       # Gemini function-call declarations
│   │   ├── prompts.py     # System prompts + formatters
│   │   ├── schemas.py     # Pydantic request/response models
│   │   ├── config.py      # Pydantic-settings from env/.env
│   │   ├── deps.py        # FastAPI Depends() type aliases
│   │   └── __init__.py    # __version__
│   └── tests/
│       ├── conftest.py            # Per-test DB + isolated FastAPI fixtures
│       ├── test_endpoints.py      # HTTP round-trips
│       ├── test_agentic.py        # Loop with stub LLM/retriever
│       ├── test_rag.py            # RRF + recency (pure functions)
│       ├── test_llm_parsers.py    # Response parsing (no network)
│       ├── test_processor.py      # Chunking algorithm
│       ├── test_store.py          # Store CRUD
│       ├── test_chunks_and_search.py
│       └── test_sessions.py
├── extension/
│   ├── src/
│   │   ├── content/capture.ts     # Content script: page/selection/form capture
│   │   ├── background/sw.ts       # Service worker: batching, tool execution
│   │   ├── panel/
│   │   │   ├── App.tsx            # Tabbed shell (Ask / Activity)
│   │   │   ├── AskTab.tsx         # Query UI + chat history
│   │   │   ├── ActivityTab.tsx    # Settings + memory management
│   │   │   ├── App.css            # Styles
│   │   │   ├── index.html         # Side-panel entry
│   │   │   └── main.tsx           # React DOM render
│   │   └── lib/
│   │       ├── types.ts           # Shared TypeScript types + DEFAULT_BLOCKLIST
│   │       ├── api.ts             # HTTP client (BackendError, postIngest, queryStart…)
│   │       ├── messages.ts        # IPC message protocol + type guards
│   │       ├── storage.ts         # chrome.storage.local wrapper + isHostBlocked
│   │       └── queryLoop.ts       # runQueryWithTools state machine
│   ├── manifest.config.ts         # Chrome MV3 manifest
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── package.json
└── .github/workflows/
    ├── backend.yml    # ruff + mypy + pytest on Python 3.11/3.12
    └── extension.yml  # lint + typecheck + build, uploads dist/
```

## Commands

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

ruff check .              # lint
ruff format --check .     # format check
ruff format .             # auto-format
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
npm run typecheck   # tsc --noEmit
npm run build       # production bundle → dist/
npm run dev         # Vite dev mode (HMR on port 5174)
```

Load `extension/dist/` as an unpacked extension in Chrome (Developer mode → Load unpacked).

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
  /memories            → CRUD on stored events
```

### Backend Module Responsibilities

- **`config.py`** — `Settings` via `pydantic-settings`. Env vars: `GOOGLE_API_KEY`, `LLM_MODEL` (default `gemini-2.5-flash`), `EMBEDDING_MODEL` (default `gemini-embedding-001`), `DB_PATH` (default `~/.pc_agent/memory.db`), `BACKEND_HOST` (default `127.0.0.1`), `BACKEND_PORT` (default `8765`). Singleton via `get_settings()`; `reset_settings_for_tests()` for test isolation.

- **`store.py`** — Thread-safe SQLite wrapper. Schema: `events`, `chunks`, `chunk_vectors` (sqlite-vec virtual table, 768-dim), `chunks_fts` (FTS5 mirror with porter tokenizer). FTS5 kept in sync via SQL triggers on chunks. WAL mode + foreign keys enabled. Prefers `pysqlite3` over stdlib for extension loading on macOS. Per-request connections via context manager. `_sanitize_fts_query()` reduces the FTS5 MATCH query to safe `"token" OR "token"` form (alphanumeric, ≥2 chars). Singletons: `get_store()` / `reset_store_for_tests()`.

- **`processor.py`** — Chunks events into overlapping windows (`CHUNK_SIZE_CHARS=2000`, `CHUNK_OVERLAP_CHARS=200`, `MIN_CHUNK_CHARS=80`), nudging boundaries to sentence breaks (period or whitespace) in the last 10% of the window. Batch-embeds 100 at a time (`EMBED_BATCH_SIZE=100`). Embedding errors are swallowed — chunks remain searchable via FTS5.

- **`rag.py`** — Hybrid retrieval: vector ANN (`VECTOR_K=20`) + FTS5 keyword search (`FTS_K=20`), merged with Reciprocal Rank Fusion (`RRF_C=60`) and a 14-day recency half-life boost (`RECENCY_BOOST=0.2`, `RECENCY_HALFLIFE_DAYS=14`). Top 8 chunks (`FUSED_K=8`) go to the LLM. Vector branch failure degrades gracefully to FTS-only.

- **`llm.py`** — `google-genai` SDK wrapper. Asymmetric embeddings (`RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for queries). Exponential backoff on 429/5xx: initial 1s, max 32s, 5 retries. 60s timeout on all LLM calls. Agentic turns use function-calling mode; final answer uses JSON mode with a structured citation schema. Cannot combine `response_mime_type=application/json` with `tools` — that's why the two call modes are separate methods.

- **`agentic.py`** — Core agentic loop (`run_until_blocked_or_done`). Drives Gemini turns until one of three outcomes: a browser tool is requested (returns `LoopOutcome.pending` to the client for execution), a free-text answer is produced, or `MAX_TURNS=5` is reached. `search_memory` runs server-side; `visit_page`/`extract_from_page` require a client round-trip via `/query/continue`. `drive_and_finalize` orchestrates the loop, calls `llm.final_answer`, deduplicates citations by URL, and always drops the session on completion or error.

- **`sessions.py`** — In-memory session store keyed by UUID hex. Holds Gemini chat history, `chunks_seen` dict (for citations on final answer), turn counter. `SESSION_IDLE_TIMEOUT_SEC=300` (5 min), eviction thread runs every 60s. Singletons: `get_session_store()` / `reset_session_store_for_tests()`.

- **`schemas.py`** — Pydantic request/response DTOs. `QueryResponse` is a discriminated union: either `{answer, citations}` (done) or `{session_id, pending_tool, args}` (browser tool needed). `EventIn` enforces field length limits. `CitationOut.from_model()` / `EventOut.from_row()` are factory classmethods.

- **`deps.py`** — All FastAPI `Depends()` singletons live here as `Annotated` type aliases (`StoreDep`, `ProcessorDep`, `RetrieverDep`, `LLMDep`, `SessionsDep`, `SettingsDep`). Override via `app.dependency_overrides` in tests.

- **`tools.py`** — Three Gemini function declarations: `search_memory` (query), `visit_page` (url, wait_for_selector?), `extract_from_page` (url, what, css_hint?). All bundled into a single `Tool` group via `all_tools()`. `ToolName` is a `StrEnum`.

- **`prompts.py`** — System prompt and per-turn prompt builders. `AGENTIC_SYSTEM_INSTRUCTION` tells the model to always start with `search_memory` before fetching live pages. `FINAL_ANSWER_SYSTEM_INSTRUCTION` enforces citation-backed answers. `format_search_results` truncates chunks to 600 chars for the function_response feedback. `FINAL_ANSWER_SCHEMA` enforces `{answer, citations: [{url, ts, snippet}]}`.

- **`main.py`** — Thin HTTP layer. CORS allows only `chrome-extension://` origins. Ingest queues processing as a `BackgroundTask`. `/query/continue` looks up the last `function_call` name in history to reconstruct the tool response. Startup lifespan logs a warning if `GOOGLE_API_KEY` is missing.

### Extension Module Responsibilities

- **`content/capture.ts`** — Runs at `document_idle` on every page. Three capture surfaces: (1) page extraction via `@mozilla/readability` with `document.body.innerText` fallback; (2) selections debounced 600ms, min 8 chars; (3) form inputs on submit + trailing-edge input (800ms) + blur. Skips incognito (honor pause/blocklist), password/CC/OTP fields (autocomplete tokens + name heuristics), and blocklisted domains. Max text: 50,000 chars.

- **`background/sw.ts`** — Service worker. Receives `CaptureMessage` from content scripts: dedupes (30s window keyed on type+url+text[:64]), buffers up to 100 events, flushes via `chrome.alarms` every 5 seconds. Executes `visit_page`/`extract_from_page` by opening background tabs (15s load timeout), optionally waiting for a CSS selector (250ms polls, 5s timeout), running `pageExtractor()` as an injected script (must be self-contained — no closures), then closing the tab. Enforces pause + blocklist at the SW layer as well (defense in depth).

- **`lib/queryLoop.ts`** — State machine for the query flow: sends `queryStart`, loops up to `MAX_TOOL_HOPS=5` browser tool round-trips via `runBrowserTool()` → `queryContinue()`, calls `onToolStart`/`onToolEnd` hooks for UI status updates. Returns the final `QueryResponse`.

- **`lib/storage.ts`** — `isHostBlocked()` uses suffix-match: `"example.com"` blocks `"www.example.com"` but not `"notexample.com"`. Default blocklist: Google accounts, Microsoft login, 1Password, LastPass, Bitwarden, Bank of America, Chase, Wells Fargo.

- **`panel/`** — React side panel. `AskTab` for queries (chat-like message history, Enter to submit, Shift+Enter for newline, citation list per answer). `ActivityTab` for memory management (pause toggle, domain blocklist CRUD, recent captures with "forget" per item).

### Testing Approach

Tests use a per-test temporary SQLite DB and an isolated FastAPI `TestClient`. Dependency injection is overridden via `app.dependency_overrides`. No mocking of Gemini by default — integration tests that call the real API are skipped unless `GOOGLE_API_KEY` is set. Agentic loop tests use `StubLLM` and `StubRetriever` dataclasses that return pre-programmed sequences without touching the network.

### CI/CD

- **`backend.yml`**: Triggers on push/PR when `backend/**` changes. Matrix: Python 3.11 and 3.12. Steps: `ruff check`, `ruff format --check`, `mypy app`, `pytest -q`.
- **`extension.yml`**: Triggers on push/PR when `extension/**` changes. Node 20. Steps: `npm ci`, `npm run lint`, `npm run typecheck`, `npm run build`. Uploads `dist/` as an artifact (14-day retention).

## Key Conventions

- **Embeddings are asymmetric**: always use `RETRIEVAL_DOCUMENT` task type when storing chunks, `RETRIEVAL_QUERY` when embedding a user question.
- **Browser tools are two-phase**: the backend returns `pending_tool` + `args`; the extension executes the tool and POSTs the result to `/query/continue`. Never try to execute `visit_page`/`extract_from_page` server-side.
- **Session state is ephemeral**: sessions live only in memory and expire after 5 minutes of inactivity. The client must re-start if the session is gone.
- **SQLite writes are single-writer**: use separate connections per request; don't share connections across threads.
- **Ruff double-quote strings**: the formatter enforces double quotes — don't use single quotes in Python.
- **mypy strict on `app/`**: imports of `sqlite_vec` and `google.genai` have `# type: ignore` by convention since they lack stubs.
- **Singleton caching pattern**: every major object (`Store`, `Processor`, `Retriever`, `GeminiClient`, `SessionStore`) has `get_X()` and `reset_X_for_tests()` functions using module-level globals.
- **Graceful degradation**: embedding failures don't block chunking (FTS5 still works); vector search failure falls back to FTS5; ingest network errors in the extension are swallowed silently.
- **Defense in depth in the extension**: both the content script and the service worker enforce pause + blocklist independently. `pageExtractor` passed to `executeScript` must be self-contained with no closures.
- **FTS5 sanitization**: always route user queries through `_sanitize_fts_query()` before passing to FTS5 MATCH; the grammar is picky and raw user input can cause `OperationalError`.
- **Citation deduplication**: `drive_and_finalize` deduplicates citations by URL, preserving first occurrence order.
