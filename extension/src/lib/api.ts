/** Tiny HTTP client for the local backend. */

import type {
  IngestEvent,
  IngestRequest,
  IngestResponse,
  MemoriesResponse,
  QueryResponse,
} from "./types";

export const BACKEND_BASE = "http://127.0.0.1:8765";

export class BackendError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "BackendError";
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { allowStatuses?: number[] } = {},
): Promise<T> {
  const { allowStatuses, ...rest } = init;
  const resp = await fetch(`${BACKEND_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(rest.headers ?? {}) },
    ...rest,
  });
  if (!resp.ok && !(allowStatuses ?? []).includes(resp.status)) {
    const body = await resp.text().catch(() => "");
    throw new BackendError(resp.status, body || `${resp.status} ${resp.statusText}`);
  }
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

export async function postIngest(events: IngestEvent[]): Promise<IngestResponse | null> {
  if (events.length === 0) return null;
  const body: IngestRequest = { events };
  try {
    return await request<IngestResponse>("/ingest", {
      method: "POST",
      body: JSON.stringify(body),
    });
  } catch (err) {
    // Backend may simply be down — that's fine, we just drop the batch.
    console.debug("[pc_agent] ingest network error", err);
    return null;
  }
}

export async function queryStart(question: string): Promise<QueryResponse> {
  return request<QueryResponse>("/query/start", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export async function listMemories(
  limit = 50,
  offset = 0,
): Promise<MemoriesResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return request<MemoriesResponse>(`/memories?${params}`);
}

export async function deleteMemory(id: number): Promise<void> {
  await request<void>(`/memories/${id}`, { method: "DELETE" });
}
