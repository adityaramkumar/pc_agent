/** Tiny HTTP client for the local backend. Single endpoint for now. */

import type { IngestEvent, IngestRequest, IngestResponse } from "./types";

export const BACKEND_BASE = "http://127.0.0.1:8765";

export async function postIngest(events: IngestEvent[]): Promise<IngestResponse | null> {
  if (events.length === 0) return null;
  const body: IngestRequest = { events };
  try {
    const resp = await fetch(`${BACKEND_BASE}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      console.warn("[pc_agent] ingest failed", resp.status, await resp.text());
      return null;
    }
    return (await resp.json()) as IngestResponse;
  } catch (err) {
    // Backend may simply be down — that's fine, we just drop the batch.
    console.debug("[pc_agent] ingest network error", err);
    return null;
  }
}
