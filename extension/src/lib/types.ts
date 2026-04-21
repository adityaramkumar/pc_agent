/** Shapes shared between the service worker, content script, and side panel. */

export type EventType = "page_visit" | "selection" | "form_input";

export interface IngestEvent {
  type: EventType;
  url: string;
  title: string | null;
  text: string | null;
  ts: number;
  meta: Record<string, unknown>;
}

export interface IngestRequest {
  events: IngestEvent[];
}

export interface IngestResponse {
  ingested: number;
  ids: number[];
}
