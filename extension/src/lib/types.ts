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

export interface Citation {
  url: string;
  ts: number;
  snippet: string;
}

export interface QueryResponse {
  answer: string | null;
  citations: Citation[];
  session_id: string | null;
  pending_tool: string | null;
  args: Record<string, unknown> | null;
}

export interface MemoryEvent {
  id: number;
  type: EventType;
  url: string;
  title: string | null;
  text: string | null;
  ts: number;
  meta: Record<string, unknown>;
}

export interface MemoriesResponse {
  total: number;
  events: MemoryEvent[];
}

export interface UserSettings {
  paused: boolean;
  blocklist: string[];
}

export const DEFAULT_BLOCKLIST: string[] = [
  "accounts.google.com",
  "login.microsoftonline.com",
  "1password.com",
  "lastpass.com",
  "bitwarden.com",
  "bankofamerica.com",
  "chase.com",
  "wellsfargo.com",
];

export const DEFAULT_SETTINGS: UserSettings = {
  paused: false,
  blocklist: DEFAULT_BLOCKLIST,
};
