/** Message protocol between content script, service worker, and side panel. */

import type { IngestEvent } from "./types";

export interface CaptureMessage {
  kind: "capture";
  event: IngestEvent;
}

export interface RunToolMessage {
  kind: "run_tool";
  tool: "visit_page" | "extract_from_page";
  args: {
    url?: string;
    wait_for_selector?: string;
    what?: string;
    css_hint?: string;
  };
}

export interface RunToolResult {
  ok: boolean;
  url?: string;
  text?: string;
  error?: string;
}

export type ExtensionMessage = CaptureMessage | RunToolMessage;

export function isCaptureMessage(value: unknown): value is CaptureMessage {
  if (!value || typeof value !== "object") return false;
  return (value as { kind?: unknown }).kind === "capture";
}

export function isRunToolMessage(value: unknown): value is RunToolMessage {
  if (!value || typeof value !== "object") return false;
  return (value as { kind?: unknown }).kind === "run_tool";
}
