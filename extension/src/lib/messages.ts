/** Message protocol between content script and service worker. */

import type { IngestEvent } from "./types";

export interface CaptureMessage {
  kind: "capture";
  event: IngestEvent;
}

export type ExtensionMessage = CaptureMessage;

export function isExtensionMessage(value: unknown): value is ExtensionMessage {
  if (!value || typeof value !== "object") return false;
  const candidate = value as { kind?: unknown };
  return candidate.kind === "capture";
}
