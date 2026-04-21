/**
 * Service worker.
 *
 * Today this owns:
 *   - Receiving capture events from the in-page content script (page
 *     visits, selections, form inputs), deduping near-duplicates within
 *     a short window, and POSTing batches to the local backend on a 5s
 *     flush interval.
 *   - Opening the side panel when the toolbar action is clicked.
 *
 * The action loop's tool executor (`visit_page` / `extract_from_page`) lands
 * in the `action_loop` step and will piggyback on this same message channel.
 */

import { postIngest } from "../lib/api";
import { isExtensionMessage } from "../lib/messages";
import type { IngestEvent } from "../lib/types";

const FLUSH_INTERVAL_MIN = 5 / 60; // chrome.alarms wants minutes; 5 seconds
const DEDUPE_WINDOW_MS = 30_000;
const MAX_BUFFER = 100;

const buffer: IngestEvent[] = [];
const recentByKey = new Map<string, number>();

function dedupeKey(event: IngestEvent): string {
  // Page visits dedupe on URL alone; richer events dedupe on URL + a content
  // hash so e.g. multiple distinct selections on the same page all flow.
  if (event.type === "page_visit") return `pv|${event.url}`;
  return `${event.type}|${event.url}|${event.text?.slice(0, 64) ?? ""}`;
}

function pruneRecent(now: number): void {
  for (const [key, ts] of recentByKey) {
    if (now - ts > DEDUPE_WINDOW_MS * 2) recentByKey.delete(key);
  }
}

function enqueue(event: IngestEvent): void {
  const key = dedupeKey(event);
  const last = recentByKey.get(key);
  if (last !== undefined && event.ts - last < DEDUPE_WINDOW_MS) return;
  recentByKey.set(key, event.ts);
  buffer.push(event);
  if (buffer.length >= MAX_BUFFER) void flush();
}

async function flush(): Promise<void> {
  if (buffer.length === 0) return;
  const batch = buffer.splice(0, buffer.length);
  await postIngest(batch);
}

chrome.runtime.onInstalled.addListener(() => {
  console.log("[pc_agent] installed");
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!isExtensionMessage(message)) return false;
  enqueue(message.event);
  pruneRecent(Date.now());
  sendResponse({ ok: true });
  return false; // synchronous response
});

chrome.alarms.create("pc_agent_flush", { periodInMinutes: FLUSH_INTERVAL_MIN });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "pc_agent_flush") void flush();
});

chrome.action.onClicked.addListener(async (tab) => {
  if (tab.windowId !== undefined) {
    await chrome.sidePanel.open({ windowId: tab.windowId });
  }
});
