/**
 * Service worker.
 *
 * Today this owns:
 *   - Capturing page-visit events from chrome.tabs.onUpdated, deduping
 *     repeats within a short window, and POSTing batches to the local
 *     backend on a 5-second flush interval.
 *   - Opening the side panel when the toolbar action is clicked.
 *
 * The richer capture surface (Readability, selections, non-password form
 * inputs) and the action loop's tool executor land in subsequent commits.
 */

import { postIngest } from "../lib/api";
import type { IngestEvent } from "../lib/types";

const FLUSH_INTERVAL_MS = 5_000;
const DEDUPE_WINDOW_MS = 30_000;
const MAX_BUFFER = 100;

const buffer: IngestEvent[] = [];
const recentByUrl = new Map<string, number>();

function shouldCapture(url: string | undefined): url is string {
  if (!url) return false;
  if (!url.startsWith("http://") && !url.startsWith("https://")) return false;
  return true;
}

function enqueue(event: IngestEvent): void {
  const last = recentByUrl.get(event.url);
  if (last !== undefined && event.ts - last < DEDUPE_WINDOW_MS) return;
  recentByUrl.set(event.url, event.ts);
  buffer.push(event);
  if (buffer.length >= MAX_BUFFER) void flush();
}

async function flush(): Promise<void> {
  if (buffer.length === 0) return;
  const batch = buffer.splice(0, buffer.length);
  await postIngest(batch);
}

function pruneRecent(now: number): void {
  for (const [url, ts] of recentByUrl) {
    if (now - ts > DEDUPE_WINDOW_MS * 2) recentByUrl.delete(url);
  }
}

chrome.runtime.onInstalled.addListener(() => {
  console.log("[pc_agent] installed");
});

chrome.tabs.onUpdated.addListener((_tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete") return;
  if (tab.incognito) return;
  if (!shouldCapture(tab.url)) return;

  const now = Date.now();
  enqueue({
    type: "page_visit",
    url: tab.url,
    title: tab.title ?? null,
    text: null,
    ts: now,
    meta: { tabId: _tabId, windowId: tab.windowId ?? null },
  });
  pruneRecent(now);
});

chrome.alarms.create("pc_agent_flush", {
  periodInMinutes: FLUSH_INTERVAL_MS / 60_000,
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "pc_agent_flush") void flush();
});

chrome.action.onClicked.addListener(async (tab) => {
  if (tab.windowId !== undefined) {
    await chrome.sidePanel.open({ windowId: tab.windowId });
  }
});
