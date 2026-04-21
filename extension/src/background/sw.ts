/**
 * Service worker.
 *
 * Two responsibilities today:
 *
 *   1. Receiving capture events from content scripts (page visits,
 *      selections, form inputs), deduping, batching, and POSTing to the
 *      backend on a 5-second flush.
 *   2. Executing the side panel's `visit_page` / `extract_from_page` tool
 *      requests: open a background tab, wait for an optional CSS selector,
 *      pull text out of the page, close the tab, return the result.
 */

import { postIngest } from "../lib/api";
import {
  isCaptureMessage,
  isRunToolMessage,
  type RunToolMessage,
  type RunToolResult,
} from "../lib/messages";
import { isUrlBlocked, loadSettings, watchSettings } from "../lib/storage";
import { DEFAULT_SETTINGS, type IngestEvent, type UserSettings } from "../lib/types";

const FLUSH_INTERVAL_MIN = 5 / 60;
const DEDUPE_WINDOW_MS = 30_000;
const MAX_BUFFER = 100;

const TAB_LOAD_TIMEOUT_MS = 15_000;

const buffer: IngestEvent[] = [];
const recentByKey = new Map<string, number>();

let settings: UserSettings = { ...DEFAULT_SETTINGS };
void loadSettings().then((s) => {
  settings = s;
});
watchSettings((s) => {
  settings = s;
});

// --- Capture pipeline ----------------------------------------------------

function dedupeKey(event: IngestEvent): string {
  if (event.type === "page_visit") return `pv|${event.url}`;
  return `${event.type}|${event.url}|${event.text?.slice(0, 64) ?? ""}`;
}

function pruneRecent(now: number): void {
  for (const [key, ts] of recentByKey) {
    if (now - ts > DEDUPE_WINDOW_MS * 2) recentByKey.delete(key);
  }
}

function enqueue(event: IngestEvent): void {
  // Defense in depth: the content script also checks these, but the SW is the
  // last line before the network. Pause / blocklist apply here too.
  if (settings.paused) return;
  if (isUrlBlocked(event.url, settings.blocklist)) return;

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

// --- Tool executor (visit_page / extract_from_page) ---------------------

interface ExtractArgs {
  waitForSelector: string | null;
  cssHint: string | null;
}

/**
 * This function runs in the page's content world via chrome.scripting.executeScript.
 * Must be self-contained (no closures, no imports).
 */
function pageExtractor(args: ExtractArgs): { text: string | null } {
  const POLL_INTERVAL = 250;
  const MAX_WAIT = 5_000;
  const MAX_LEN = 50_000;

  function pickText(): string | null {
    if (args.cssHint) {
      const el = document.querySelector(args.cssHint);
      if (el instanceof HTMLElement) return el.innerText;
      if (el != null) return el.textContent;
      return null;
    }
    return document.body?.innerText ?? null;
  }

  async function waitForSelector(): Promise<boolean> {
    if (!args.waitForSelector) return true;
    const start = Date.now();
    while (Date.now() - start < MAX_WAIT) {
      if (document.querySelector(args.waitForSelector)) return true;
      await new Promise((r) => setTimeout(r, POLL_INTERVAL));
    }
    return false;
  }

  // executeScript returns the resolved value of an async IIFE.
  return waitForSelector().then(() => {
    const raw = pickText();
    const text = raw ? raw.slice(0, MAX_LEN) : null;
    return { text };
  }) as unknown as { text: string | null };
}

async function waitForTabComplete(tabId: number, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      chrome.tabs.onUpdated.removeListener(listener);
      reject(new Error(`tab ${tabId} did not finish loading within ${timeoutMs}ms`));
    }, timeoutMs);
    const listener = (
      updatedId: number,
      changeInfo: chrome.tabs.TabChangeInfo,
    ): void => {
      if (updatedId === tabId && changeInfo.status === "complete") {
        clearTimeout(timer);
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    };
    chrome.tabs.onUpdated.addListener(listener);
  });
}

async function runTool(message: RunToolMessage): Promise<RunToolResult> {
  const { tool, args } = message;
  const url = args.url;
  if (!url) return { ok: false, error: "missing url" };
  if (isUrlBlocked(url, settings.blocklist)) {
    return { ok: false, url, error: "url is on the blocklist; refusing to visit" };
  }

  let tab: chrome.tabs.Tab | null = null;
  try {
    tab = await chrome.tabs.create({ active: false, url });
    if (tab?.id == null) return { ok: false, error: "failed to open tab" };
    await waitForTabComplete(tab.id, TAB_LOAD_TIMEOUT_MS);

    const extractArgs: ExtractArgs = {
      waitForSelector: args.wait_for_selector ?? null,
      cssHint: tool === "extract_from_page" ? (args.css_hint ?? null) : null,
    };
    const [result] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: pageExtractor,
      args: [extractArgs],
    });

    const payload = result?.result as { text: string | null } | undefined;
    const text = payload?.text ?? null;
    return {
      ok: text !== null,
      url,
      text: text ?? undefined,
      error: text === null ? "extraction returned no text" : undefined,
    };
  } catch (err) {
    return {
      ok: false,
      url,
      error: err instanceof Error ? err.message : String(err),
    };
  } finally {
    if (tab?.id != null) {
      try {
        await chrome.tabs.remove(tab.id);
      } catch {
        /* tab might already be gone */
      }
    }
  }
}

// --- Wiring -------------------------------------------------------------

chrome.runtime.onInstalled.addListener(() => {
  console.log("[pc_agent] installed");
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (isCaptureMessage(message)) {
    enqueue(message.event);
    pruneRecent(Date.now());
    sendResponse({ ok: true });
    return false;
  }

  if (isRunToolMessage(message)) {
    runTool(message).then(sendResponse);
    return true; // async response
  }

  return false;
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
