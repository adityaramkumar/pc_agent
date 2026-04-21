/**
 * Content script. Three capture surfaces:
 *
 *   1. Page extraction — Mozilla Readability on document_idle. Falls back
 *      to title + visible body text if Readability declines (which happens
 *      on most SPAs and short pages).
 *   2. Selections — `selectionchange` debounced; sends a `selection` event
 *      whenever the user finishes highlighting non-trivial text.
 *   3. Form inputs — captures non-password fields on form submit and
 *      debounced trailing-edge `input` (so we get search queries, message
 *      drafts, etc., but never passwords / cc numbers / OTP codes).
 *
 * All events are forwarded to the service worker via runtime.sendMessage;
 * the SW handles dedupe, batching, and the actual POST to the backend.
 */

import { Readability, isProbablyReaderable } from "@mozilla/readability";

import type { CaptureMessage } from "../lib/messages";
import type { EventType, IngestEvent } from "../lib/types";

const SELECTION_DEBOUNCE_MS = 600;
const INPUT_DEBOUNCE_MS = 800;
const MIN_SELECTION_LEN = 8;
const MIN_INPUT_LEN = 2;
const MAX_TEXT_LEN = 50_000;

const SENSITIVE_AUTOCOMPLETE = new Set([
  "current-password",
  "new-password",
  "one-time-code",
  "cc-number",
  "cc-csc",
  "cc-exp",
  "cc-exp-month",
  "cc-exp-year",
]);

function send(event: IngestEvent): void {
  const message: CaptureMessage = { kind: "capture", event };
  // The SW may not be alive; sendMessage swallows that case for us. We catch
  // any thrown errors so the page never sees a violation.
  try {
    chrome.runtime.sendMessage(message).catch(() => {
      /* SW asleep or extension reloading; drop silently */
    });
  } catch {
    /* same */
  }
}

function makeEvent(
  type: EventType,
  text: string | null,
  meta: Record<string, unknown> = {},
): IngestEvent {
  return {
    type,
    url: location.href,
    title: document.title || null,
    text: text ? text.slice(0, MAX_TEXT_LEN) : null,
    ts: Date.now(),
    meta,
  };
}

// --- 1. Page extraction --------------------------------------------------

function extractPage(): string | null {
  // Readability mutates the document it's given, so always work on a clone.
  const cloned = document.cloneNode(true) as Document;
  if (isProbablyReaderable(cloned)) {
    const article = new Readability(cloned).parse();
    if (article?.textContent) return article.textContent.trim();
  }
  // Fallback: visible body text. Cheap, often noisy, but better than nothing
  // for SPAs and tiny pages.
  return document.body?.innerText?.trim() || null;
}

function emitPageVisit(): void {
  const text = extractPage();
  send(
    makeEvent("page_visit", text, {
      readerable: text != null && text.length > 200,
    }),
  );
}

// Page is already idle by the time content_scripts:run_at=document_idle fires,
// but defer one frame so SPA shells finish their initial mount.
requestAnimationFrame(() => emitPageVisit());

// --- 2. Selections -------------------------------------------------------

let selectionTimer: number | undefined;

document.addEventListener("selectionchange", () => {
  if (selectionTimer !== undefined) clearTimeout(selectionTimer);
  selectionTimer = window.setTimeout(() => {
    const sel = document.getSelection();
    const text = sel?.toString().trim() ?? "";
    if (text.length < MIN_SELECTION_LEN) return;
    send(makeEvent("selection", text));
  }, SELECTION_DEBOUNCE_MS);
});

// --- 3. Form inputs ------------------------------------------------------

type AnyField = HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement;

function isSensitive(el: AnyField): boolean {
  if (el instanceof HTMLInputElement) {
    if (el.type === "password" || el.type === "hidden" || el.type === "file") {
      return true;
    }
  }
  const autocomplete = el.getAttribute("autocomplete")?.toLowerCase() ?? "";
  for (const token of autocomplete.split(/\s+/)) {
    if (SENSITIVE_AUTOCOMPLETE.has(token)) return true;
  }
  // Heuristics: ARIA roles or names that suggest credentials/2FA.
  const name = (el.getAttribute("name") || el.id || "").toLowerCase();
  if (/(password|passwd|otp|cvv|cvc|ssn|pin)/.test(name)) return true;
  return false;
}

function fieldValue(el: AnyField): string {
  if (el instanceof HTMLSelectElement) {
    return Array.from(el.selectedOptions, (o) => o.label || o.value).join(", ");
  }
  return el.value ?? "";
}

function fieldSelector(el: AnyField): string {
  if (el.id) return `#${CSS.escape(el.id)}`;
  const name = el.getAttribute("name");
  if (name) return `${el.tagName.toLowerCase()}[name="${name}"]`;
  return el.tagName.toLowerCase();
}

function emitFieldEvent(el: AnyField, reason: string): void {
  if (isSensitive(el)) return;
  const value = fieldValue(el).trim();
  if (value.length < MIN_INPUT_LEN) return;
  send(
    makeEvent("form_input", value, {
      reason,
      selector: fieldSelector(el),
      tag: el.tagName.toLowerCase(),
      input_type: el instanceof HTMLInputElement ? el.type : null,
    }),
  );
}

document.addEventListener(
  "submit",
  (e) => {
    const form = e.target;
    if (!(form instanceof HTMLFormElement)) return;
    const fields = form.querySelectorAll<AnyField>("input, textarea, select");
    fields.forEach((field) => emitFieldEvent(field, "submit"));
  },
  true,
);

const inputTimers = new WeakMap<AnyField, number>();

document.addEventListener(
  "input",
  (e) => {
    const target = e.target;
    if (
      !(
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
      )
    ) {
      return;
    }
    const existing = inputTimers.get(target);
    if (existing !== undefined) clearTimeout(existing);
    const timer = window.setTimeout(() => {
      inputTimers.delete(target);
      emitFieldEvent(target, "input_settle");
    }, INPUT_DEBOUNCE_MS);
    inputTimers.set(target, timer);
  },
  true,
);

document.addEventListener(
  "blur",
  (e) => {
    const target = e.target;
    if (
      !(
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
      )
    ) {
      return;
    }
    const existing = inputTimers.get(target);
    if (existing !== undefined) {
      clearTimeout(existing);
      inputTimers.delete(target);
      emitFieldEvent(target, "blur");
    }
  },
  true,
);
