/**
 * Content script entrypoint.
 *
 * Real capture (Readability + selection + non-password input) lands in the
 * `capture_layer` step. For now we just log a heartbeat so it's obvious the
 * script injected.
 */

console.log("[pc_agent] content script loaded for", location.href);

export {};
