/**
 * Orchestrates the agentic query loop from the side panel.
 *
 *   panel                backend               service worker
 *   -----                -------               ---------------
 *   queryStart(question) ->
 *                        returns {session_id, pending_tool, args}
 *   runBrowserTool(tool, args) -> SW opens bg tab and extracts
 *                        <- tool result (text or error)
 *   queryContinue(id, tool_result) ->
 *                        ...loops until {answer, citations}
 */

import { queryContinue, queryStart } from "./api";
import type { RunToolMessage, RunToolResult } from "./messages";
import type { QueryResponse } from "./types";

export const MAX_TOOL_HOPS = 5;

type BrowserTool = "visit_page" | "extract_from_page";

function isBrowserTool(value: unknown): value is BrowserTool {
  return value === "visit_page" || value === "extract_from_page";
}

export async function runBrowserTool(
  tool: BrowserTool,
  args: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const message: RunToolMessage = {
    kind: "run_tool",
    tool,
    args: args as RunToolMessage["args"],
  };
  const result = (await chrome.runtime.sendMessage(message)) as
    | RunToolResult
    | undefined;
  if (!result) return { ok: false, error: "no response from background" };
  return result as unknown as Record<string, unknown>;
}

export interface QueryLoopHooks {
  /** Called each time a browser-side tool is about to run. */
  onToolStart?: (tool: BrowserTool, url: string | null) => void;
  /** Called when the loop returns (success or gives up). */
  onToolEnd?: () => void;
}

/**
 * Run a question to completion, delegating visit_page / extract_from_page
 * round-trips to the service worker and forwarding results back to the
 * backend. Returns the final QueryResponse (may still have `pending_tool`
 * set if we ran out of hops, in which case the caller should show an
 * error and move on).
 */
export async function runQueryWithTools(
  question: string,
  hooks: QueryLoopHooks = {},
): Promise<QueryResponse> {
  let response = await queryStart(question);

  for (let hops = 0; hops < MAX_TOOL_HOPS; hops += 1) {
    if (response.answer != null || !response.session_id || !response.pending_tool) {
      hooks.onToolEnd?.();
      return response;
    }

    const tool = response.pending_tool;
    if (!isBrowserTool(tool)) {
      hooks.onToolEnd?.();
      return {
        ...response,
        answer: `(model requested unknown tool: ${tool})`,
      };
    }

    const args = response.args ?? {};
    const url = typeof args["url"] === "string" ? (args["url"] as string) : null;
    hooks.onToolStart?.(tool, url);
    const result = await runBrowserTool(tool, args);
    response = await queryContinue(response.session_id, result);
  }

  hooks.onToolEnd?.();
  // Exhausted all hops without reaching a final answer.
  if (response.pending_tool) {
    return {
      answer:
        "(the assistant needed more steps than allowed; please try a more specific question)",
      citations: [],
      session_id: null,
      pending_tool: null,
      args: null,
    };
  }
  return response;
}
