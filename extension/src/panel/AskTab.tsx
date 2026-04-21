import { useCallback, useEffect, useRef, useState } from "react";

import { BackendError } from "../lib/api";
import { runQueryWithTools } from "../lib/queryLoop";
import type { Citation } from "../lib/types";

interface UserMessage {
  role: "user";
  text: string;
  ts: number;
}

interface AssistantMessage {
  role: "assistant";
  text: string;
  citations: Citation[];
  error?: boolean;
  ts: number;
}

type Message = UserMessage | AssistantMessage;

function formatTs(ts: number): string {
  return new Date(ts).toLocaleString();
}

function CitationList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;
  return (
    <ul className="citations">
      {citations.map((c, i) => (
        <li key={`${c.url}-${i}`} className="citation">
          <a href={c.url} target="_blank" rel="noreferrer">
            {c.url}
          </a>
          <span className="citation-ts">{formatTs(c.ts)}</span>
          {c.snippet && <p className="citation-snippet">{c.snippet}</p>}
        </li>
      ))}
    </ul>
  );
}

export function AskTab() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = scrollerRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages, loading, status]);

  const submit = useCallback(async () => {
    const question = draft.trim();
    if (!question || loading) return;
    setDraft("");
    setLoading(true);
    setStatus(null);
    setMessages((m) => [...m, { role: "user", text: question, ts: Date.now() }]);
    try {
      const resp = await runQueryWithTools(question, {
        onToolStart: (_tool, url) => {
          setStatus(url ? `Looking at ${url}...` : "Looking...");
        },
        onToolEnd: () => setStatus(null),
      });
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: resp.answer ?? "(no answer)",
          citations: resp.citations,
          ts: Date.now(),
        },
      ]);
    } catch (err) {
      const msg =
        err instanceof BackendError
          ? `Backend error (${err.status}): ${err.message}`
          : err instanceof Error
            ? err.message
            : "Unknown error";
      setMessages((m) => [
        ...m,
        { role: "assistant", text: msg, citations: [], error: true, ts: Date.now() },
      ]);
    } finally {
      setStatus(null);
      setLoading(false);
    }
  }, [draft, loading]);

  return (
    <section className="ask">
      <div ref={scrollerRef} className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <p>
              Ask me about anything you've read or written in the browser. The backend must
              be running on <code>localhost:8765</code>.
            </p>
            <p className="muted">
              Examples:
              <br />
              &middot; What was that pricing page Sam sent me?
              <br />
              &middot; Find the GitHub issue I read about websocket reconnects.
              <br />
              &middot; What did I draft to Ana yesterday?
            </p>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="msg user">
              <div className="msg-body">{m.text}</div>
            </div>
          ) : (
            <div key={i} className={`msg assistant${m.error ? " error" : ""}`}>
              <div className="msg-body">{m.text}</div>
              <CitationList citations={m.citations} />
            </div>
          ),
        )}
        {loading && (
          <div className="msg assistant loading">
            <div className="msg-body">{status ?? "Thinking..."}</div>
          </div>
        )}
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <textarea
          rows={2}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask a question..."
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void submit();
            }
          }}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !draft.trim()}>
          Ask
        </button>
      </form>
    </section>
  );
}
