import { useCallback, useEffect, useState } from "react";

import { BackendError, deleteMemory, listMemories } from "../lib/api";
import { loadSettings, saveSettings, watchSettings } from "../lib/storage";
import type { MemoryEvent, UserSettings } from "../lib/types";

function formatTs(ts: number): string {
  return new Date(ts).toLocaleString();
}

function hostFromUrl(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

function MemoryRow({ event, onDelete }: { event: MemoryEvent; onDelete: () => void }) {
  return (
    <li className="memory">
      <div className="memory-head">
        <span className={`badge badge-${event.type}`}>{event.type.replace("_", " ")}</span>
        <a href={event.url} target="_blank" rel="noreferrer" className="memory-title">
          {event.title || hostFromUrl(event.url)}
        </a>
        <button
          className="memory-forget"
          title="Forget this memory"
          onClick={onDelete}
          aria-label="Forget this memory"
        >
          forget
        </button>
      </div>
      <div className="memory-meta">
        <span>{hostFromUrl(event.url)}</span>
        <span>·</span>
        <span>{formatTs(event.ts)}</span>
      </div>
      {event.text && <p className="memory-snippet">{event.text.slice(0, 240)}...</p>}
    </li>
  );
}

export function ActivityTab() {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [events, setEvents] = useState<MemoryEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [newDomain, setNewDomain] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await listMemories();
      setEvents(resp.events);
      setTotal(resp.total);
    } catch (err) {
      const msg =
        err instanceof BackendError
          ? `Backend error (${err.status})`
          : err instanceof Error
            ? err.message
            : "Unknown error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSettings().then(setSettings);
    const unsub = watchSettings(setSettings);
    // Initial load on mount. The plugin warns about calling setState in
    // effects, but the canonical "fetch on mount" pattern is exactly this.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
    return unsub;
  }, [refresh]);

  const togglePause = useCallback(async () => {
    if (!settings) return;
    const next = { ...settings, paused: !settings.paused };
    setSettings(next);
    await saveSettings(next);
  }, [settings]);

  const addBlock = useCallback(async () => {
    if (!settings) return;
    const value = newDomain.trim().toLowerCase();
    if (!value || settings.blocklist.includes(value)) return;
    const next = { ...settings, blocklist: [...settings.blocklist, value] };
    setSettings(next);
    setNewDomain("");
    await saveSettings(next);
  }, [newDomain, settings]);

  const removeBlock = useCallback(
    async (domain: string) => {
      if (!settings) return;
      const next = {
        ...settings,
        blocklist: settings.blocklist.filter((d) => d !== domain),
      };
      setSettings(next);
      await saveSettings(next);
    },
    [settings],
  );

  const onDelete = useCallback(
    async (id: number) => {
      try {
        await deleteMemory(id);
        setEvents((prev) => prev.filter((e) => e.id !== id));
        setTotal((t) => Math.max(0, t - 1));
      } catch (err) {
        console.warn("delete failed", err);
      }
    },
    [],
  );

  return (
    <section className="activity">
      <div className="settings-card">
        <label className="row">
          <input
            type="checkbox"
            checked={settings?.paused ?? false}
            onChange={togglePause}
            disabled={!settings}
          />
          <span>{settings?.paused ? "Capture paused" : "Capturing"}</span>
        </label>

        <div className="blocklist">
          <div className="blocklist-head">Domain blocklist</div>
          <ul>
            {(settings?.blocklist ?? []).map((d) => (
              <li key={d}>
                <span>{d}</span>
                <button onClick={() => removeBlock(d)} aria-label={`Remove ${d}`}>
                  remove
                </button>
              </li>
            ))}
          </ul>
          <div className="blocklist-add">
            <input
              type="text"
              placeholder="example.com"
              value={newDomain}
              onChange={(e) => setNewDomain(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void addBlock();
                }
              }}
            />
            <button onClick={addBlock} disabled={!newDomain.trim()}>
              add
            </button>
          </div>
        </div>
      </div>

      <div className="list-head">
        <h2>Recent captures ({total})</h2>
        <button onClick={() => void refresh()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {events.length === 0 && !loading && !error && (
        <p className="muted">
          No memories yet. Start browsing some pages — captures will appear here.
        </p>
      )}

      <ul className="memories">
        {events.map((e) => (
          <MemoryRow key={e.id} event={e} onDelete={() => void onDelete(e.id)} />
        ))}
      </ul>
    </section>
  );
}
