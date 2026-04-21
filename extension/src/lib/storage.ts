/** Typed wrapper around chrome.storage.local for user settings. */

import { DEFAULT_SETTINGS, type UserSettings } from "./types";

const SETTINGS_KEY = "pc_agent.settings";

/**
 * Suffix-match: an entry "example.com" blocks "example.com" and any
 * subdomain like "www.example.com", but not "notexample.com".
 */
export function isHostBlocked(host: string, blocklist: string[]): boolean {
  const h = host.toLowerCase();
  return blocklist.some((entry) => {
    const e = entry.trim().toLowerCase();
    if (!e) return false;
    return h === e || h.endsWith("." + e);
  });
}

export function isUrlBlocked(url: string, blocklist: string[]): boolean {
  try {
    return isHostBlocked(new URL(url).host, blocklist);
  } catch {
    return false;
  }
}

export async function loadSettings(): Promise<UserSettings> {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  const raw = stored[SETTINGS_KEY] as Partial<UserSettings> | undefined;
  if (!raw) return { ...DEFAULT_SETTINGS };
  return {
    paused: typeof raw.paused === "boolean" ? raw.paused : DEFAULT_SETTINGS.paused,
    blocklist: Array.isArray(raw.blocklist) ? raw.blocklist : DEFAULT_SETTINGS.blocklist,
  };
}

export async function saveSettings(settings: UserSettings): Promise<void> {
  await chrome.storage.local.set({ [SETTINGS_KEY]: settings });
}

export function watchSettings(cb: (settings: UserSettings) => void): () => void {
  const listener = (
    changes: { [key: string]: chrome.storage.StorageChange },
    area: chrome.storage.AreaName,
  ): void => {
    if (area !== "local") return;
    if (!(SETTINGS_KEY in changes)) return;
    void loadSettings().then(cb);
  };
  chrome.storage.onChanged.addListener(listener);
  return () => chrome.storage.onChanged.removeListener(listener);
}
