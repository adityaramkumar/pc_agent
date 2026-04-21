/**
 * Service worker entrypoint.
 *
 * Real ingestion batching and tool-executor logic land in subsequent commits.
 * For now this just ensures the extension installs cleanly and the side panel
 * opens when the toolbar action is clicked.
 */

chrome.runtime.onInstalled.addListener(() => {
  console.log("[pc_agent] installed");
});

chrome.action.onClicked.addListener(async (tab) => {
  if (tab.windowId !== undefined) {
    await chrome.sidePanel.open({ windowId: tab.windowId });
  }
});
