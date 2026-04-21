import { defineManifest } from "@crxjs/vite-plugin";
import pkg from "./package.json" with { type: "json" };

export default defineManifest({
  manifest_version: 3,
  name: "pc_agent",
  version: pkg.version,
  description: pkg.description,
  permissions: ["tabs", "scripting", "storage", "sidePanel", "activeTab"],
  host_permissions: ["<all_urls>", "http://127.0.0.1:8765/*"],
  background: {
    service_worker: "src/background/sw.ts",
    type: "module",
  },
  content_scripts: [
    {
      matches: ["<all_urls>"],
      js: ["src/content/capture.ts"],
      run_at: "document_idle",
      all_frames: false,
    },
  ],
  side_panel: {
    default_path: "src/panel/index.html",
  },
  action: {
    default_title: "pc_agent",
  },
});
