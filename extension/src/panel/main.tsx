import React from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";

const container = document.getElementById("root");
if (!container) {
  throw new Error("Missing #root element in side panel");
}

createRoot(container).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
