import { useState } from "react";

import "./App.css";
import { ActivityTab } from "./ActivityTab";
import { AskTab } from "./AskTab";

type Tab = "ask" | "activity";

export function App() {
  const [tab, setTab] = useState<Tab>("ask");
  return (
    <div className="app">
      <header className="topbar">
        <h1>pc_agent</h1>
        <nav className="tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "ask"}
            className={tab === "ask" ? "active" : ""}
            onClick={() => setTab("ask")}
          >
            Ask
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "activity"}
            className={tab === "activity" ? "active" : ""}
            onClick={() => setTab("activity")}
          >
            Activity
          </button>
        </nav>
      </header>
      <main className="content">{tab === "ask" ? <AskTab /> : <ActivityTab />}</main>
    </div>
  );
}
