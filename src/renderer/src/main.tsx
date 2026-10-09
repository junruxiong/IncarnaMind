import { StrictMode } from "react";
import { flushSync } from "react-dom";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./fonts.css";
import "./styles.css";
import "./viewer/viewer.css";
import { installErrorLog } from "./errorLog";
import { installTestHooks } from "./viewerControls";

installErrorLog();
installTestHooks();

const container = document.getElementById("root");
if (!container) throw new Error("The page has no #root element.");

const root = createRoot(container);
// At once, not in a later task: the window shows with its first paint, which then has the
// app's frame (`AppShell`) rather than an empty page.
flushSync(() =>
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  ),
);
