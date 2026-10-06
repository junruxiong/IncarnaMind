import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";
import "./viewer/viewer.css";
import { installTestHooks } from "./viewerControls";

installTestHooks();

const container = document.getElementById("root");
if (!container) throw new Error("The page has no #root element.");

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
