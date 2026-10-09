import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { files } from "./core";
import "./fonts.css";
import "./styles.css";
import "./viewer/viewer.css";
import { installErrorLog } from "./errorLog";
import { installTestHooks } from "./viewerControls";

installErrorLog();
installTestHooks();

// Which title bar the window has, and whether it's full screen, for styles.css ("The title bar").
document.documentElement.dataset.titleBar = files.titleBar;
files.onFullScreenChange((fullScreen) =>
  document.documentElement.toggleAttribute("data-full-screen", fullScreen),
);

const container = document.getElementById("root");
if (!container) throw new Error("The page has no #root element.");

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
