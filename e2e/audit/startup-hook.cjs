/**
 * UI audit: where the main process spends its startup. Loaded before the
 * app's main script with NODE_OPTIONS="--require <this file>" (a dev Electron
 * honours it); writes AUDIT_STARTUP_OUT as JSON once the window has shown.
 * Times are ms since the process started.
 */
const Module = require("node:module");
const { writeFileSync } = require("node:fs");
const { performance } = require("node:perf_hooks");

const out = process.env.AUDIT_STARTUP_OUT;
if (out && process.type === "browser") {
  const now = () => Math.round(performance.now());
  const marks = { hookLoaded: now() };
  const requires = [];
  const originalLoad = Module._load;
  let depth = 0;
  Module._load = function load(...args) {
    const [request, , isMain] = args;
    const started = performance.now();
    depth++;
    try {
      return originalLoad.apply(this, args);
    } finally {
      depth--;
      const ms = performance.now() - started;
      if (isMain) marks.mainScriptDone = now();
      // Requires made by the app's own bundle (depth 1), with everything they pull in.
      else if (depth === 1 && ms >= 2) requires.push({ request, ms: Math.round(ms) });
    }
  };
  const { app, BrowserWindow } = require("electron");
  app.once("will-finish-launching", () => {
    marks.willFinishLaunching = now();
  });
  app.once("ready", () => {
    marks.ready = now();
  });
  app.once("browser-window-created", (_event, window) => {
    marks.windowCreated = now();
    window.webContents.once("did-start-loading", () => {
      marks.rendererStartLoading = now();
    });
    window.webContents.once("dom-ready", () => {
      marks.rendererDomReady = now();
    });
    window.once("ready-to-show", () => {
      marks.readyToShow = now();
      setTimeout(() => {
        marks.written = now();
        writeFileSync(
          out,
          JSON.stringify(
            { marks, requires: requires.sort((a, b) => b.ms - a.ms).slice(0, 25) },
            null,
            2,
          ),
        );
      }, 1500);
    });
  });
  void BrowserWindow;
}
