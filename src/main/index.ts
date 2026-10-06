import { join, resolve } from "node:path";
import { app, BrowserWindow, dialog, type IpcMainInvokeEvent, ipcMain } from "electron";
import { type Core, coreApiMethods, createCore, resolveLanguage } from "../core";
import { channelFor, EVENT_CHANNEL } from "../shared/bridge";
import { translate } from "../shared/i18n";
import { createElectronAdapters, systemBrowser } from "./platform";

// Points the app at another data folder: the smoke test uses a temporary one.
// Set before anything reads `userData`, so Chromium's own data moves there too.
const dataDirOverride = process.env.INCARNAMIND_DATA_DIR;
if (dataDirOverride) app.setPath("userData", resolve(dataDirOverride));

/** Set by electron-vite in development; absent in a built app. */
const rendererUrl = process.env.ELECTRON_RENDERER_URL;

/** Test-only launch flag: lets the smoke tests open UI that nothing else opens yet (the Document viewer). */
const testHooks = process.env.INCARNAMIND_TEST_HOOKS === "1";

let core: Core | undefined;

function isFromOurRenderer(event: IpcMainInvokeEvent): boolean {
  const url = event.senderFrame?.url ?? "";
  return rendererUrl ? url.startsWith(rendererUrl) : url.startsWith("file://");
}

/**
 * Serves the core's public interface to the renderer: one IPC channel per method,
 * and every core event pushed to every window on one event channel.
 */
function exposeCore(core: Core): void {
  for (const method of coreApiMethods) {
    ipcMain.handle(channelFor(method), (event, ...args: unknown[]) => {
      if (!isFromOurRenderer(event)) throw new Error("Refused a call from an unknown page.");
      return Reflect.apply(core[method], core, args) as Promise<unknown>;
    });
  }
  core.onAnyEvent((name, payload) => {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.send(EVENT_CHANNEL, name, payload);
    }
  });
}

function createWindow(): BrowserWindow {
  const window = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 560,
    show: false,
    title: "IncarnaMind",
    backgroundColor: "#e5e7eb",
    webPreferences: {
      preload: join(__dirname, "../preload/index.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  window.once("ready-to-show", () => window.show());

  // Links open in the User's browser, never inside the app.
  window.webContents.setWindowOpenHandler(({ url }) => {
    systemBrowser.open(url).catch(() => undefined);
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event) => {
    if (event.url !== window.webContents.getURL()) event.preventDefault();
  });

  const query: Record<string, string> = testHooks ? { testHooks: "1" } : {};
  if (rendererUrl) {
    const url = new URL(rendererUrl);
    for (const [name, value] of Object.entries(query)) url.searchParams.set(name, value);
    void window.loadURL(url.href);
  } else {
    void window.loadFile(join(__dirname, "../renderer/index.html"), { query });
  }
  return window;
}

function showStartupError(error: unknown): void {
  const language = resolveLanguage("system", app.getPreferredSystemLanguages());
  const message = error instanceof Error ? error.message : String(error);
  dialog.showErrorBox(
    translate(language, "error.startup.title"),
    translate(language, "error.startup.body", { message }),
  );
}

app.whenReady().then(() => {
  try {
    core = createCore(createElectronAdapters());
  } catch (error) {
    showStartupError(error);
    app.quit();
    return;
  }
  exposeCore(core);
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("will-quit", () => {
  core?.close();
});
