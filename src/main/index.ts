import { join, resolve } from "node:path";
import {
  app,
  BrowserWindow,
  dialog,
  type IpcMainEvent,
  type IpcMainInvokeEvent,
  ipcMain,
} from "electron";
import {
  type ChatModelFactory,
  type Core,
  coreApiMethods,
  createCore,
  resolveLanguage,
  type Settings,
} from "../core";
import type { Language } from "../core/language";
import { channelFor, EVENT_CHANNEL } from "../shared/bridge";
import { translate } from "../shared/i18n";
import { registerDocumentScheme, serveDocumentFiles } from "./documentProtocol";
import { serveFileActions } from "./files";
import { startLogging } from "./logging";
import { installAppMenu } from "./menu";
import { chooseExecutor, createElectronAdapters, systemBrowser } from "./platform";
import { reportFullScreen, titleBarOptions } from "./titleBar";
import { registerUpdateCheck, startAutoUpdates } from "./updater";
import { keepWindowPlace, windowPlace } from "./windowState";

// Points the app at another data folder: the smoke test uses a temporary one.
// Set before anything reads `userData`, so Chromium's own data moves there too.
const dataDirOverride = process.env.INCARNAMIND_DATA_DIR;
if (dataDirOverride) app.setPath("userData", resolve(dataDirOverride));

// The log in the data folder's `logs/`, from the start, so it has what goes wrong at startup too.
const logger = startLogging(app.getPath("userData"));

// Whether Skill scripts run in the OS sandbox, found out while Electron starts.
const scriptExecutor = chooseExecutor(logger);

/** Set by electron-vite in development; absent in a built app. */
const rendererUrl = process.env.ELECTRON_RENDERER_URL;

/** Test-only launch flag: lets the smoke tests drive UI directly, e.g. open the Document viewer at a page. */
const testHooks = process.env.INCARNAMIND_TEST_HOOKS === "1";

registerDocumentScheme();

/**
 * The scripted chat model the smoke tests answer Questions with. Only a test
 * build (`electron-vite build --mode test`) has it: in any other build this
 * condition is false at build time, so the model isn't bundled at all.
 */
async function testChatModel(): Promise<ChatModelFactory | undefined> {
  if (import.meta.env.MODE === "test" && process.env.INCARNAMIND_FAKE_CHAT === "1") {
    const { createFakeChatModel } = await import("./fakeChatModel");
    return createFakeChatModel;
  }
  return undefined;
}

let core: Core | undefined;

function isFromOurRenderer(event: IpcMainInvokeEvent | IpcMainEvent): boolean {
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
  const dataDir = app.getPath("userData");
  const place = windowPlace(dataDir, { width: 1280, height: 800 }, { width: 900, height: 560 });
  const window = new BrowserWindow({
    ...(place.x !== undefined && place.y !== undefined ? { x: place.x, y: place.y } : {}),
    width: place.width,
    height: place.height,
    minWidth: 900,
    minHeight: 560,
    show: false,
    title: "IncarnaMind",
    // The frame (DESIGN.md), as the approved canvas has it: the sidebar's colour, which the
    // first paint shows at the top left, under macOS's traffic lights.
    backgroundColor: "#F4F5F7",
    // No title row of the system's: its window buttons sit in the app's top band.
    ...titleBarOptions(process.platform),
    webPreferences: {
      preload: join(__dirname, "../preload/index.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  window.once("ready-to-show", () => {
    if (place.maximized) window.maximize();
    window.show();
  });
  keepWindowPlace(window, dataDir);
  reportFullScreen(window);

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

/**
 * The application menu, in the interface language, and again whenever it
 * changes. Reload and the developer tools only outside a packaged app.
 */
function installMenuInLanguage(core: Core): void {
  const developer = !app.isPackaged;
  let shown: Language | null = null;
  const install = (language: Language) => {
    if (language === shown) return;
    shown = language;
    installAppMenu(language, { developer });
  };
  install(resolveLanguage("system", app.getPreferredSystemLanguages()));
  core.getSettings().then(
    (settings) => install(settings.language),
    () => undefined,
  );
  core.onAnyEvent((name, payload) => {
    if (name === "settings.changed") install((payload as Settings).language);
  });
}

function showStartupError(error: unknown): void {
  logger.exception("app.startFailed", error);
  const language = resolveLanguage("system", app.getPreferredSystemLanguages());
  const message = error instanceof Error ? error.message : String(error);
  dialog.showErrorBox(
    translate(language, "error.startup.title"),
    translate(language, "error.startup.body", { message }),
  );
}

app.whenReady().then(async () => {
  const createChatModel = await testChatModel();
  const executor = await scriptExecutor;
  // The window's page starts loading while the core starts. Its calls reach the core once
  // `exposeCore` has run, below: they wait until this synchronous run is done.
  createWindow();
  try {
    core = createCore({
      ...createElectronAdapters(logger),
      ...(executor && { executor }),
      ...(createChatModel && { createChatModel }),
    });
  } catch (error) {
    showStartupError(error);
    app.quit();
    return;
  }
  exposeCore(core);
  registerUpdateCheck(core);
  serveFileActions(core, {
    dataDir: app.getPath("userData"),
    trusted: isFromOurRenderer,
    logger,
  });
  serveDocumentFiles(core, rendererUrl ? new URL(rendererUrl).origin : null);
  installMenuInLanguage(core);
  // Only a packaged app checks for updates; the smoke tests must never reach GitHub.
  if (!testHooks) startAutoUpdates(core);

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
