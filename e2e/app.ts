import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  type ElectronApplication,
  _electron as electron,
  expect,
  type Locator,
  type Page,
} from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import type { DocumentLocation } from "../src/shared/documentViewer";
import type { TestHooks } from "../src/shared/testHooks";

const appDir = resolve(__dirname, "..");

export interface RunningApp {
  app: ElectronApplication;
  window: Page;
}

export interface LaunchOptions {
  /**
   * Every chat model is the scripted one in `src/main/fakeChatModel.ts`, so
   * Questions are answered with no network or keys. Needs a test build
   * (`npm run test:smoke` makes one).
   */
  fakeChat?: boolean;
  /**
   * Where crash reports go: a test build offers them only with this, and
   * ignores the DSN a real build is made with. Point it at a local server.
   */
  sentryDsn?: string;
}

/**
 * Launches the built app (`out/`) on the given data folder, with test hooks on
 * and the fake embedding model, which needs no download.
 */
export async function launchApp(
  dataDir: string,
  { fakeChat = false, sentryDsn }: LaunchOptions = {},
): Promise<RunningApp> {
  const env: Record<string, string> = {};
  for (const [name, value] of Object.entries(process.env)) {
    if (value !== undefined) env[name] = value;
  }
  delete env.ELECTRON_RUN_AS_NODE;
  delete env.INCARNAMIND_TEST_SENTRY_DSN;
  if (sentryDsn) env.INCARNAMIND_TEST_SENTRY_DSN = sentryDsn;
  env.INCARNAMIND_DATA_DIR = dataDir;
  env.INCARNAMIND_TEST_HOOKS = "1";
  if (fakeChat) env.INCARNAMIND_FAKE_CHAT = "1";
  env.INCARNAMIND_TEST_EMBEDDER = "fake";

  const app = await electron.launch({ args: [appDir], env });
  const window = await app.firstWindow();
  await window.getByTestId("new-mind").waitFor();
  return { app, window };
}

/**
 * On a fresh data folder the first-run chat setup covers the window: choose
 * "set up later". The choice is remembered, so later launches don't show it.
 */
export async function dismissChatSetup(window: Page): Promise<void> {
  const setup = window.getByTestId("chat-setup");
  await setup.getByTestId("chat-setup-later").click();
  await setup.waitFor({ state: "hidden" });
}

/**
 * Sets up a local chat model (Ollama's, so nothing needs consent), through the
 * core's bridge as the setup screen would. With `fakeChat`, the scripted model answers.
 */
export async function useLocalChatModel(window: Page, modelId = "fake-model"): Promise<void> {
  await window.evaluate(async (model) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.saveChatProvider({ kind: "ollama", modelId: model });
  }, modelId);
}

/** Opens Settings from the sidebar, on its Privacy page. */
export async function openPrivacySettings(window: Page): Promise<Locator> {
  await window.getByRole("button", { name: "Settings" }).click();
  await window.getByTestId("settings-tab-privacy").click();
  const privacy = window.getByTestId("privacy-settings");
  await privacy.waitFor();
  return privacy;
}

/**
 * Test hook for the system save dialog: from now on the main process answers
 * it with `filePath` instead of showing it, and records what each call asked
 * (see `saveDialogsAsked`).
 */
export async function interceptSaveDialog(app: ElectronApplication, filePath: string) {
  await app.evaluate(({ dialog }, path) => {
    const asked: Electron.SaveDialogOptions[] = [];
    (globalThis as { saveDialogsAsked?: unknown }).saveDialogsAsked = asked;
    dialog.showSaveDialog = (async (...args: unknown[]) => {
      asked.push(args.at(-1) as Electron.SaveDialogOptions);
      return { canceled: false, filePath: path };
    }) as typeof dialog.showSaveDialog;
  }, filePath);
}

/** What the intercepted save dialog was asked, call by call. */
export function saveDialogsAsked(app: ElectronApplication) {
  return app.evaluate(
    () =>
      (globalThis as { saveDialogsAsked?: Electron.SaveDialogOptions[] }).saveDialogsAsked ?? [],
  );
}

/**
 * Test hook for opening a folder in the system's file manager: from now on
 * the main process records the path instead (see `pathsOpened`).
 */
export async function interceptOpenPath(app: ElectronApplication) {
  await app.evaluate(({ shell }) => {
    const opened: string[] = [];
    (globalThis as { pathsOpened?: unknown }).pathsOpened = opened;
    shell.openPath = async (path: string) => {
      opened.push(path);
      return "";
    };
  });
}

/** The paths the intercepted file manager was asked to open. */
export function pathsOpened(app: ElectronApplication) {
  return app.evaluate(() => (globalThis as { pathsOpened?: string[] }).pathsOpened ?? []);
}

/**
 * Test hook for showing a file in the system's file manager: from now on the
 * main process records the path instead (see `pathsShown`).
 */
export async function interceptShowItemInFolder(app: ElectronApplication) {
  await app.evaluate(({ shell }) => {
    const shown: string[] = [];
    (globalThis as { pathsShown?: unknown }).pathsShown = shown;
    shell.showItemInFolder = (path: string) => {
      shown.push(path);
    };
  });
}

/** The paths the intercepted file manager was asked to show. */
export function pathsShown(app: ElectronApplication) {
  return app.evaluate(() => (globalThis as { pathsShown?: string[] }).pathsShown ?? []);
}

/**
 * Test hook for the system's open dialog: from now on the main process
 * answers it with `path` instead of showing it, e.g. the folder to link.
 */
export async function interceptOpenDialog(app: ElectronApplication, path: string) {
  await app.evaluate(({ dialog }, picked) => {
    dialog.showOpenDialog = (async () => ({
      canceled: false,
      filePaths: [picked],
    })) as unknown as typeof dialog.showOpenDialog;
  }, path);
}

/**
 * Test hook for the system browser: from now on the main process records the
 * URLs it would open (see `urlsOpened`), and opens nothing.
 */
export async function interceptOpenExternal(app: ElectronApplication) {
  await app.evaluate(({ shell }) => {
    const opened: string[] = [];
    (globalThis as { urlsOpened?: unknown }).urlsOpened = opened;
    shell.openExternal = async (url: string) => {
      opened.push(url);
    };
  });
}

/** The URLs the intercepted system browser was asked to open. */
export function urlsOpened(app: ElectronApplication) {
  return app.evaluate(() => (globalThis as { urlsOpened?: string[] }).urlsOpened ?? []);
}

/** Opens the empty Document viewer panel through the test hook. */
export async function openViewer(window: Page): Promise<void> {
  await window.evaluate(() => {
    const hooks = (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks;
    if (!hooks) throw new Error("Test hooks are off: launch with INCARNAMIND_TEST_HOOKS=1.");
    hooks.openViewer();
  });
}

/** Opens a Document in the viewer at a location, as a Citation will, through the test hook. */
export async function openDocumentAt(window: Page, location: DocumentLocation): Promise<void> {
  await window.evaluate((at) => {
    const hooks = (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks;
    if (!hooks) throw new Error("Test hooks are off: launch with INCARNAMIND_TEST_HOOKS=1.");
    hooks.openDocument(at);
  }, location);
}

/**
 * The next Skill import in Settings gets this path instead of the system's
 * open dialog, which a test can't drive, through the test hook.
 */
export async function interceptSkillPicker(window: Page, path: string): Promise<void> {
  await window.evaluate((picked) => {
    const hooks = (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks;
    if (!hooks) throw new Error("Test hooks are off: launch with INCARNAMIND_TEST_HOOKS=1.");
    hooks.interceptSkillPicker(picked);
  }, path);
}

/** Adds files through the sidebar's file picker and waits until each is processed and ready. */
export async function addDocuments(window: Page, paths: string[]): Promise<void> {
  await window.getByTestId("add-documents-input").setInputFiles(paths);
  const items = window.getByTestId("document-list-item");
  await expect(items).toHaveCount(paths.length);
  for (let index = 0; index < paths.length; index++) {
    await expect(items.nth(index)).toHaveAttribute("data-status", "ready");
  }
}

/** The id of the Document listed in the sidebar under `name`. */
export async function documentIdOf(window: Page, name: string): Promise<string> {
  const id = await window
    .getByTestId("document-list-item")
    .filter({ hasText: name })
    .getAttribute("data-document-id");
  if (!id) throw new Error(`No Document named ${name} in the sidebar.`);
  return id;
}

/** How many of a canvas's pixels are dark: more than zero once something is drawn on it. */
export function darkPixels(canvas: Locator): Promise<number> {
  return canvas.evaluate((element) => {
    const drawing = element as unknown as {
      width: number;
      height: number;
      getContext(kind: "2d"): {
        getImageData(x: number, y: number, w: number, h: number): { data: Uint8ClampedArray };
      };
    };
    const { data } = drawing.getContext("2d").getImageData(0, 0, drawing.width, drawing.height);
    let dark = 0;
    for (let index = 0; index < data.length; index += 4) {
      if ((data[index] as number) < 128 && (data[index + 3] as number) > 0) dark++;
    }
    return dark;
  });
}

/** The rendered width of an element, in CSS pixels. */
export async function widthOf(locator: Locator): Promise<number> {
  const box = await locator.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  return box.width;
}

/** Drags an element horizontally by `dx` CSS pixels. */
export async function dragBy(window: Page, handle: Locator, dx: number): Promise<void> {
  const box = await handle.boundingBox();
  if (!box) throw new Error("The drag handle isn't visible.");
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await window.mouse.move(x, y);
  await window.mouse.down();
  await window.mouse.move(x + dx, y, { steps: 5 });
  await window.mouse.up();
}

/** A fresh, empty data folder. Remove it with `removeDataFolder`. */
export const createDataFolder = () => mkdtemp(join(tmpdir(), "incarnamind-smoke-"));

export const removeDataFolder = (dataDir: string) =>
  rm(dataDir, { recursive: true, force: true, maxRetries: 3 });
