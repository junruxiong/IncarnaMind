import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  type ElectronApplication,
  _electron as electron,
  expect,
  type Locator,
  type Page,
  test,
} from "@playwright/test";
import type { Editor } from "@tiptap/core";
import * as Y from "yjs";
import type { AssistantPlace, CoreBridge } from "../src/core/api";
import type { FakeTranscriberOptions } from "../src/core/transcription/fake";
import type { DocumentLocation } from "../src/shared/documentViewer";
import type { TestHooks } from "../src/shared/testHooks";

const appDir = resolve(__dirname, "..");

/**
 * The apps this test launched that are still open, by the data folder each
 * runs on: `removeDataFolder` closes them, keeping a picture of the window if
 * the test failed.
 */
const openApps = new Map<ElectronApplication, string>();

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
  /**
   * Where usage data goes: a test build sends it only with this, to a local
   * server standing in for PostHog, and ignores the project a real build is
   * made with. `testerBuild`: as a test build (for invited testers) would, on until the
   * User turns it off.
   */
  usageData?: { host: string; testerBuild?: boolean };
  /**
   * The app ships its examples (`resources/examples`), so a first run opens
   * on the example Mind with "Get started". Without, a test build ships none.
   */
  examples?: boolean;
  /**
   * The app's home folder, for a test that looks at what the app writes there. Without one,
   * each launch gets its own temporary home, so nothing the app writes there is the User's.
   */
  home?: string;
  /**
   * Where Look up asks Crossref and arXiv: a local server standing in for
   * both (tests/helpers/referenceServices.ts). A test build honours it.
   */
  referenceLookup?: string;
  /**
   * Where Find sources asks its providers: a local server standing in for
   * all of them (tests/helpers/sourceServices.ts). A test build honours it.
   */
  sources?: string;
  /**
   * Where the ChatGPT plan signs in and asks: the origin of a local server
   * standing in for OpenAI (tests/helpers/fakeOpenAI.ts). A test build honours it.
   */
  chatGpt?: string;
  /**
   * Leaves the window at the size and place it opened at, instead of making
   * it `WINDOW_SIZE`: for a test of how the window opens.
   */
  keepWindow?: boolean;
  /**
   * What a Meeting's fake capture does with the computer's sound: plays a
   * tone (by default), is denied, or gives only silence. The microphone is
   * always a fake that makes a tone: no test opens a device.
   */
  computerSound?: "allowed" | "denied" | "silent";
  /** What a Meeting's fake speech-to-text engine does: slow, failing to read stretches, failing over. */
  transcriber?: FakeTranscriberOptions;
  /**
   * The port of a local server standing in for the web (#128): web search's
   * fetcher connects there for every host, its guard still deciding. A test
   * build honours it.
   */
  web?: number;
  /**
   * What the fake text recognition model reads on each scanned page (#397):
   * a JSON file of a `FakeScanScript`. Without, it reads nothing on them.
   */
  scans?: string;
  /**
   * The scripted Task that asks for approval and writes files (src/main/scriptedTask.ts),
   * started with `startScriptedTask`. Needs a test build.
   */
  scriptedTasks?: boolean;
}

/**
 * The window every spec runs in, the same on every machine, and a height that
 * fits the smallest screen the specs run on. The app opens no bigger than the
 * screen, and GitHub's macOS runners have a 1024 × 768 one (677px high once
 * the menu bar and the Dock are off it), where a window opened 1024 wide folds
 * the Mind's margins away. macOS lets a window be wider than its screen, but
 * not taller, so each launch sets this size and checks the page got it, and
 * says so if a screen is too short for it. It is the page's size, the window's
 * content: on Linux the window also holds the menu bar, above the page.
 *
 * It is wider than the app's default (1280): the assistant docked on the right
 * takes 328px (#253), and this leaves the card about the room a 1280px window
 * gave it before, so groups sit side by side and the Mind keeps its margins.
 */
export const WINDOW_SIZE = { width: 1600, height: 660 };

/**
 * The key that, held with an arrow, moves the caret a word at a time, and with
 * Shift too selects by word: Option on macOS, Ctrl on Windows and Linux.
 */
export const WORD_KEY = process.platform === "darwin" ? "Alt" : "Control";

/**
 * Launches the built app (`out/`) on the given data folder, with test hooks on
 * and the fake embedding model, which needs no download, in a window of
 * `WINDOW_SIZE` whatever the screen's.
 */
export async function launchApp(
  dataDir: string,
  {
    fakeChat = false,
    sentryDsn,
    usageData,
    examples = false,
    home,
    referenceLookup,
    sources,
    chatGpt,
    keepWindow = false,
    computerSound = "allowed",
    transcriber,
    scans,
    web,
    scriptedTasks = false,
  }: LaunchOptions = {},
): Promise<RunningApp> {
  const env: Record<string, string> = {};
  for (const [name, value] of Object.entries(process.env)) {
    if (value !== undefined) env[name] = value;
  }
  delete env.ELECTRON_RUN_AS_NODE;
  delete env.INCARNAMIND_TEST_SENTRY_DSN;
  delete env.INCARNAMIND_TEST_EXAMPLES;
  delete env.INCARNAMIND_TEST_POSTHOG_KEY;
  delete env.INCARNAMIND_TEST_POSTHOG_HOST;
  delete env.INCARNAMIND_TEST_TESTER_BUILD;
  delete env.POSTHOG_CAPTURE_MODE;
  delete env.INCARNAMIND_TEST_REFERENCE_LOOKUP;
  delete env.INCARNAMIND_TEST_SOURCES;
  delete env.INCARNAMIND_TEST_CHATGPT;
  delete env.INCARNAMIND_TEST_TRANSCRIBER_OPTIONS;
  delete env.INCARNAMIND_TEST_SCANS;
  delete env.INCARNAMIND_TEST_WEB;
  delete env.INCARNAMIND_TEST_TASKS;
  // A test never uses the ChatGPT sign-in an agent's shell may point at for checks by hand.
  delete env.INCARNAMIND_CHATGPT_PROFILE;
  env.INCARNAMIND_TEST_AUDIO = "fake";
  env.INCARNAMIND_TEST_AUDIO_SYSTEM = computerSound;
  if (examples) env.INCARNAMIND_TEST_EXAMPLES = "1";
  if (sentryDsn) env.INCARNAMIND_TEST_SENTRY_DSN = sentryDsn;
  if (usageData) {
    env.INCARNAMIND_TEST_POSTHOG_KEY = "test-project-key";
    env.INCARNAMIND_TEST_POSTHOG_HOST = usageData.host;
    if (usageData.testerBuild) env.INCARNAMIND_TEST_TESTER_BUILD = "1";
  }
  const appHome = home ?? (await mkdtemp(join(tmpdir(), "incarnamind-home-")));
  env.HOME = appHome;
  if (referenceLookup) env.INCARNAMIND_TEST_REFERENCE_LOOKUP = referenceLookup;
  if (sources) env.INCARNAMIND_TEST_SOURCES = sources;
  if (chatGpt) env.INCARNAMIND_TEST_CHATGPT = chatGpt;
  if (scans) env.INCARNAMIND_TEST_SCANS = scans;
  if (web) env.INCARNAMIND_TEST_WEB = String(web);
  if (scriptedTasks) env.INCARNAMIND_TEST_TASKS = "1";
  env.INCARNAMIND_DATA_DIR = dataDir;
  env.INCARNAMIND_TEST_HOOKS = "1";
  if (fakeChat) env.INCARNAMIND_FAKE_CHAT = "1";
  env.INCARNAMIND_TEST_EMBEDDER = "fake";
  env.INCARNAMIND_TEST_TRANSCRIBER = "fake";
  if (transcriber) env.INCARNAMIND_TEST_TRANSCRIBER_OPTIONS = JSON.stringify(transcriber);

  const app = await electron.launch({ args: [appDir], env });
  openApps.set(app, dataDir);
  app.on("close", () => {
    openApps.delete(app);
    if (!home) void rm(appHome, { recursive: true, force: true, maxRetries: 3 });
  });
  const window = await app.firstWindow();
  await window.getByTestId("plus-menu").waitFor();
  if (!keepWindow) {
    await resizeContent(app, window, WINDOW_SIZE.width, WINDOW_SIZE.height, {
      waitForHeight: true,
      message: `the window is ${WINDOW_SIZE.width} × ${WINDOW_SIZE.height}: is the screen too short for it?`,
    });
  }
  return { app, window };
}

/**
 * Starts the scripted Task (`LaunchOptions.scriptedTasks`) in a Mind: for each
 * file it asks for approval, then writes the file into `folder`. Returns its id.
 */
export const startScriptedTask = (
  app: ElectronApplication,
  mindId: string,
  input: { folder: string; files: string[] },
) =>
  app.evaluate(
    (_electron, [mind, given]) =>
      (
        globalThis as unknown as {
          incarnamindScriptedTask: { start(mindId: string, input: unknown): string };
        }
      ).incarnamindScriptedTask.start(mind, given),
    [mindId, input] as const,
  );

/**
 * Stands in for the system's notifications: each one shown is recorded
 * instead (`notificationsShown`). Notifications made before this aren't.
 */
export const recordNotifications = (app: ElectronApplication) =>
  app.evaluate(({ Notification }) => {
    const shown: { title: string; body: string }[] = [];
    (globalThis as { notificationsShown?: typeof shown }).notificationsShown = shown;
    Notification.isSupported = () => true;
    Notification.prototype.show = function show(this: Electron.Notification) {
      shown.push({ title: this.title, body: this.body });
    };
  });

export const notificationsShown = (app: ElectronApplication) =>
  app.evaluate(
    () =>
      (globalThis as { notificationsShown?: { title: string; body: string }[] })
        .notificationsShown ?? [],
  );

/** Chooses the Edit menu's Undo or Redo, as the application menu has it now. */
export const clickEditItem = (app: ElectronApplication, id: "edit.undo" | "edit.redo") =>
  app.evaluate(({ Menu }, itemId) => {
    Menu.getApplicationMenu()?.getMenuItemById(itemId)?.click();
  }, id);

/**
 * "New Mind" from the sidebar header's "+" menu, the one place to add: a Mind
 * in no Folder. Returns once the menu has gone: keys a test presses a
 * millisecond apart can outrun the editor while frames are busy.
 */
export async function newMind(window: Page): Promise<void> {
  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-new-mind").click();
  await motionSettled(window);
}

/** A Folder's row in the sidebar's Folders view, with what is listed in it, by its name. */
export const sidebarFolder = (window: Page, name: string) =>
  window
    .getByTestId("library-folders")
    .getByTestId("library-folder")
    .filter({
      has: window.locator(':scope > div [data-testid="row-text"]', { hasText: name }),
    });

/** Opens a Folder's page in the Library's tab, from its menu in the sidebar: "Open in the Library". */
export async function openFolderInLibrary(window: Page, name: string): Promise<void> {
  const folder = sidebarFolder(window, name);
  await folder.getByTestId("folder-row").hover();
  await folder.getByTestId("folder-menu").click();
  await folder.getByTestId("folder.open").click();
  await expect(window.getByTestId("library")).toBeVisible();
}

/**
 * Opens a Folder's form (its name, description and evidence choice) under its
 * section in the Library, from its menu in the sidebar: "Edit Description…".
 */
export async function editFolderInLibrary(window: Page, name: string): Promise<Locator> {
  const folder = sidebarFolder(window, name);
  await folder.getByTestId("folder-row").hover();
  await folder.getByTestId("folder-menu").click();
  await folder.getByTestId("folder.editDescription").click();
  const form = window.getByTestId("library").getByTestId("folder-form");
  await expect(form).toBeVisible();
  return form;
}

/**
 * Shows the sidebar's Source locations view: each Linked folder with its
 * folders as on disk, and "Other Documents".
 */
export async function showSourceLocations(window: Page): Promise<void> {
  const button = window.getByTestId("browse-views").getByRole("button", {
    name: "Source locations",
  });
  await button.click();
  await expect(button).toHaveAttribute("aria-pressed", "true");
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

/**
 * Turns embeddings on with the built-in model (the smoke tests' fake), through
 * the core's bridge as Settings would: they are off by default.
 */
export async function turnOnEmbeddings(window: Page): Promise<void> {
  await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.saveEmbeddingProvider({ kind: "built-in" });
  });
}

/**
 * Keys are kept with the system's keyring. On Linux with none running, as on
 * CI's Linux runner, the User first accepts storing them in plain text, once
 * on this computer. With `form`, accepts it as the User does, in the notice a
 * form that saves a key shows ("No keyring is running"); without, through the
 * core's bridge, as that notice does. With a keyring there's nothing to accept.
 */
export async function acceptPlainTextKeys(window: Page, form?: Locator): Promise<void> {
  const { protection, plainTextAccepted } = await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.getSecretStorage(),
  );
  if (protection !== "plain-text" || plainTextAccepted) return;
  if (!form) {
    await window.evaluate(async () => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await bridge.acceptPlainTextSecretStorage();
    });
    return;
  }
  const notice = form.getByTestId("plain-text-secrets");
  await notice.getByRole("checkbox").click();
  // Accepted, the key can be saved, and the notice goes.
  await expect(notice).toBeHidden();
}

/** The pages in Settings' list. */
export type SettingsPage =
  | "general"
  | "models"
  | "search"
  | "meetings"
  | "connectors"
  | "skills"
  | "agents"
  | "privacy";

/** Opens Settings from the sidebar, then one of the pages in its list. Returns the dialog. */
export async function openSettings(window: Page, page: SettingsPage): Promise<Locator> {
  await window.getByRole("button", { name: "Settings" }).click();
  return showSettingsPage(window, page);
}

/** In open Settings, shows one of the pages in its list. Returns the dialog. */
export async function showSettingsPage(window: Page, page: SettingsPage): Promise<Locator> {
  const settings = window.getByTestId("settings");
  await settings.getByTestId(`settings-nav-${page}`).click();
  await expect(settings).toHaveAttribute("data-page", page);
  return settings;
}

/** Closes Settings with its close button. */
export async function closeSettings(window: Page): Promise<void> {
  const settings = window.getByTestId("settings");
  await settings.getByRole("button", { name: "Close Settings" }).click();
  await expect(settings).toBeHidden();
}

/** Opens Settings from the sidebar, on its Privacy page. */
export async function openPrivacySettings(window: Page): Promise<Locator> {
  await openSettings(window, "privacy");
  const privacy = window.getByTestId("privacy-settings");
  await privacy.waitFor();
  return privacy;
}

/** Opens a Document's "More" menu in the sidebar, for rename, its original file, and delete. */
export async function openDocumentMenu(item: Locator): Promise<void> {
  await item.hover();
  await item.getByTestId("document-file-menu").click();
}

/**
 * Opens a Document's Tags menu in the sidebar and returns the Tags it has
 * (each a checked item, with `data-source` and, if unsure, `data-needs-review`).
 * Esc closes the menu again.
 */
export async function openDocumentTags(item: Locator): Promise<Locator> {
  await item.hover();
  await item.getByTestId("document-tags-menu").click();
  const menu = item.getByTestId("document-tags-popover");
  await expect(menu).toBeVisible();
  // The Tags it has, in the picker's field.
  return menu.getByTestId("tag-token");
}

/**
 * Adds a Tag to the sidebar's filter (Documents with any Tag chosen show), or,
 * choosing a Tag already chosen, takes it out.
 */
export async function filterByTag(window: Page, name: string): Promise<void> {
  await window.getByTestId("tag-filter-menu").click();
  await window
    .getByTestId("tag-filters")
    .getByRole("menuitemcheckbox", { name, exact: true })
    .click();
}

/**
 * Sets where new files go (the Output location) through the core's bridge, as
 * Settings › General would, so the next save goes there without asking.
 */
export async function useOutputLocation(window: Page, folder: string): Promise<void> {
  await window.evaluate(async (path) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.setOutputLocation(path);
  }, folder);
}

/** The result line's saved file, once it says it saved one. */
export async function savedPath(window: Page, name: string): Promise<string> {
  const done = window.getByTestId("export-result").getByTestId("export-done");
  await expect(done).toHaveText(`Saved ${name}`, { timeout: 30_000 });
  return (await done.getAttribute("title")) ?? "";
}

/** The menu of the Mind's tab, opened by a right-click on the tab, as a person opens it. */
export async function tabMenu(window: Page): Promise<Locator> {
  const tab = window.getByRole("tab", { selected: true }).first();
  await pointAndClick(window, tab, { button: "right" });
  const menu = tab.getByTestId("tab-actions");
  await expect(menu).toBeVisible();
  return menu;
}

/** Export options…, from the tab's menu; resolves once the sheet shows `title`. */
export async function openOptions(window: Page, title: string): Promise<void> {
  await pointAndClick(window, (await tabMenu(window)).getByTestId("mind.exportOptions"));
  await expect(window.getByTestId("export-dialog").getByRole("heading")).toHaveText(title);
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

/**
 * "Link a folder…" from the sidebar's "+" menu (or `button`): the system's
 * folder picker (answered by the test with `path`), then the link dialog's
 * preview. Returns the dialog, with the preview counted. Choose a layout in
 * it, then `confirmLink`.
 */
export async function previewLink(
  app: ElectronApplication,
  window: Page,
  path: string,
  button?: Locator,
): Promise<Locator> {
  await interceptOpenDialog(app, path);
  if (button) await button.click();
  else {
    await window.getByTestId("plus-menu").click();
    await window.getByTestId("plus-link-folder").click();
  }
  const dialog = window.getByTestId("link-folder-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.getByTestId("link-folder-files")).toBeVisible();
  return dialog;
}

/** "Link folder" in the link dialog; it closes. */
export async function confirmLink(dialog: Locator): Promise<void> {
  await dialog.getByTestId("link-folder-confirm").click();
  await expect(dialog).toBeHidden();
}

/** Links a folder as the User does: "Link a folder…", the picker, then "Link folder" in the dialog. */
export async function linkFolderFromSidebar(
  app: ElectronApplication,
  window: Page,
  path: string,
): Promise<void> {
  await confirmLink(await previewLink(app, window, path));
}

/** A Linked folder's own row in the sidebar, by its folder's name on disk. */
export const linkedFolderRow = (window: Page, name: string) =>
  window.locator('[data-testid="folder-item"][data-root="true"]').filter({
    has: window.getByTestId("row-text").getByText(name, { exact: true }),
  });

/** Opens a Linked folder's "More" menu from its row. Returns the menu. */
export async function openLinkedFolderMenu(row: Locator): Promise<Locator> {
  await row.hover();
  await row.getByTestId("linked-folder-menu").click();
  const menu = row.getByTestId("linked-folder-actions");
  await expect(menu).toBeVisible();
  return menu;
}

/**
 * Opens the Library in a group beside the Mind, the focus staying in the
 * Mind, or closes every group but the Mind's, through the core's bridge: the
 * Mind then has half the card, as beside a Document, or all of it.
 */
export async function setGroupBeside(window: Page, open: boolean): Promise<void> {
  await window.evaluate(async (show) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const layout = await bridge.getLayout("main");
    if (show && layout.groups.length === 1) {
      await bridge.changeLayout("main", {
        type: "open",
        item: { kind: "library" },
        where: { in: "side" },
        focus: false,
      });
    } else if (!show) {
      for (const group of layout.groups.slice(1)) {
        await bridge.changeLayout("main", { type: "closeGroup", groupId: group.id });
      }
    }
  }, open);
  await expect(window.getByTestId("tab-group")).toHaveCount(open ? 2 : 1);
}

/**
 * Places the window's assistant (#253) through the core's bridge, as its ⋯
 * menu does: for a spec about something else that needs the card's room, such
 * as two groups side by side in a narrow window (docked at the bottom).
 */
export async function placeAssistant(window: Page, place: Partial<AssistantPlace>): Promise<void> {
  await window.evaluate(async (given) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.changeLayout("main", { type: "assistant", place: given });
  }, place);
  const assistant = window.getByTestId("assistant");
  if (place.dock) await expect(assistant).toHaveAttribute("data-chosen-dock", place.dock);
  if (place.folded !== undefined) {
    if (place.folded) await expect(assistant).toHaveAttribute("data-folded", "true");
    else await expect(assistant).not.toHaveAttribute("data-folded");
  }
}

/**
 * Opens a Document at a location as a Citation does, through the test hook: as a
 * preview in the group right of the Mind the User is in (made if needed); with
 * no Mind in front, as from the sidebar.
 */
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

/** Not in a Folder lists its first five Documents, then "Show N more Documents". */
const FIRST_LISTED = 5;

/**
 * Adds files through the sidebar's file picker and waits until each is
 * processed and ready. Each shows Not in a Folder (its first five, if more).
 */
export async function addDocuments(window: Page, paths: string[]): Promise<void> {
  await window.getByTestId("add-documents-input").setInputFiles(paths);
  const items = window.getByTestId("document-list-item");
  const listed = Math.min(paths.length, FIRST_LISTED);
  await expect(items).toHaveCount(listed);
  for (let index = 0; index < listed; index++) {
    await expect(items.nth(index)).toHaveAttribute("data-status", "ready");
  }
  if (paths.length > listed) await expectReadyDocuments(window, paths.length);
}

/**
 * Sizes the app's page, e.g. to its narrowest (900px wide, the minimum) or a
 * wide one, as `WINDOW_SIZE` is: its content, without Linux's menu bar. The
 * screen may not have room for the height (see `WINDOW_SIZE`), so only the
 * width is waited for.
 */
export async function setWindowSize(
  app: ElectronApplication,
  window: Page,
  width: number,
  height: number,
): Promise<void> {
  await resizeContent(app, window, width, height);
}

/**
 * Asks for the content size once, waits until the page has it, then until the
 * page has drawn a frame at it. Chromium sends a page its next size only after
 * the page has drawn at the one before. On a runner's first launch the first
 * frames can take seconds, so a size asked right after the launch's waited
 * that long, while the page still reported the old one (#432). The launch also
 * waits for the height; `setWindowSize` only the width.
 */
async function resizeContent(
  app: ElectronApplication,
  window: Page,
  width: number,
  height: number,
  { message, waitForHeight = false }: { message?: string; waitForHeight?: boolean } = {},
): Promise<void> {
  const browserWindow = await app.browserWindow(window);
  await browserWindow.evaluate((win, asked) => win.setContentSize(asked.width, asked.height), {
    width,
    height,
  });
  const look = () =>
    window.evaluate(() => ({ width: globalThis.innerWidth, height: globalThis.innerHeight }));
  try {
    await expect
      .poll(
        async () => {
          const page = await look();
          return [page.width, waitForHeight ? page.height : height];
        },
        { message },
      )
      .toEqual([width, height]);
  } catch (error) {
    const content = await browserWindow.evaluate((win) => win.getContentSize());
    const page = await look();
    throw new Error(
      `${message ?? "The window did not reach its size"}: asked ${width} × ${height}; the window's content is ${content.join(" × ")}, the page ${page.width} × ${page.height}.`,
      { cause: error },
    );
  }
  await window.evaluate(
    () =>
      new Promise((drawn) => requestAnimationFrame(() => requestAnimationFrame(() => drawn(null)))),
  );
}

/** Switches the interface's language, as Settings does. */
export function setLanguage(window: Page, language: "en" | "zh-CN"): Promise<unknown> {
  return window.evaluate(
    (value) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: value },
      }),
    language,
  );
}

/** How many animations are running (src/renderer/src/motion.ts), leaving copies included. */
export function motionRunning(window: Page): Promise<number> {
  return window.evaluate(
    () =>
      (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks?.motionRunning() ??
      0,
  );
}

/** The copies of closed menus and popovers still fading where they were, which nothing else can see. */
export function leavingCopies(window: Page) {
  return window.evaluate(
    () =>
      (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks?.leavingCopies() ??
      [],
  );
}

/**
 * How an element shows now: its opacity, its transform's scale and vertical
 * shift, its transform origin, and where it is laid out without the transform.
 */
export function motionLook(target: Locator) {
  return target.evaluate((element) => {
    const style = getComputedStyle(element);
    const matrix = new DOMMatrixReadOnly(style.transform === "none" ? undefined : style.transform);
    const [originX = "0", originY = "0"] = style.transformOrigin.split(" ");
    const box = element.getBoundingClientRect();
    return {
      opacity: Number(style.opacity),
      scale: matrix.a,
      y: matrix.f,
      originX: Number.parseFloat(originX),
      originY: Number.parseFloat(originY),
      left: box.left - matrix.e - (1 - matrix.a) * Number.parseFloat(originX),
      top: box.top - matrix.f - (1 - matrix.d) * Number.parseFloat(originY),
    };
  });
}

/**
 * Waits until nothing moves (src/renderer/src/motion.ts): no animation
 * running, and no closed menu still fading where it was. For a test that
 * measures or pictures what just opened, closed or moved.
 */
export async function motionSettled(window: Page): Promise<void> {
  await expect
    .poll(() =>
      window.evaluate(
        () =>
          (
            globalThis as { incarnamindTestHooks?: TestHooks }
          ).incarnamindTestHooks?.motionRunning() ?? 0,
      ),
    )
    .toBe(0);
}

/**
 * Clicks `target` where it is now, as a person does, without waiting for it to
 * stop moving (a menu still growing from its button): the app takes a click
 * during an animation. For a test that races work in the background, which
 * waiting would let finish first.
 */
export async function clickNow(window: Page, target: Locator): Promise<void> {
  await expect(target).toBeVisible();
  const box = await target.boundingBox();
  if (!box) throw new Error("The element isn't on screen.");
  await window.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await window.mouse.down();
  await window.mouse.up();
}

/** Plays every animation at `speed` (0.2: five times slower), so a test can look, or act, during one. */
export async function setMotionSpeed(window: Page, speed: number): Promise<void> {
  await window.evaluate(
    (value) =>
      (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks?.setMotionSpeed(
        value,
      ),
    speed,
  );
}

/**
 * Waits until the core has this many Documents ready, wherever the sidebar
 * lists them (a Folder starts closed, and lists its first few).
 */
export async function expectReadyDocuments(window: Page, count: number): Promise<void> {
  await expect
    .poll(
      () =>
        window.evaluate(
          async () =>
            (
              await (
                globalThis as unknown as { incarnamind: CoreBridge }
              ).incarnamind.listDocuments()
            ).filter((each) => each.status === "ready").length,
        ),
      { timeout: 30_000 },
    )
    .toBe(count);
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

/**
 * Checks that the window's first group's tabs start at the card's left edge:
 * flush with it on macOS, where the sidebar's button sits in the sidebar's
 * header beside the traffic lights; on Windows and Linux just after that
 * button, which starts the strip there (SidebarToggle), 8px in and 6px before
 * the first tab (styles.css).
 */
export async function expectFirstTabAtCardStart(window: Page, firstTab: Locator): Promise<void> {
  const card = await window.getByTestId("card").boundingBox();
  const tab = await firstTab.boundingBox();
  if (!card || !tab) throw new Error("The card or its first tab isn't visible.");
  const toggle = window.getByTestId("tab-strip").first().getByTestId("sidebar-toggle");
  if (process.platform === "darwin") {
    await expect(toggle).toHaveCount(0);
    expect(tab.x).toBeCloseTo(card.x, 0);
    return;
  }
  const button = await toggle.boundingBox();
  if (!button) throw new Error("The sidebar's button isn't at the start of the strip.");
  expect(button.x - card.x).toBeCloseTo(8, 0);
  expect(tab.x - (button.x + button.width)).toBeCloseTo(6, 0);
}

/**
 * Slides the mouse onto an element, in steps, as a person does, and clicks it:
 * with the left button, or the right to open its menu.
 */
export async function pointAndClick(
  window: Page,
  target: Locator,
  {
    button = "left",
    modifiers = [],
  }: { button?: "left" | "right"; modifiers?: readonly ("ControlOrMeta" | "Shift")[] } = {},
): Promise<void> {
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await window.mouse.move(x, y, { steps: 8 });
  for (const key of modifiers) await window.keyboard.down(key);
  try {
    await window.mouse.click(x, y, { button });
  } finally {
    for (const key of modifiers.toReversed()) await window.keyboard.up(key);
  }
}

/**
 * Drags an element horizontally by `dx` CSS pixels, as a person grabs it: a
 * few pixels off its centre (a divider is a 1px rule with a wider strip to
 * grab), after sliding the mouse there.
 */
export async function dragBy(window: Page, handle: Locator, dx: number): Promise<void> {
  const box = await handle.boundingBox();
  if (!box) throw new Error("The drag handle isn't visible.");
  const x = box.x + box.width / 2 - 3;
  const y = box.y + box.height / 2;
  await window.mouse.move(x - 30, y);
  await window.mouse.move(x, y, { steps: 6 });
  await window.mouse.down();
  await window.mouse.move(x + dx, y, { steps: 5 });
  await window.mouse.up();
}

/**
 * Drags a tab as a person does: slides onto it, presses, slides to (x, y) in
 * small steps, then, once `whileHeld` has looked at what the drag shows, lets go.
 */
export async function dragTabTo(
  window: Page,
  tab: Locator,
  to: { x: number; y: number },
  whileHeld?: () => Promise<void>,
): Promise<void> {
  const box = await tab.boundingBox();
  if (!box) throw new Error("The tab isn't visible.");
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await window.mouse.move(x - 20, y + 30);
  await window.mouse.move(x, y, { steps: 5 });
  await window.mouse.down();
  try {
    await window.mouse.move(to.x, to.y, { steps: 16 });
    await whileHeld?.();
  } finally {
    await window.mouse.up();
  }
}

/**
 * Moves the mouse as a person does onto a Block's handle: onto the Block's
 * first line, then left across the margin in small steps (not a jump, which
 * would skip the margin in between), and returns the handle.
 */
export async function slideToHandle(window: Page, block: Locator): Promise<Locator> {
  const text = await block.boundingBox();
  if (!text) throw new Error("The Block isn't visible.");
  const y = text.y + Math.min(text.height, 28) / 2;
  await window.mouse.move(text.x + 8, y, { steps: 4 });
  const handle = window.getByTestId("block-handle");
  await expect(handle).toBeVisible();
  const grip = await handle.boundingBox();
  if (!grip) throw new Error("The block handle isn't visible.");
  await window.mouse.move(grip.x + grip.width / 2, grip.y + grip.height / 2, { steps: 12 });
  await expect(handle).toBeVisible();
  return handle;
}

/**
 * Drags a Block by its handle, as a person does: slide onto the handle, press,
 * move straight up or down the margin to the top of `target`, and let go.
 */
export async function dragBlock(window: Page, block: Locator, target: Locator): Promise<void> {
  const handle = await slideToHandle(window, block);
  const grip = await handle.boundingBox();
  const to = await target.boundingBox();
  if (!grip || !to) throw new Error("The handle or the target isn't visible.");
  const x = grip.x + grip.width / 2;
  await window.mouse.down();
  await window.mouse.move(x, to.y + 4, { steps: 12 });
  await window.mouse.up();
}

/**
 * Where `words` are drawn in a line of text: their left and right edges, and
 * their middle line. They may run across marks (AI text and the User's, bold).
 */
async function wordsBox(line: Locator, words: string) {
  const box = await line.evaluate((element, wanted) => {
    const walker = element.ownerDocument.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    const texts: Text[] = [];
    for (let node = walker.nextNode(); node; node = walker.nextNode()) texts.push(node as Text);
    const at = texts
      .map((text) => text.data)
      .join("")
      .indexOf(wanted);
    if (at < 0) return null;
    /** The text and offset `offset` characters in; an end falls at the end of a text, a start at the next. */
    const place = (offset: number, end: boolean) => {
      let start = 0;
      for (const text of texts) {
        if (offset < start + text.length || (end && offset === start + text.length)) {
          return { text, offset: offset - start };
        }
        start += text.length;
      }
      return null;
    };
    const from = place(at, false);
    const to = place(at + wanted.length, true);
    if (!from || !to) return null;
    const range = element.ownerDocument.createRange();
    range.setStart(from.text, from.offset);
    range.setEnd(to.text, to.offset);
    const rect = range.getBoundingClientRect();
    return { left: rect.left, right: rect.right, y: rect.top + rect.height / 2 };
  }, words);
  if (!box) throw new Error(`"${words}" isn't in the line.`);
  return box;
}

/**
 * Selects `words` in a line of text as a person does with the mouse: slides to
 * the first letter, presses, slides along to the end of them and lets go. With
 * `to`, the drag goes on to the end of other words, in a line further down.
 */
export async function slideSelect(
  window: Page,
  line: Locator,
  words: string,
  to?: { line: Locator; words: string },
): Promise<void> {
  const box = await wordsBox(line, words);
  const end = to ? await wordsBox(to.line, to.words) : box;
  await window.mouse.move(box.left - 6, box.y, { steps: 4 });
  await window.mouse.move(box.left + 1, box.y, { steps: 2 });
  await window.mouse.down();
  await window.mouse.move(end.right - 1, end.y, { steps: 12 });
  await window.mouse.up();
  // After a drag, ProseMirror puts its selection back into the page in a task queued as the
  // mouse comes up; a key pressed before that task runs is undone by it. This waits for it.
  await window.evaluate(() => new Promise((resolve) => setTimeout(resolve)));
}

/**
 * Enter in a Mind's title, which takes the cursor into the Mind's text, and
 * waits until it is there: a new Mind's text comes a moment after its title,
 * and until then what is typed still goes into the title.
 */
export async function enterContent(window: Page): Promise<void> {
  await window.getByTestId("mind-title").press("Enter");
  await expect(window.getByTestId("mind-editor")).toBeFocused();
}

/**
 * Puts the cursor at the end of a line of text as a person does: slides to
 * the right of its words and clicks there (Home and End don't move the
 * cursor on macOS).
 */
export async function clickEndOf(window: Page, line: Locator): Promise<void> {
  const box = await line.boundingBox();
  if (!box) throw new Error("The line isn't visible.");
  const x = box.x + box.width - 4;
  const y = box.y + box.height - 10;
  await window.mouse.move(x, y, { steps: 6 });
  await window.mouse.click(x, y);
  // A key pressed next acts where the editor's cursor is: wait until it is in the line.
  await expect
    .poll(() =>
      line.evaluate((element) => {
        const root = element.closest(".ProseMirror") as (Element & { editor?: Editor }) | null;
        const view = root?.editor?.view;
        if (!view) return false;
        const { from, empty } = view.state.selection;
        const at = view.domAtPos(from).node;
        return empty && element.contains(at);
      }),
    )
    .toBe(true);
}

/** Slides the mouse onto an element, in steps, as a person does, and leaves it there. */
export async function pointAt(window: Page, target: Locator): Promise<void> {
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  await window.mouse.move(box.x + Math.min(box.width / 2, 40), box.y + box.height / 2, {
    steps: 8,
  });
}

/** Presses Tab, or Shift+Tab if `target` comes before the focus, until `target` has the focus. */
export async function tabTo(window: Page, target: Locator, max = 40): Promise<void> {
  const key = (await target.evaluate((element) => {
    const focused = document.activeElement;
    return (
      focused !== null &&
      focused !== document.body &&
      (focused.compareDocumentPosition(element) & Node.DOCUMENT_POSITION_PRECEDING) !== 0
    );
  }))
    ? "Shift+Tab"
    : "Tab";
  for (let step = 0; step < max; step++) {
    if (await target.evaluate((element) => element === document.activeElement)) return;
    await window.keyboard.press(key);
  }
  await expect(target).toBeFocused();
}

/** Sets the sidebar's width, as dragging its edge does (165 is the narrowest), and waits for it. */
export async function setSidebarWidth(window: Page, width: number): Promise<void> {
  await window.evaluate(
    (sidebarWidth) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        device: { sidebarWidth },
      }),
    width,
  );
  await expect
    .poll(async () => (await window.getByTestId("sidebar").boundingBox())?.width)
    .toBeCloseTo(width, 0);
}

/** The assistant's Answers (#201), in the conversation it shows. */
export const assistantAnswers = (window: Page) =>
  window.getByTestId("assistant").getByTestId("assistant-answer");

/**
 * Asks a Question as a person does: ⌘J, the words, Enter. It is asked in the
 * assistant's conversation (#201); resolves with its Answer there once it is
 * written (the scripted model's, with `fakeChat`).
 */
export async function askInAssistant(window: Page, question: string): Promise<Locator> {
  const answers = assistantAnswers(window);
  const before = await answers.count();
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type(question);
  await window.keyboard.press("Enter");
  const questions = window.getByTestId("assistant").getByTestId("assistant-question");
  await expect(questions.last()).toContainText(question);
  await expect(answers).toHaveCount(before + 1);
  const answer = answers.last();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  return answer;
}

/** The tiny MCP server the core tests use: `lookup_tide` is read-only, `book_boat` isn't. */
const TIDE_SERVER = resolve(__dirname, "../tests/fixtures/mcp-server.mjs");

/** Adds the tiny server as a Connector through the core's bridge, waits until it's ready, and returns its id. */
export async function addTideConnector(window: Page): Promise<string> {
  return window.evaluate(
    async ({ node, server }) => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const connector = await bridge.addConnector({ name: "Tides", command: node, args: [server] });
      for (let tries = 0; tries < 300; tries++) {
        const found = (await bridge.listConnectors()).find((each) => each.id === connector.id);
        if (found?.state === "ready") return connector.id;
        await new Promise((done) => setTimeout(done, 100));
      }
      throw new Error("The Connector didn't get ready.");
    },
    { node: process.execPath, server: TIDE_SERVER },
  );
}

/**
 * Unfolds an Answer's steps in the assistant (#391) with the mouse, unless
 * they are open, and returns their timeline: the reasoning, the searches,
 * the Skills and Connector calls it made, and the check.
 */
export async function openSteps(window: Page, answer: Locator): Promise<Locator> {
  const toggle = answer.getByTestId("answer-steps");
  if ((await toggle.getAttribute("aria-expanded")) !== "true") await pointAndClick(window, toggle);
  const timeline = answer.getByTestId("answer-timeline");
  await expect(timeline).toBeVisible();
  return timeline;
}

/**
 * Adds an Answer in the assistant to the Mind in focus, at its cursor (or
 * end), with the mouse, as a person keeps one (#201), and waits until it is in.
 */
export async function addToNote(window: Page, answer: Locator): Promise<void> {
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  await pointAndClick(window, answer.getByTestId("answer-add"));
  await expect(answer.getByTestId("answer-added")).toBeVisible();
}

/** A Block of a Mind as Minds held them before Questions went to the assistant (#201). */
export type BlockFromBefore =
  | { note: string }
  | { heading: string }
  | { question: string; answer: string };

/**
 * Makes a Mind as one written before Questions were asked in the assistant
 * (#201) holds them: its Questions and Answers in the note, among its Notes.
 * Made through the core's bridge, in the stored shape. Returns its id.
 */
export async function mindWrittenBefore(
  window: Page,
  title: string,
  blocks: BlockFromBefore[],
): Promise<string> {
  const doc = new Y.Doc();
  const fragment = doc.getXmlFragment("blocks");
  const element = (
    type: string,
    text: string,
    attributes: Record<string, number> = {},
    id: string = crypto.randomUUID(),
  ) => {
    const node = new Y.XmlElement(type);
    node.setAttribute("id", id);
    // Stored as the editor stores them: a heading's level is a number.
    for (const [key, value] of Object.entries(attributes)) {
      node.setAttribute(key, value as unknown as string);
    }
    if (text) node.insert(0, [new Y.XmlText(text)]);
    return node;
  };
  for (const block of blocks) {
    if ("note" in block) fragment.push([element("paragraph", block.note)]);
    else if ("heading" in block) fragment.push([element("heading", block.heading, { level: 2 })]);
    else {
      const questionId = crypto.randomUUID();
      const question = element("question", block.question, {}, questionId);
      const answer = new Y.XmlElement("answer");
      answer.setAttribute("id", crypto.randomUUID());
      answer.setAttribute("questionId", questionId);
      answer.setAttribute("status", "done");
      answer.insert(0, [element("paragraph", block.answer)]);
      fragment.push([question, answer]);
    }
  }
  const update = Array.from(Y.encodeStateAsUpdate(doc));
  return window.evaluate(
    async ([name, change]) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const mind = await core.createMind({ title: name });
      await core.applyMindUpdate(mind.id, new Uint8Array(change));
      await core.closeMind(mind.id);
      return mind.id;
    },
    [title, update] as const,
  );
}

/**
 * Stubs the window's clipboard, so a test can read what Copy wrote and the
 * computer's own clipboard stays as it was. `copied()` gives it, by type.
 */
export async function stubClipboard(window: Page) {
  await window.evaluate(() => {
    const copied: Record<string, string> = {};
    (globalThis as { copied?: unknown }).copied = copied;
    navigator.clipboard.write = async (items: ClipboardItem[]) => {
      for (const item of items) {
        for (const type of item.types) copied[type] = await (await item.getType(type)).text();
      }
    };
  });
  return () =>
    window.evaluate(() => (globalThis as { copied?: Record<string, string> }).copied ?? {});
}

/** A fresh, empty data folder. Remove it with `removeDataFolder`. */
export const createDataFolder = () => mkdtemp(join(tmpdir(), "incarnamind-smoke-"));

/** Closes an app, and kills it if it hasn't quit in 10 s, so one stuck app can't hold up the worker. */
async function closeOrKill(app: ElectronApplication): Promise<void> {
  const quit = app.close().catch(() => undefined);
  let timer: NodeJS.Timeout | undefined;
  const stuck = new Promise<"stuck">((resolve) => {
    timer = setTimeout(() => resolve("stuck"), 10_000);
  });
  if ((await Promise.race([quit, stuck])) === "stuck") app.process().kill("SIGKILL");
  clearTimeout(timer);
}

/**
 * Removes a data folder, after closing an app the test left open on it. If
 * the test failed, a screenshot of that app's window as the test left it is
 * kept first, as app-window.png with the test's results, which CI uploads
 * (Playwright's own screenshots and traces leave an Electron app's window out).
 */
export async function removeDataFolder(dataDir: string): Promise<void> {
  const info = test.info();
  for (const [app, folder] of [...openApps]) {
    if (folder !== dataDir) continue;
    openApps.delete(app);
    const window = app.windows()[0];
    if (info.status !== info.expectedStatus && window) {
      const path = info.outputPath("app-window.png");
      const shot = await window.screenshot({ path }).catch(() => undefined);
      if (shot) await info.attach("app-window", { path, contentType: "image/png" });
    }
    await closeOrKill(app);
  }
  await rm(dataDir, { recursive: true, force: true, maxRetries: 3 });
}

/** The core's bridge in the page, for a test to set things up or read them: `(await coreOf(window)).evaluate(...)`. */
export const coreOf = (window: Page) =>
  window.evaluateHandle(() => (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind);
