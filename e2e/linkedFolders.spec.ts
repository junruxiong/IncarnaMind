import { createHash } from "node:crypto";
import { lstat, mkdir, readdir, readFile, realpath, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge, LinkedFolder } from "../src/core/api";
import {
  confirmLink,
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  interceptOpenPath,
  interceptShowItemInFolder,
  launchApp,
  linkedFolderRow,
  linkFolderFromSidebar,
  openDocumentMenu,
  openLinkedFolderMenu,
  pathsOpened,
  pathsShown,
  previewLink,
  removeDataFolder,
} from "./app";

/*
 * Linked folders in the sidebar (ADR-0010): "Add folder…" previews what
 * linking takes before anything is read, the folder's row says how it is
 * (indexing, paused, unavailable, online-only files, nothing in it), its
 * menu pauses, re-lays out, shows and unlinks it, and its Documents say
 * when their file is missing or can't be reached. Set INCARNAMIND_SCREENSHOTS
 * to a folder to also save screenshots of each there.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

let dataDir: string;
/** Where the Linked folders are made: outside the data folder. */
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await realpath(await createDataFolder());
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

async function screenshot(target: Page | Locator, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

/** The version of the Document at `path` that is indexed, through the core's bridge. */
const contentHashOf = (window: Page, path: string) =>
  window.evaluate(async (wanted) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return (await bridge.listDocuments()).find((each) => each.path === wanted)?.contentHash;
  }, path);

/** The Linked folder at `path`, through the core's bridge. */
const linkedAt = (window: Page, path: string) =>
  window.evaluate(async (wanted) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return (await bridge.listLinkedFolders()).find((each) => each.path === wanted);
  }, path) as Promise<LinkedFolder | undefined>;

/** Writes `files` (relative paths) under `root`, each holding `text` or its own name. */
async function writeTree(root: string, files: string[], text?: string): Promise<void> {
  for (const file of files) {
    await mkdir(dirname(join(root, file)), { recursive: true });
    await writeFile(join(root, file), text ?? `${file}.\n`);
  }
}

/** Long enough for many Passages, so indexing a few of these stays on screen for a moment. */
const LONG_TEXT = Array.from(
  { length: 4000 },
  (_, index) => `Line ${index}: test loss falls as a power law in model size and data.`,
).join("\n");

/** Every file and folder under `root`, with its size, modified time, mode and content hash. */
async function snapshot(root: string): Promise<string[]> {
  const entries: string[] = [];
  const visit = async (folder: string) => {
    for (const name of (await readdir(folder)).sort()) {
      const path = join(folder, name);
      const info = await lstat(path);
      if (info.isDirectory()) {
        entries.push(`${path}/ ${info.mtimeMs}`);
        await visit(path);
      } else {
        const hash = createHash("sha256")
          .update(await readFile(path))
          .digest("hex");
        entries.push(`${path} ${info.size} ${info.mtimeMs} ${info.mode} ${hash}`);
      }
    }
  };
  await visit(root);
  return entries;
}

test("a linked folder's files become Documents, and follow edits and deletions on disk", async () => {
  const library = join(sources, "Library");
  await mkdir(join(library, "Sub"), { recursive: true });
  const notes = join(library, "Reading notes.md");
  const report = join(library, "Sub", "Report.txt");
  await writeFile(
    notes,
    "# Reading notes\n\nAttention lets a model focus on the relevant words.\n",
  );
  await writeFile(report, "Revenue grew by ten percent.\n");

  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  // "Add folder…" asks with the system's folder picker, which the test answers, then the link dialog.
  await linkFolderFromSidebar(app, window, library);

  const items = window.getByTestId("document-list-item");
  await expect(items).toHaveCount(2);
  const notesItem = items.filter({ hasText: "Reading notes" });
  const reportItem = items.filter({ hasText: "Report" });
  await expect(notesItem).toHaveAttribute("data-status", "ready");
  await expect(reportItem).toHaveAttribute("data-status", "ready");
  await expect(notesItem).toHaveAttribute("data-file-status", "available");
  // The Linked folder is a Folder, with its own folders inside.
  await expect(window.getByTestId("folder-item").filter({ hasText: "Library" })).toHaveCount(1);

  // Edited on disk: indexed again, as a new version, while the app runs.
  const before = await contentHashOf(window, notes);
  await writeFile(
    notes,
    "# Reading notes\n\nSelf-attention compares every word with every other.\n",
  );
  await expect.poll(() => contentHashOf(window, notes), { timeout: 15_000 }).not.toBe(before);
  await expect(notesItem).toHaveAttribute("data-status", "ready");
  await expect
    .poll(() =>
      window.evaluate(async () => {
        const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
        return (await bridge.searchPassages("Self-attention", { mode: "keyword" })).length;
      }),
    )
    .toBe(1);

  // Deleted on disk: still listed, as missing.
  await rm(report);
  await expect(reportItem).toHaveAttribute("data-file-status", "missing", { timeout: 15_000 });
  await expect(items).toHaveCount(2);
  await expect(notesItem).toHaveAttribute("data-file-status", "available");
  await app.close();
});

test("Add folder… shows what linking would take, then the folder's row shows indexing until it's done", async () => {
  const library = join(sources, "Library");
  await writeTree(
    library,
    ["Scaling laws.txt", "Chinchilla.txt", "2026/Attention.txt", "2026/Retrieval.txt"],
    LONG_TEXT,
  );
  // iCloud Drive's stubs for two files that aren't downloaded: never read, only counted.
  await writeTree(library, [".Remote paper.pdf.icloud", "2026/.Remote notes.md.icloud"], "plist");

  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await interceptShowItemInFolder(app);
  const dialog = await previewLink(app, window, library);

  // The folder, what it holds and how long indexing takes, the files it skips, and how it shows.
  await expect(dialog.getByTestId("link-folder-name")).toHaveText("Library");
  await expect(dialog.getByTestId("link-folder-path")).toHaveText(library);
  await expect(dialog.getByTestId("link-folder-files")).toHaveText(/^4 supported files, 1\.\d MB$/);
  await expect(dialog.getByTestId("link-folder-time")).toContainText(
    /Indexing takes (under a minute|about .+) on this computer, newest files first/,
  );
  await expect(dialog.getByTestId("link-folder-online-only")).toContainText(
    "2 files are online only",
  );
  await expect(dialog.getByTestId("link-folder-layout-tree")).toBeChecked();
  await expect(dialog.getByTestId("link-folder-merge")).toHaveCount(0);
  await screenshot(window, "link-dialog");
  // Nothing is linked before the User says so.
  expect(await linkedAt(window, library)).toBeUndefined();
  await confirmLink(dialog);

  // Indexing: its count and a thin bar on the folder's one 28px row.
  const row = linkedFolderRow(window, "Library");
  await expect(row).toHaveAttribute("data-state", "indexing");
  await expect(row.getByTestId("folder-status")).toContainText(/Indexing \d of 4/);
  await expect(row.getByTestId("folder-progress")).toBeVisible();
  expect((await row.boundingBox())?.height).toBeCloseTo(28, 0);
  await screenshot(window.getByTestId("sidebar"), "row-indexing");

  // Done: the bar goes, and the row says what it skipped.
  await expect(row).toHaveAttribute("data-state", "online-only", { timeout: 60_000 });
  await expect(row.getByTestId("folder-status")).toContainText("2 online-only");
  await expect(row.getByTestId("folder-progress")).toHaveCount(0);
  await expect(window.getByTestId("document-list-item")).toHaveCount(4);

  // Its menu offers to download and index them (not done here: it would ask iCloud Drive).
  const menu = await openLinkedFolderMenu(row);
  await expect(menu.getByTestId("linked-folder-download")).toHaveText(
    "Download and index 2 online-only files",
  );
  await expect(menu.getByTestId("linked-folder-pause")).toHaveText("Pause indexing");
  await expect(menu.getByTestId("linked-folder-layout")).toHaveText("Show as a flat list");
  await screenshot(window, "row-menu");
  // Show in Finder (Explorer, the file manager) shows the folder itself, never anything else.
  await menu.getByTestId("linked-folder-reveal").click();
  await expect.poll(() => pathsShown(app)).toEqual([library]);
  await app.close();
});

test("pausing a Linked folder holds its indexing until it is resumed", async () => {
  // Indexing the five long files, after the pause, takes a while with the test's embedding model.
  test.setTimeout(150_000);
  const library = join(sources, "Big library");
  await writeTree(
    library,
    Array.from({ length: 5 }, (_, index) => `Paper ${index + 1}.txt`),
    LONG_TEXT,
  );
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await linkFolderFromSidebar(app, window, library);
  const row = linkedFolderRow(window, "Big library");
  await expect(row).toHaveAttribute("data-state", "indexing");

  let menu = await openLinkedFolderMenu(row);
  await menu.getByTestId("linked-folder-pause").click();
  await expect(row).toHaveAttribute("data-state", "paused");
  await expect(row.getByTestId("folder-status")).toHaveText(/^Paused/);
  const paused = await linkedAt(window, library);
  expect(paused?.status).toBe("paused");
  expect(paused?.progress.indexed).toBeLessThan(paused?.progress.files ?? 0);
  // The bar stays where it stopped, greyed, and the footer doesn't count its files as processing.
  await expect(row.getByTestId("folder-progress")).toBeVisible();
  await expect(window.getByTestId("sidebar-status")).not.toContainText("Processing");
  await screenshot(window.getByTestId("sidebar"), "row-paused");

  menu = await openLinkedFolderMenu(row);
  await expect(menu.getByTestId("linked-folder-pause")).toHaveText("Resume indexing");
  await menu.getByTestId("linked-folder-pause").click();
  await expect(row).not.toHaveAttribute("data-state", "paused");
  await expect(row).toHaveAttribute("data-state", "idle", { timeout: 120_000 });
  await expect(row.getByTestId("folder-status")).toHaveCount(0);
  expect((await linkedAt(window, library))?.progress).toEqual({ files: 5, indexed: 5 });
  await app.close();
});

test("a Linked folder shows as Folders or a flat list, chosen when linking and from its menu", async () => {
  // Zotero's storage: a folder per item, one file in each. Linking suggests a flat list.
  const storage = join(sources, "storage");
  await writeTree(storage, [
    "ABCD1234/Attention Is All You Need.txt",
    "EFGH5678/Scaling Laws.txt",
    "IJKL9012/Chinchilla.txt",
    "MNOP3456/Retrieval-Augmented Generation.txt",
  ]);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const dialog = await previewLink(app, window, storage);
  await expect(dialog.getByTestId("link-folder-layout-flat")).toBeChecked();
  await expect(dialog.getByText("Suggested")).toHaveCount(1);
  // The User prefers its folders, as on disk.
  await dialog.getByTestId("link-folder-layout-tree").check();
  await confirmLink(dialog);

  const row = linkedFolderRow(window, "storage");
  const documents = window.getByTestId("document-list-item");
  const subfolders = window.locator('[data-testid="folder-item"]:not([data-root])');
  await expect(row).toHaveAttribute("data-layout", "tree");
  await expect(documents).toHaveCount(4);
  await expect(subfolders).toHaveCount(4);
  await expect(documents.first()).toHaveAttribute("data-depth", "2");

  // A flat list: every Document right under the folder, no subfolders.
  let menu = await openLinkedFolderMenu(row);
  await menu.getByTestId("linked-folder-layout").click();
  await expect(row).toHaveAttribute("data-layout", "flat");
  await expect(subfolders).toHaveCount(0);
  await expect(documents).toHaveCount(4);
  for (const item of await documents.all()) await expect(item).toHaveAttribute("data-depth", "1");
  await screenshot(window.getByTestId("sidebar"), "row-flat");

  // And back.
  menu = await openLinkedFolderMenu(row);
  await expect(menu.getByTestId("linked-folder-layout")).toHaveText("Show as folders");
  await menu.getByTestId("linked-folder-layout").click();
  await expect(row).toHaveAttribute("data-layout", "tree");
  await expect(subfolders).toHaveCount(4);
  await app.close();
});

test("a file deleted on disk shows as missing, still opens, and can be removed from IncarnaMind", async () => {
  const library = join(sources, "Library");
  await writeTree(library, ["Kept.txt", "Deleted later.txt"]);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await interceptOpenPath(app);
  await interceptShowItemInFolder(app);
  await linkFolderFromSidebar(app, window, library);
  const row = linkedFolderRow(window, "Library");
  await expect(row).toHaveAttribute("data-state", "idle");
  const items = window.getByTestId("document-list-item");
  const doomed = items.filter({ hasText: "Deleted later" });
  await expect(doomed).toHaveAttribute("data-status", "ready");

  await rm(join(library, "Deleted later.txt"));
  await expect(doomed).toHaveAttribute("data-file-status", "missing", { timeout: 15_000 });
  // Muted, with "Missing" at its end, on its one 28px line.
  const status = doomed.getByTestId("document-status");
  await expect(status).toHaveText(/^Missing/);
  await expect(status).toHaveAttribute("title", /File missing/);
  await expect(doomed).toHaveCSS("color", "rgb(101, 107, 116)");
  expect((await doomed.boundingBox())?.height).toBeCloseTo(28, 0);
  await screenshot(window.getByTestId("sidebar"), "document-missing");

  // Opening it still opens the viewer, as for any Document.
  await doomed.getByTestId("open-document").click();
  await expect(window.getByTestId("viewer")).toBeVisible();

  // Its file can't be opened or shown: the items say why, and do nothing.
  await openDocumentMenu(doomed);
  const actions = doomed.getByTestId("document-file-actions");
  for (const id of ["document-open-externally", "document-show-in-folder"]) {
    const item = actions.getByTestId(id);
    await expect(item).toHaveAttribute("aria-disabled", "true");
    await expect(item).toHaveAttribute("title", "The file is missing from its folder.");
  }
  await screenshot(window, "document-missing-menu");
  // Clicked anyway (Playwright would wait for them to be enabled): nothing happens.
  await actions.getByTestId("document-open-externally").click({ force: true });
  await actions.getByTestId("document-show-in-folder").click({ force: true });
  expect(await pathsOpened(app)).toEqual([]);
  expect(await pathsShown(app)).toEqual([]);

  // "Remove from IncarnaMind…" asks first, then the row goes.
  await actions.getByTestId("remove-document").click();
  const confirm = window.getByTestId("delete-document-dialog");
  await expect(confirm.getByRole("heading")).toHaveText("Remove this Document?");
  await screenshot(window, "document-remove-dialog");
  await confirm.getByTestId("confirm-delete-document").click();
  await expect(doomed).toHaveCount(0);
  await expect(items).toHaveCount(1);
  await app.close();
});

test("unlinking asks first, and leaves the folder on disk exactly as it was", async () => {
  const library = join(sources, "Library");
  await writeTree(library, ["Paper.txt", "Notes/Reading notes.md", "Notes/Deep/Draft.txt"]);
  const before = await snapshot(library);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await linkFolderFromSidebar(app, window, library);
  const row = linkedFolderRow(window, "Library");
  await expect(row).toHaveAttribute("data-state", "idle");
  const items = window.getByTestId("document-list-item");
  await expect(items).toHaveCount(3);

  // Cancel changes nothing.
  let menu = await openLinkedFolderMenu(row);
  await menu.getByTestId("unlink-folder").click();
  const dialog = window.getByTestId("unlink-folder-dialog");
  await expect(dialog.getByRole("heading")).toHaveText("Unlink “Library”?");
  await expect(dialog).toContainText("The folder and the files in it aren't touched.");
  await expect(dialog).toContainText("Citations to its Documents stay in your Minds");
  await screenshot(window, "unlink-dialog");
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).toBeHidden();
  await expect(items).toHaveCount(3);

  menu = await openLinkedFolderMenu(row);
  await menu.getByTestId("unlink-folder").click();
  await dialog.getByTestId("confirm-unlink-folder").click();
  await expect(row).toHaveCount(0);
  await expect(items).toHaveCount(0);
  expect(await linkedAt(window, library)).toBeUndefined();
  // Nothing linked and nothing added: the section says how to start again.
  await expect(window.getByTestId("documents-empty")).toBeVisible();
  // Byte for byte, time for time, as it was.
  expect(await snapshot(library)).toEqual(before);
  await app.close();
});

test("a Linked folder that can't be reached shows as unavailable, and so do its Documents", async () => {
  const drive = join(sources, "Drive");
  await writeTree(drive, ["Field notes.txt", "Survey.md"]);
  let running = await launchApp(dataDir);
  await dismissChatSetup(running.window);
  await linkFolderFromSidebar(running.app, running.window, drive);
  await expect(linkedFolderRow(running.window, "Drive")).toHaveAttribute("data-state", "idle");
  await running.app.close();

  // The drive is unplugged while the app is closed.
  await rename(drive, join(sources, "Drive (unplugged)"));
  running = await launchApp(dataDir);
  const { window } = running;
  const row = linkedFolderRow(window, "Drive");
  await expect(row).toHaveAttribute("data-state", "unavailable", { timeout: 15_000 });
  await expect(row.getByTestId("folder-status")).toHaveText(/^Unavailable/);
  await expect(row).toHaveCSS("color", "rgb(101, 107, 116)");
  const items = window.getByTestId("document-list-item");
  await expect(items).toHaveCount(2);
  for (const item of await items.all()) {
    await expect(item).toHaveAttribute("data-file-status", "unavailable");
    await expect(item.getByTestId("document-status")).toHaveText(/^Unavailable/);
  }
  await screenshot(window.getByTestId("sidebar"), "row-unavailable");

  const menu = await openLinkedFolderMenu(row);
  await expect(menu.getByTestId("linked-folder-reveal")).toHaveAttribute("aria-disabled", "true");
  await expect(menu.getByTestId("linked-folder-reveal")).toHaveAttribute(
    "title",
    "The folder can't be reached right now.",
  );
  await window.keyboard.press("Escape");
  await openDocumentMenu(items.first());
  await expect(
    items.first().getByTestId("document-file-actions").getByTestId("document-open-externally"),
  ).toHaveAttribute("title", "The file can't be reached right now.");
  await running.app.close();
});

test("an empty Linked folder says so; with nothing linked or added, the section says how to start", async () => {
  const empty = join(sources, "Projects");
  await mkdir(empty);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);

  const start = window.getByTestId("documents-empty");
  await expect(start).toBeVisible();
  await expect(start.getByTestId("empty-add-linked-folder")).toHaveText("Add folder…");
  await expect(start.getByTestId("empty-add-documents")).toHaveText("Add Documents");
  await screenshot(window.getByTestId("sidebar"), "documents-empty");

  // Its "Add folder…" links a folder as the header's does.
  const dialog = await previewLink(
    app,
    window,
    empty,
    start.getByTestId("empty-add-linked-folder"),
  );
  await expect(dialog.getByTestId("link-folder-files")).toHaveText(
    "No supported files in it yet. Files added to it later are indexed then.",
  );
  await expect(dialog.getByTestId("link-folder-time")).toHaveCount(0);
  await confirmLink(dialog);

  const row = linkedFolderRow(window, "Projects");
  await expect(row).toHaveAttribute("data-state", "empty");
  await expect(row.getByTestId("folder-status")).toHaveText(/^No supported files/);
  // Nothing to unfold: no chevron, no toggle.
  await expect(row.getByTestId("folder-toggle")).toHaveCount(0);
  await expect(start).toHaveCount(0);
  await screenshot(window.getByTestId("sidebar"), "row-empty");
  await app.close();
});

test("linking a folder that overlaps a Linked folder says what happens: they merge, or nothing changes", async () => {
  const papers = join(sources, "Papers");
  await writeTree(papers, ["Survey.txt", "2026/Attention.txt", "2027/Scaling.txt"]);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await linkFolderFromSidebar(app, window, join(papers, "2026"));
  await expect(linkedFolderRow(window, "2026")).toHaveAttribute("data-state", "idle");

  // A folder inside it is linked already: the dialog says so, and only closes.
  await interceptOpenDialog(app, join(papers, "2026"));
  await window.getByTestId("add-linked-folder").click();
  const dialog = window.getByTestId("link-folder-dialog");
  await expect(dialog.getByTestId("link-folder-inside")).toContainText(
    "This folder is inside “2026”, which is linked already",
  );
  await expect(dialog.getByTestId("link-folder-confirm")).toHaveCount(0);
  await dialog.getByRole("button", { name: "Close" }).click();
  await expect(dialog).toBeHidden();

  // The folder around it: the Linked folder inside merges into it, its Documents with it.
  const around = await previewLink(app, window, papers);
  await expect(around.getByTestId("link-folder-merge")).toContainText(
    "“2026”, linked already, is inside this folder: it merges into this one",
  );
  await screenshot(window, "link-dialog-merge");
  await confirmLink(around);
  await expect(linkedFolderRow(window, "Papers")).toHaveAttribute("data-state", "idle");
  await expect(linkedFolderRow(window, "2026")).toHaveCount(0);
  await expect(window.getByTestId("document-list-item")).toHaveCount(3);
  await app.close();
});
