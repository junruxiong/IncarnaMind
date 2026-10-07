import { mkdir, realpath, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  removeDataFolder,
} from "./app";

let dataDir: string;
/** Where the Linked folder is made: outside the data folder. */
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

/** The version of the Document at `path` that is indexed, through the core's bridge. */
const contentHashOf = (window: Page, path: string) =>
  window.evaluate(async (wanted) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return (await bridge.listDocuments()).find((each) => each.path === wanted)?.contentHash;
  }, path);

test("a linked folder's files become Documents, and follow edits and deletions on disk", async () => {
  const library = join(await realpath(sources), "Library");
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
  // "Add folder…" asks with the system's folder picker, which the test answers.
  await interceptOpenDialog(app, library);
  await window.getByTestId("add-linked-folder").click();

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
