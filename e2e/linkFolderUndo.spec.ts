import { existsSync } from "node:fs";
import { mkdir, realpath, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Page, test } from "@playwright/test";
import {
  addDocuments,
  coreOf,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  leaveText,
  linkedFolderRow,
  linkFolderFromSidebar,
  openLinkedFolderMenu,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

let dataDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

/** "Deals", by the path the core keeps: links followed (on macOS the temporary folder is one). */
async function folderOfNotes(): Promise<string> {
  const folder = join(await realpath(sources), "Deals");
  await mkdir(join(folder, "Inner"), { recursive: true });
  await writeFile(join(folder, "Term sheet.md"), "# Term sheet\n\nThe price is 4 million.\n");
  await writeFile(join(folder, "Teaser.md"), "# Teaser\n\nA harbour business for sale.\n");
  await writeFile(join(folder, "Inner", "Memo.md"), "# Memo\n\nThe seller wants a quick close.\n");
  return folder;
}

/** What the Edit menu's Undo would take back now, by its label. */
const undoItem = (app: ElectronApplication) =>
  app.evaluate(({ Menu }) => {
    const item = Menu.getApplicationMenu()?.getMenuItemById("edit.undo");
    return { label: item?.label ?? "", enabled: item?.enabled ?? false };
  });

/** The Library as the core has it: the Linked folders' paths, and each Document with its Linked folder's path. */
const library = async (window: Page) =>
  (await coreOf(window)).evaluate(async (core) => {
    const [documents, linked] = await Promise.all([core.listDocuments(), core.listLinkedFolders()]);
    const pathOf = (id: string | null) => linked.find((each) => each.id === id)?.path ?? null;
    return {
      linked: linked.map((each) => each.path).sort(),
      documents: documents
        .map((each) => ({ name: each.name, id: each.id, in: pathOf(each.linkedFolderId) }))
        .sort((a, b) => a.name.localeCompare(b.name)),
    };
  });

/** An older change that stays on the stack by design: a new Folder. */
async function makeFolder(window: Page): Promise<void> {
  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-new-folder").click();
  const field = window.getByTestId("folder-rename-field");
  await field.fill("Finance");
  await field.press("Enter");
  await expect(field).toHaveCount(0);
}

async function start(app: ElectronApplication, window: Page): Promise<void> {
  await setWindowSize(app, window, 1100, 700);
  await dismissChatSetup(window);
}

test("linking a folder offers Undo; ⌘Z unlinks it, the files stay on disk, and the change before it is next", async () => {
  const folder = await folderOfNotes();
  const { app, window } = await launchApp(dataDir);
  await start(app, window);
  await makeFolder(window);

  await linkFolderFromSidebar(app, window, folder);
  await expect(window.getByTestId("undo-line")).toContainText("Linked “Deals”");
  await expect.poll(async () => (await library(window)).documents).toHaveLength(3);

  await leaveText(window);
  await expect.poll(() => undoItem(app)).toEqual({ label: "Undo Link “Deals”", enabled: true });
  await window.keyboard.press("ControlOrMeta+z");
  await expect(linkedFolderRow(window, "Deals")).toHaveCount(0);
  await expect.poll(() => library(window)).toEqual({ linked: [], documents: [] });
  expect(existsSync(join(folder, "Teaser.md"))).toBe(true);
  await expect.poll(() => undoItem(app)).toEqual({ label: "Undo New Folder", enabled: true });

  await window.keyboard.press("ControlOrMeta+Shift+z");
  await expect(linkedFolderRow(window, "Deals")).toHaveCount(1);
  await expect.poll(async () => (await library(window)).documents).toHaveLength(3);
  await app.close();
});

test("Undo of a link keeps a Document added on its own, also after Redo links it again", async () => {
  const folder = await folderOfNotes();
  const { app, window } = await launchApp(dataDir);
  await start(app, window);

  await linkFolderFromSidebar(app, window, folder);
  await leaveText(window);
  await window.keyboard.press("ControlOrMeta+z");
  await expect(linkedFolderRow(window, "Deals")).toHaveCount(0);
  await addDocuments(window, [join(folder, "Teaser.md")]);
  const teaser = (await library(window)).documents[0];

  await leaveText(window);
  await window.keyboard.press("ControlOrMeta+Shift+z");
  await expect(linkedFolderRow(window, "Deals")).toHaveCount(1);
  await expect
    .poll(async () => (await library(window)).documents)
    .toContainEqual({ name: "Teaser", id: teaser?.id, in: folder });
  await window.keyboard.press("ControlOrMeta+z");

  await expect
    .poll(() => library(window))
    .toEqual({
      linked: [],
      documents: [{ name: "Teaser", id: teaser?.id, in: null }],
    });
  await app.close();
});

test("Undo of a link that merged a Linked folder links that folder again, its Documents kept", async () => {
  const folder = await folderOfNotes();
  const inner = join(folder, "Inner");
  const { app, window } = await launchApp(dataDir);
  await start(app, window);
  await linkFolderFromSidebar(app, window, inner);
  await expect.poll(async () => (await library(window)).documents).toHaveLength(1);
  const memo = (await library(window)).documents[0];

  await linkFolderFromSidebar(app, window, folder);
  await expect.poll(async () => (await library(window)).linked).toEqual([folder]);
  await leaveText(window);
  await window.keyboard.press("ControlOrMeta+z");

  await expect
    .poll(() => library(window))
    .toEqual({
      linked: [inner],
      documents: [{ name: "Memo", id: memo?.id, in: inner }],
    });

  // Linked again under a new id, the inner folder's own link still undoes.
  await expect.poll(() => undoItem(app)).toEqual({ label: "Undo Link “Inner”", enabled: true });
  await window.keyboard.press("ControlOrMeta+z");
  await expect.poll(() => library(window)).toEqual({ linked: [], documents: [] });
  await app.close();
});

test("unlinking a folder that merged another takes both links' Undo away", async () => {
  const folder = await folderOfNotes();
  const { app, window } = await launchApp(dataDir);
  await start(app, window);
  await linkFolderFromSidebar(app, window, join(folder, "Inner"));
  await linkFolderFromSidebar(app, window, folder);
  await expect.poll(async () => (await library(window)).linked).toEqual([folder]);

  const menu = await openLinkedFolderMenu(linkedFolderRow(window, "Deals"));
  await menu.getByTestId("linkedFolder.unlink").click();
  await window.getByTestId("confirm-unlink-folder").click();
  await expect(linkedFolderRow(window, "Deals")).toHaveCount(0);

  await leaveText(window);
  await expect.poll(() => undoItem(app)).toEqual({ label: "Undo", enabled: false });
  await app.close();
});

test("unlinking from the folder's menu takes the link's Undo away", async () => {
  const folder = await folderOfNotes();
  const { app, window } = await launchApp(dataDir);
  await start(app, window);
  await makeFolder(window);
  await linkFolderFromSidebar(app, window, folder);

  const menu = await openLinkedFolderMenu(linkedFolderRow(window, "Deals"));
  await menu.getByTestId("linkedFolder.unlink").click();
  await window.getByTestId("confirm-unlink-folder").click();
  await expect(linkedFolderRow(window, "Deals")).toHaveCount(0);

  await leaveText(window);
  await expect.poll(() => undoItem(app)).toEqual({ label: "Undo New Folder", enabled: true });
  await app.close();
});

for (const [language, refused] of [
  ["en", "Couldn’t do it again: "],
  ["zh-CN", "无法重做："],
] as const) {
  test(`a Redo after the folder was deleted says why in plain words, and nothing is lost (${language})`, async () => {
    const folder = await folderOfNotes();
    const { app, window } = await launchApp(dataDir);
    await start(app, window);
    await setLanguage(window, language);
    await linkFolderFromSidebar(app, window, folder);
    await leaveText(window);
    await window.keyboard.press("ControlOrMeta+z");
    await expect(linkedFolderRow(window, "Deals")).toHaveCount(0);
    await rm(folder, { recursive: true });

    await window.keyboard.press("ControlOrMeta+Shift+z");

    const line = window.getByTestId("undo-line");
    await expect(line).toContainText(refused);
    await expect(line).not.toContainText("Error");
    expect(await library(window)).toEqual({ linked: [], documents: [] });
    await app.close();
  });
}
