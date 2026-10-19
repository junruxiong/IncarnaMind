import { mkdir, writeFile } from "node:fs/promises";
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
  newMind,
  removeDataFolder,
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

async function folderOfNotes(): Promise<string> {
  const folder = join(sources, "Deals");
  await mkdir(folder, { recursive: true });
  await writeFile(join(folder, "Term sheet.md"), "# Term sheet\n\nThe price is 4 million.\n");
  await writeFile(join(folder, "Teaser.md"), "# Teaser\n\nA harbour business for sale.\n");
  return folder;
}

const mindCount = async (window: Page) =>
  (await coreOf(window)).evaluate(async (core) => (await core.listMinds()).length);

/** What the Edit menu's Undo would take back now, by its label. */
const undoItem = (app: ElectronApplication) =>
  app.evaluate(({ Menu }) => {
    const item = Menu.getApplicationMenu()?.getMenuItemById("edit.undo");
    return { label: item?.label ?? "", enabled: item?.enabled ?? false };
  });

test("linking a folder offers Undo, and ⌘Z unlinks it rather than an older change", async () => {
  const folder = await folderOfNotes();
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 700);
    await dismissChatSetup(window);
    await newMind(window);
    const minds = await mindCount(window);

    await linkFolderFromSidebar(app, window, folder);
    const line = window.getByTestId("undo-line");
    await expect(line).toContainText("Linked “Deals”");
    await expect(linkedFolderRow(window, "Deals")).toHaveCount(1);

    await leaveText(window);
    await expect.poll(() => undoItem(app)).toEqual({ label: "Undo Link Folder", enabled: true });
    await window.keyboard.press("ControlOrMeta+z");
    await expect(linkedFolderRow(window, "Deals")).toHaveCount(0);
    expect(await mindCount(window)).toBe(minds);

    await window.keyboard.press("ControlOrMeta+Shift+z");
    await expect(linkedFolderRow(window, "Deals")).toHaveCount(1);
  } finally {
    await app.close();
  }
});

test("a link that takes over Documents added before says Undo can't take it back, and ⌘Z reaches nothing older", async () => {
  const folder = await folderOfNotes();
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 700);
    await dismissChatSetup(window);
    await addDocuments(window, [join(folder, "Teaser.md")]);
    await newMind(window);
    const minds = await mindCount(window);

    await linkFolderFromSidebar(app, window, folder);
    await expect(window.getByTestId("undo-line")).toContainText(
      "It took over Documents you had added from it, so Undo can't take it back",
    );

    await leaveText(window);
    await expect.poll(() => undoItem(app)).toEqual({ label: "Undo", enabled: false });
    await window.keyboard.press("ControlOrMeta+z");
    await expect(linkedFolderRow(window, "Deals")).toHaveCount(1);
    expect(await mindCount(window)).toBe(minds);
  } finally {
    await app.close();
  }
});
