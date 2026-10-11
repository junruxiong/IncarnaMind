import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  sidebarFolder,
} from "./app";

/*
 * Folders are projects (#111): the sidebar's one tree lists each Folder's
 * Minds above its Documents, and Not in a Folder holds the rest. A Folder
 * opens in place; a Mind made in it asks within it; Minds move by their
 * menu or by dragging, with Undo; a Folder renames in place; deleting one
 * moves its Minds out. Set INCARNAMIND_SCREENSHOTS to a folder to also save
 * screenshots of the sidebar there.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

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

async function screenshot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const bridge = (window: Page) => ({
  createFolder: (name: string) =>
    window.evaluate(
      async (folderName) =>
        (
          await (
            globalThis as unknown as { incarnamind: CoreBridge }
          ).incarnamind.createLibraryGroup({ name: folderName, description: "" })
        ).id,
      name,
    ),
  file: (documentName: string, folderId: string | null) =>
    window.evaluate(
      async ({ name, folder }) => {
        const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
        const document = (await core.listDocuments()).find((each) => each.name === name);
        if (!document) throw new Error(`No Document named ${name}.`);
        await core.assignDocumentGroup(document.id, folder);
      },
      { name: documentName, folder: folderId },
    ),
});

/** Moves the mouse onto an element in small steps, as a person does. */
async function slideTo(window: Page, target: Locator): Promise<{ x: number; y: number }> {
  const box = await target.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  const point = { x: box.x + Math.min(40, box.width / 2), y: box.y + box.height / 2 };
  await window.mouse.move(point.x, point.y, { steps: 8 });
  return point;
}

const notInAFolder = (window: Page) => window.getByTestId("not-in-a-folder");
const mindRow = (window: Page, title: string) =>
  window.getByTestId("mind-row").filter({ hasText: title });

test("each Folder lists its Minds above its Documents and opens in place; a Mind made in it asks within it, resolved when asked", async () => {
  await writeFile(
    join(sources, "Tide tables.txt"),
    "Neap tides are the smallest of the month.\nSpring tides happen at new moon and at full moon.\n",
  );
  await writeFile(
    join(sources, "Spring menu.txt"),
    "Spring tides happen when the market sells mussels.\n",
  );
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await dismissChatSetup(window);
    await connectLocalChatModel(window);
    const ocean = await bridge(window).createFolder("Ocean research");
    await window.evaluate(
      async (paths) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.addDocuments(paths),
      [join(sources, "Tide tables.txt"), join(sources, "Spring menu.txt")],
    );
    await expect(
      window.locator('[data-testid="document-list-item"][data-status="ready"]'),
    ).toHaveCount(2);
    await bridge(window).file("Tide tables", ocean);

    // One tree: the Folder, closed, with its count; Not in a Folder, which searches everything.
    const folder = sidebarFolder(window, "Ocean research");
    await expect(folder).toHaveAttribute("data-expanded", "false");
    await expect(folder.getByTestId("browse-count")).toHaveText("1");
    await expect(notInAFolder(window)).toContainText("Not in a Folder");
    await expect(window.getByTestId("not-in-a-folder-hint")).toHaveText("Searches everything");
    await expect(
      notInAFolder(window).getByTestId("document-list-item").getByTestId("row-text"),
    ).toHaveText("Spring menu");

    // Clicked, the Folder opens in place: the main area stays as it was.
    await expect(window.getByTestId("mind-none-open")).toBeVisible();
    await slideTo(window, folder.getByTestId("folder-row"));
    await window.mouse.down();
    await window.mouse.up();
    await expect(folder).toHaveAttribute("data-expanded", "true");
    await expect(folder.getByTestId("folder-row")).toHaveAttribute("aria-expanded", "true");
    await expect(folder.getByTestId("document-list-item").getByTestId("row-text")).toHaveText(
      "Tide tables",
    );
    await expect(window.getByTestId("mind-none-open")).toBeVisible();
    await expect(window.getByTestId("library")).toHaveCount(0);

    // "New Mind here", from its menu (a right-click, as in Finder): listed in the Folder,
    // above its Documents, and open.
    await folder.getByTestId("folder-row").click({ button: "right" });
    await folder.getByTestId("folder-new-mind-here").click();
    await window.getByTestId("mind-title").fill("Tide questions");
    const listed = folder.locator(":scope > ul > li");
    await expect(listed.first()).toHaveAttribute("data-testid", "mind-row");
    await expect(listed.first()).toContainText("Tide questions");
    await expect(listed.nth(1)).toHaveAttribute("data-testid", "document-list-item");
    await screenshot(window, "folder-tree");

    // Its Questions start with the Folder as their Search scope: a chip that says so.
    const editor = window.getByTestId("mind-editor");
    await editor.click();
    await window.keyboard.press("ControlOrMeta+j");
    const composer = window.getByTestId("composer");
    const chip = composer.getByTestId("scope-chip");
    await expect(chip).toHaveText("Ocean research");
    await expect(chip).toHaveAttribute("data-own", "true");
    await expect(chip).toHaveAttribute(
      "title",
      "Searches the 1 Document in this Folder. Remove it to search everything.",
    );
    await window.keyboard.type("When do spring tides happen?");
    await window.keyboard.press("Enter");
    const answers = editor.getByTestId("answer");
    await expect(answers.first()).toHaveAttribute("data-status", "done", { timeout: 15_000 });
    await expect(editor.getByTestId("question").first().getByTestId("scope-chip")).toHaveText(
      "Ocean research",
    );
    await expect(answers.first().getByTestId("citation-chip")).toHaveAttribute(
      "aria-label",
      /^Citation 1: Tide tables/,
    );

    // Filed out and in since, the Folder is searched as it is when asked.
    await bridge(window).file("Tide tables", null);
    await bridge(window).file("Spring menu", ocean);
    await expect(folder.getByTestId("document-list-item").getByTestId("row-text")).toHaveText(
      "Spring menu",
    );
    await composer.getByTestId("composer-input").click();
    await window.keyboard.type("When do spring tides happen, again?");
    await window.keyboard.press("Enter");
    await expect(answers.nth(1)).toHaveAttribute("data-status", "done", { timeout: 15_000 });
    await expect(answers.nth(1).getByTestId("citation-chip")).toHaveAttribute(
      "aria-label",
      /^Citation 1: Spring menu/,
    );

    // Its × takes the Folder out: every Document is searched.
    await chip.getByTestId("scope-chip-remove").click();
    await expect(composer.getByTestId("composer-scope-all")).toBeVisible();
    await composer.getByTestId("composer-input").click();
    await window.keyboard.type("When do spring tides happen, anywhere?");
    await window.keyboard.press("Enter");
    const third = answers.nth(2);
    await expect(third).toHaveAttribute("data-status", "done", { timeout: 15_000 });
    await expect(editor.getByTestId("question").nth(2).getByTestId("question-scope")).toHaveCount(
      0,
    );
    await third.getByTestId("answer-tools-toggle").click();
    await expect(third.getByTestId("answer-tool-call")).toContainText("2 Passages");

    // Clicked again, the Folder closes in place.
    await folder.getByTestId("folder-row").click();
    await expect(folder).toHaveAttribute("data-expanded", "false");
    await expect(folder.getByTestId("mind-row")).toHaveCount(0);
  } finally {
    await app.close();
  }
});

test("a Mind moves by its menu, by the keyboard and by dragging, each with Undo; a Folder renames in place; a deleted Folder's Minds go to Not in a Folder", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await dismissChatSetup(window);
    await connectLocalChatModel(window);
    await bridge(window).createFolder("Finance");
    await bridge(window).createFolder("Research");
    await newMind(window);
    await window.getByTestId("mind-title").fill("Shortlist");
    const shortlist = mindRow(window, "Shortlist");
    await expect(notInAFolder(window).getByTestId("mind-list-item")).toHaveText("Shortlist");
    const finance = sidebarFolder(window, "Finance");
    const research = sidebarFolder(window, "Research");
    const undoLine = window.getByTestId("undo-line");

    // ↑ and ↓ move through the tree's rows: the Folders, Not in a Folder, then its Mind.
    await finance.getByTestId("folder-row").focus();
    await window.keyboard.press("ArrowDown");
    await expect(research.getByTestId("folder-row")).toBeFocused();
    await window.keyboard.press("ArrowDown");
    await expect(window.getByTestId("not-in-a-folder-toggle")).toBeFocused();
    await window.keyboard.press("ArrowDown");
    await expect(shortlist.getByTestId("mind-list-item")).toBeFocused();
    await window.keyboard.press("ArrowUp");
    await expect(window.getByTestId("not-in-a-folder-toggle")).toBeFocused();

    // Its menu's "Move to…": a Folder picker with a field to find one.
    await shortlist.hover();
    await shortlist.getByTestId("mind-menu").click();
    const actions = shortlist.getByTestId("mind-actions");
    // In the shared order, each with its shortcut.
    await expect(actions.getByRole("menuitem")).toHaveText([
      /^Open$/,
      /^Open in a new tab$/,
      /^Move to…(⇧⌘M|Ctrl\+Shift\+M)$/,
      /^Rename(↩|F2)$/,
      /^Delete…$/,
    ]);
    await actions.getByTestId("mind-move-to").click();
    const picker = window.getByTestId("move-to-picker");
    await expect(picker).toBeVisible();
    await expect(picker.getByTestId("move-to-search")).toBeFocused();
    await expect(picker.getByTestId("move-to-option")).toHaveText([
      "Not in a Folder",
      "Finance",
      "Research",
    ]);
    await window.keyboard.type("fin");
    await expect(picker.getByTestId("move-to-option")).toHaveText(["Finance"]);
    await window.keyboard.press("Enter");
    await expect(picker).toBeHidden();
    // In Finance, which opens to show it, and a line to take it back.
    await expect(finance.getByTestId("mind-list-item")).toHaveText("Shortlist");
    await expect(notInAFolder(window).getByTestId("mind-row")).toHaveCount(0);
    await expect(undoLine.getByTestId("undo-message")).toHaveText("Moved “Shortlist” to Finance");
    // ⌘Z (Ctrl+Z), outside a text field, takes it back.
    await window.keyboard.press("ControlOrMeta+z");
    await expect(notInAFolder(window).getByTestId("mind-list-item")).toHaveText("Shortlist");
    await expect(undoLine.getByTestId("undo-message")).toHaveCount(0);

    // By the keyboard alone: Tab to the row's ⋯, Enter, down to Move to…, type, Enter.
    await shortlist.getByTestId("mind-list-item").focus();
    await window.keyboard.press("Tab");
    await expect(shortlist.getByTestId("mind-menu")).toBeFocused();
    await window.keyboard.press("Enter");
    await expect(actions.getByTestId("mind-open")).toBeFocused();
    for (let step = 0; step < 2; step++) await window.keyboard.press("ArrowDown");
    await expect(actions.getByTestId("mind-move-to")).toBeFocused();
    await window.keyboard.press("Enter");
    await expect(picker.getByTestId("move-to-search")).toBeFocused();
    await window.keyboard.type("res");
    await window.keyboard.press("Enter");
    await expect(research.getByTestId("mind-list-item")).toHaveText("Shortlist");
    await undoLine.getByTestId("undo-action").click();
    await expect(notInAFolder(window).getByTestId("mind-list-item")).toHaveText("Shortlist");

    // ⇧⌘M (Ctrl+Shift+M) on the row opens the same picker; Esc closes it, moving nothing.
    await shortlist.getByTestId("mind-list-item").focus();
    await window.keyboard.press("ControlOrMeta+Shift+KeyM");
    await expect(picker.getByTestId("move-to-search")).toBeFocused();
    await window.keyboard.press("Escape");
    await expect(picker).toBeHidden();
    await expect(shortlist.getByTestId("mind-list-item")).toBeFocused();
    await expect(notInAFolder(window).getByTestId("mind-list-item")).toHaveText("Shortlist");

    // Dragged with the mouse onto a Folder, which lights up while it's over it.
    await slideTo(window, shortlist.getByTestId("row-text"));
    await window.mouse.down();
    await slideTo(window, research.getByTestId("folder-row"));
    await expect(research).toHaveAttribute("data-drop-target", "true");
    await expect(finance).not.toHaveAttribute("data-drop-target", "true");
    await window.mouse.up();
    await expect(research).not.toHaveAttribute("data-drop-target", "true");
    await expect(research.getByTestId("mind-list-item")).toHaveText("Shortlist");
    await expect(undoLine.getByTestId("undo-message")).toHaveText("Moved “Shortlist” to Research");
    await screenshot(window, "folder-tree-moved");

    // Its composer now starts with Research; a Question asked keeps it.
    const editor = window.getByTestId("mind-editor");
    await editor.click();
    await window.keyboard.press("ControlOrMeta+j");
    const composer = window.getByTestId("composer");
    await expect(composer.getByTestId("scope-chip")).toHaveText("Research");
    await window.keyboard.type("What do my notes say?");
    await window.keyboard.press("Enter");
    const answer = editor.getByTestId("answer");
    await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });

    // A Folder renames in place, as in Finder: a double-click, Enter saves; Enter or F2 on its
    // row, Esc changes nothing.
    await research.getByTestId("folder-row").dblclick();
    // (While it is typed, the row has no name to find it by.)
    const field = window.getByTestId("folder-rename-field");
    await expect(field).toBeFocused();
    await field.fill("Research notes");
    await field.press("Enter");
    const renamed = sidebarFolder(window, "Research notes");
    await expect(renamed.getByTestId("folder-row").getByTestId("row-text")).toHaveText(
      "Research notes",
    );
    // Open as before the double-click.
    await expect(renamed).toHaveAttribute("data-expanded", "true");
    await expect(composer.getByTestId("scope-chip")).toHaveText("Research notes");
    for (const key of ["Enter", "F2"]) {
      await renamed.getByTestId("folder-row").focus();
      await window.keyboard.press(key);
      await expect(window.getByTestId("folder-rename-field")).toBeFocused();
      await window.keyboard.type("Not this");
      await window.keyboard.press("Escape");
      await expect(renamed.getByTestId("folder-row").getByTestId("row-text")).toHaveText(
        "Research notes",
      );
      await expect(renamed.getByTestId("folder-row")).toBeFocused();
    }

    // Its menu, in the shared order; Delete asks first, as it can't be undone.
    await renamed.getByTestId("folder-row").click({ button: "right" });
    const folderActions = renamed.getByTestId("folder-actions");
    await expect(folderActions.getByRole("menuitem")).toHaveText([
      /^Open in the Library$/,
      /^New Mind here$/,
      /^Rename(↩|F2)$/,
      /^Delete…$/,
    ]);
    await folderActions.getByTestId("folder-delete").click();
    const dialog = window.getByTestId("delete-folder-dialog");
    await expect(dialog).toContainText("Delete the Folder “Research notes”?");
    await expect(dialog).toContainText("move to Not in a Folder");
    await dialog.getByTestId("confirm-delete-folder").click();

    // Its Mind is Not in a Folder now: it searches everything; the Question asked keeps
    // the deleted Folder's name, struck through.
    await expect(renamed).toHaveCount(0);
    await expect(notInAFolder(window).getByTestId("mind-list-item")).toHaveText("Shortlist");
    await expect(composer.getByTestId("composer-scope-all")).toBeVisible();
    const questionChip = editor.getByTestId("question").getByTestId("scope-chip");
    await expect(questionChip).toHaveAttribute("data-deleted", "true");
    await expect(questionChip.locator("s")).toHaveText("Research notes");

    // In Chinese, the same tree.
    await window.evaluate(() =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: "zh-CN" },
      }),
    );
    await expect(notInAFolder(window)).toContainText("不在文件夹中");
    await expect(window.getByTestId("not-in-a-folder-hint")).toHaveText("搜索全部文档");
    await screenshot(window, "folder-tree-zh");
  } finally {
    await app.close();
  }
});
