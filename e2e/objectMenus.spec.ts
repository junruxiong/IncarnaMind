import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  coreOf,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  linkedFolderRow,
  linkFolderFromSidebar,
  newMind,
  removeDataFolder,
  setWindowSize,
  showSourceLocations,
  sidebarFolder,
  useLocalChatModel,
} from "./app";

/*
 * Every object has one menu, in one order (#223): a right-click opens the
 * menu its ⋯ button opens, each item shows its shortcut, and the keyboard
 * drives it. Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots
 * of the menus there.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);

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

/** Moves the mouse onto an element in small steps, as a person does, and returns where it is. */
async function slideTo(window: Page, target: Locator): Promise<{ x: number; y: number }> {
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  const point = { x: box.x + Math.min(48, box.width / 2), y: box.y + box.height / 2 };
  await window.mouse.move(point.x, point.y, { steps: 8 });
  return point;
}

async function rightClick(window: Page, target: Locator): Promise<void> {
  const point = await slideTo(window, target);
  await window.mouse.click(point.x, point.y, { button: "right" });
}

/** The commands a menu holds, in order, with its rules as "—". */
async function commandsIn(menu: Locator): Promise<string[]> {
  await expect(menu).toBeVisible();
  return menu.evaluate((element) =>
    Array.from(element.children).map((child) =>
      child.getAttribute("role") === "menuitem" ? (child.getAttribute("data-testid") ?? "") : "—",
    ),
  );
}

/**
 * Right-clicks an object, reads its menu, closes it, then opens the ⋯ button of the
 * same object and reads that: they are one menu. Returns the commands.
 */
async function sameFromBothWays(
  window: Page,
  row: Locator,
  hit: Locator,
  button: Locator,
  menu: Locator,
): Promise<string[]> {
  await rightClick(window, hit);
  const byRightClick = await commandsIn(menu);
  await window.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await slideTo(window, hit);
  await row.hover();
  await button.click();
  const byButton = await commandsIn(menu);
  expect(byButton).toEqual(byRightClick);
  await window.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  return byRightClick;
}

test("a right-click opens the menu the ⋯ button opens, for a Mind, a Document, a Folder, a Tag, a Linked folder, a tab, an Answer and a Citation", async () => {
  const library = join(sources, "Library");
  await mkdir(library, { recursive: true });
  await writeFile(join(library, "Reading notes.md"), "# Notes\n\nAttention lets a model focus.\n");
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 1600, 800);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await (await coreOf(window)).evaluate(async (core) => {
    await core.createLibraryGroup({ name: "Finance", description: "" });
    const tag = await core.createTag({ name: "Tides" });
    const [first] = await core.listDocuments();
    if (first) await core.addDocumentTag(first.id, tag.id);
  });

  // A Mind with an Answer that cites a page.
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("When do spring tides happen?");
  await window.keyboard.press("Enter");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await newMind(window);

  // A Mind: Open, Open to the side; Export; Move to…; Rename; then Delete after a rule.
  const mind = window.getByTestId("mind-row").first();
  const mindMenu = await sameFromBothWays(
    window,
    mind,
    mind.getByTestId("mind-list-item"),
    mind.getByTestId("mind-menu"),
    mind.getByTestId("mind-actions"),
  );
  expect(mindMenu).toEqual([
    "mind.open",
    "mind.openSide",
    "—",
    "mind.export",
    "—",
    "mind.moveTo",
    "—",
    "mind.rename",
    "—",
    "mind.delete",
  ]);
  await rightClick(window, mind.getByTestId("mind-list-item"));
  await screenshot(window, "menu-mind");
  await window.keyboard.press("Escape");

  // A Document in the sidebar: Open first, its own actions, its file's, Delete last.
  const document = window.getByTestId("document-list-item").first();
  const documentMenu = await sameFromBothWays(
    window,
    document,
    document.getByTestId("open-document"),
    document.getByTestId("document-file-menu"),
    document.getByTestId("document-file-actions"),
  );
  expect(documentMenu).toEqual([
    "document.open",
    "document.openTab",
    "—",
    "document.ask",
    "document.startMind",
    "—",
    "document.rename",
    "—",
    "document.openExternally",
    "document.showInFolder",
    "—",
    "document.delete",
  ]);
  await rightClick(window, document.getByTestId("open-document"));
  await screenshot(window, "menu-document");
  await window.keyboard.press("Escape");

  // A Folder.
  const folder = sidebarFolder(window, "Finance");
  const folderMenu = await sameFromBothWays(
    window,
    folder,
    folder.getByTestId("folder-row"),
    folder.getByTestId("folder-menu"),
    folder.getByTestId("folder-actions"),
  );
  expect(folderMenu).toEqual([
    "folder.open",
    "—",
    "folder.newMind",
    "—",
    "folder.rename",
    "—",
    "folder.delete",
  ]);

  // A Tag, in the sidebar's Tags view.
  await window.getByTestId("browse-views").getByRole("button", { name: "Tags" }).click();
  const tag = window.getByTestId("sidebar-tag").first();
  const tagMenu = await sameFromBothWays(
    window,
    tag,
    tag.getByTestId("row-text"),
    tag.getByTestId("tag-menu"),
    tag.getByTestId("tag-actions"),
  );
  expect(tagMenu).toEqual(["tag.open", "—", "tag.manage"]);
  await window.getByTestId("browse-views").getByRole("button", { name: "Folders" }).click();

  // A Linked folder, in Source locations.
  await linkFolderFromSidebar(app, window, library);
  await showSourceLocations(window);
  const linked = linkedFolderRow(window, "Library");
  const linkedMenu = await sameFromBothWays(
    window,
    linked,
    linked.getByTestId("row-text"),
    linked.getByTestId("linked-folder-menu"),
    linked.getByTestId("linked-folder-actions"),
  );
  expect(linkedMenu).toEqual([
    "linkedFolder.pause",
    "linkedFolder.layout",
    "—",
    "linkedFolder.reveal",
    "—",
    "linkedFolder.unlink",
  ]);
  await window.getByTestId("browse-views").getByRole("button", { name: "Folders" }).click();

  // A tab: its own actions, then Close and Close other tabs last. The keyboard opens the same menu
  // as the right-click, its ⋯ being out of sight in the tab.
  const tab = window.getByRole("tab", { selected: true }).first();
  await rightClick(window, tab);
  const tabMenu = tab.getByTestId("tab-actions");
  const byRightClick = await commandsIn(tabMenu);
  expect(byRightClick.indexOf("tab.rename")).toBeLessThan(byRightClick.indexOf("tab.close"));
  expect(byRightClick.indexOf("tab.close")).toBeLessThan(byRightClick.indexOf("tab.closeOthers"));
  // Close and its kin end the menu, after a rule.
  expect(byRightClick[byRightClick.indexOf("tab.close") - 1]).toBe("—");
  await window.keyboard.press("Escape");
  await expect(tabMenu).toBeHidden();
  await tab.focus();
  await window.keyboard.press("Shift+F10");
  expect(await commandsIn(tabMenu)).toEqual(byRightClick);
  await window.keyboard.press("Escape");
  await expect(tab).toBeFocused();

  // An Answer, in its Mind: Copy, Regenerate; then Delete.
  await window.getByTestId("mind-row").nth(1).getByTestId("mind-list-item").click();
  const written = window.getByTestId("mind-editor").getByTestId("answer");
  await expect(written).toBeVisible();
  const answerMenu = written.getByTestId("answer-actions");
  await rightClick(window, written.locator(".answer-content p").first());
  const answerByRightClick = await commandsIn(answerMenu);
  expect(answerByRightClick).toEqual(["answer.copy", "answer.regenerate", "—", "block.delete"]);
  await window.keyboard.press("Escape");
  await slideTo(window, written);
  await written.getByTestId("answer-menu").click();
  expect(await commandsIn(answerMenu)).toEqual(answerByRightClick);
  await window.keyboard.press("Escape");

  // A Citation: Open, then Remove; its card's ⋯ opens the same.
  const citation = written.getByTestId("citation");
  const citationMenu = citation.getByTestId("citation-actions");
  await rightClick(window, citation.getByTestId("citation-chip"));
  expect(await commandsIn(citationMenu)).toEqual(["citation.open", "—", "citation.remove"]);
  await screenshot(window, "menu-citation");
  await window.keyboard.press("Escape");
  await citation.getByTestId("citation-chip").hover();
  const card = window.getByTestId("citation-card");
  await expect(card).toBeVisible();
  await card.getByTestId("citation-menu").click();
  expect(await commandsIn(card.getByTestId("citation-actions"))).toEqual([
    "citation.open",
    "—",
    "citation.remove",
  ]);
  await window.keyboard.press("Escape");

  await app.close();
});

test("the context-menu key and Shift+F10 open the focused row's menu; arrows move, Enter runs, Esc closes and returns the focus to the row", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await newMind(window);
  const mind = window.getByTestId("mind-row").first();
  const row = mind.getByTestId("mind-list-item");
  const menu = mind.getByTestId("mind-actions");

  for (const key of ["ContextMenu", "Shift+F10"]) {
    await row.focus();
    await window.keyboard.press(key);
    await expect(menu).toBeVisible();
    // The first item has the focus; the arrows move through the items and wrap round.
    await expect(menu.getByTestId("mind.open")).toBeFocused();
    await window.keyboard.press("ArrowDown");
    await expect(menu.getByTestId("mind.openSide")).toBeFocused();
    await window.keyboard.press("ArrowUp");
    await window.keyboard.press("ArrowUp");
    await expect(menu.getByTestId("mind.delete")).toBeFocused();
    // Esc closes it and the focus is back on the row, not lost.
    await window.keyboard.press("Escape");
    await expect(menu).toBeHidden();
    await expect(row).toBeFocused();
  }

  // Enter runs the focused item: Export… opens the export dialog for this Mind.
  await row.focus();
  await window.keyboard.press("Shift+F10");
  await window.keyboard.press("ArrowDown");
  await window.keyboard.press("ArrowDown");
  await expect(menu.getByTestId("mind.export")).toBeFocused();
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("export-dialog")).toBeVisible();
  await expect(menu).toBeHidden();
  await window.keyboard.press("Escape");
  await expect(window.getByTestId("export-dialog")).toBeHidden();
  await app.close();
});

test("each item shows its shortcut as the system writes it, and the labels are in Chinese too", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setWindowSize(app, window, 900, 700);
  await newMind(window);
  const mind = window.getByTestId("mind-row").first();
  const menu = mind.getByTestId("mind-actions");

  await rightClick(window, mind.getByTestId("mind-list-item"));
  await expect(menu.getByTestId("mind.moveTo")).toHaveText(/^Move to…(⇧⌘M|Ctrl\+Shift\+M)$/);
  await expect(menu.getByTestId("mind.rename")).toHaveText(/^Rename(↩|F2)$/);
  await screenshot(window, "menu-mind-narrow");
  await window.keyboard.press("Escape");

  await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "zh-CN" },
    }),
  );
  await rightClick(window, mind.getByTestId("mind-list-item"));
  await expect(menu.getByRole("menuitem")).toHaveText([
    "打开",
    "在侧边打开",
    "导出…",
    /^移动到…(⇧⌘M|Ctrl\+Shift\+M)$/,
    /^重命名(↩|F2)$/,
    "删除…",
  ]);
  await screenshot(window, "menu-mind-zh-narrow");
  // The menu stays inside the window and its items on one line each, at this width.
  const box = await menu.boundingBox();
  if (!box) throw new Error("The menu isn't laid out.");
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(900);
  for (const item of await menu.getByRole("menuitem").all()) {
    expect((await item.boundingBox())?.height).toBeLessThanOrEqual(32);
  }
  await app.close();
});
