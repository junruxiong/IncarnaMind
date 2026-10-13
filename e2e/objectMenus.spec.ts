import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  addToNote,
  askInAssistant,
  coreOf,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  linkedFolderRow,
  linkFolderFromSidebar,
  menuCommands,
  mindWrittenBefore,
  motionSettled,
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

async function rightClick(window: Page, target: Locator): Promise<{ x: number; y: number }> {
  const point = await slideTo(window, target);
  await window.mouse.click(point.x, point.y, { button: "right" });
  return point;
}

/**
 * Whether the item shows the focus ring DESIGN.md gives every control: a solid accent outline, 2px,
 * drawn inside the item so the menu's scrolling can't clip it.
 */
const ringShows = (item: Locator) =>
  item.evaluate((element) => {
    const probe = document.createElement("span");
    probe.style.color = "var(--color-accent)";
    document.body.append(probe);
    const accent = getComputedStyle(probe).color;
    probe.remove();
    const style = getComputedStyle(element);
    return (
      style.outlineStyle === "solid" &&
      Number.parseFloat(style.outlineWidth) >= 2 &&
      style.outlineColor === accent &&
      Number.parseFloat(style.outlineOffset) < 0
    );
  });

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
  const byRightClick = await menuCommands(menu);
  await window.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await slideTo(window, hit);
  await row.hover();
  await button.click();
  const byButton = await menuCommands(menu);
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

  // A Mind written before Questions went to the assistant (#201), with a Question and its Answer;
  // and a Mind holding an Answer added from the assistant, with its Citation of a page.
  await mindWrittenBefore(window, "Tide questions", [
    { question: "When do spring tides happen?", answer: "At new moon and at full moon." },
  ]);
  await newMind(window);
  await window.getByTestId("mind-title").fill("Cited notes");
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await addToNote(window, await askInAssistant(window, "When do spring tides happen?"));
  await expect(editor.getByTestId("citation")).toHaveCount(1);
  await newMind(window);

  // A Mind: Open, Open to the side; Export as… (the last format first), Export options…; Move to…,
  // New Folder with this Mind; Rename; then Delete after a rule.
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
    "mind.outline",
    "mind.export",
    "mind.exportAs.pdf",
    "mind.exportAs.html",
    "mind.exportAs.markdown",
    "mind.exportOptions",
    "mind.history",
    "mind.citationStyle",
    "skill.fromMind",
    "—",
    "mind.moveTo",
    "folder.newWith",
    "—",
    "mind.rename",
    "—",
    "mind.delete",
  ]);
  await rightClick(window, mind.getByTestId("mind-list-item"));
  await menuCommands(mind.getByTestId("mind-actions"));
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
    "folder.newWith",
    "—",
    "document.tags",
    "—",
    "document.rename",
    "—",
    "document.openExternally",
    "document.showInFolder",
    "file.copyPath",
    "file.info",
    "—",
    "document.delete",
  ]);
  await rightClick(window, document.getByTestId("open-document"));
  await menuCommands(document.getByTestId("document-file-actions"));
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
    "folder.new",
    "folder.organize",
    "folder.editDescription",
    "folder.sort",
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
  expect(tagMenu).toEqual(["tag.open", "—", "tag.manage", "tag.rename", "—", "tag.delete"]);
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
    "folder.sort",
    "—",
    "linkedFolder.reveal",
    "file.copyPath",
    "file.info",
    "linkedFolder.locate",
    "—",
    "linkedFolder.unlink",
  ]);
  await window.getByTestId("browse-views").getByRole("button", { name: "Folders" }).click();

  // A tab: its own actions, then Close and Close other tabs last. The keyboard opens the same menu
  // as the right-click, its ⋯ being out of sight in the tab.
  const tab = window.getByRole("tab", { selected: true }).first();
  await rightClick(window, tab);
  const tabMenu = tab.getByTestId("tab-actions");
  const byRightClick = await menuCommands(tabMenu);
  // Its Mind's items first, then the tab's own.
  expect(byRightClick.slice(0, 10)).toEqual([
    "mind.outline",
    "mind.export",
    "mind.exportAs.pdf",
    "mind.exportAs.html",
    "mind.exportAs.markdown",
    "mind.exportOptions",
    "mind.history",
    "—",
    "mind.delete",
    "—",
  ]);
  expect(byRightClick.indexOf("tab.rename")).toBeLessThan(byRightClick.indexOf("tab.close"));
  expect(byRightClick.indexOf("tab.close")).toBeLessThan(byRightClick.indexOf("tab.closeOthers"));
  // Close and its kin end the menu, after a rule.
  expect(byRightClick[byRightClick.indexOf("tab.close") - 1]).toBe("—");
  await window.keyboard.press("Escape");
  await expect(tabMenu).toBeHidden();
  await tab.focus();
  await window.keyboard.press("Shift+F10");
  expect(await menuCommands(tabMenu)).toEqual(byRightClick);
  await window.keyboard.press("Escape");
  await expect(tab).toBeFocused();

  // An Answer, in its Mind: Copy, Regenerate; then Delete.
  await window.getByTestId("mind-list-item").filter({ hasText: "Tide questions" }).click();
  const written = window.getByTestId("mind-editor").getByTestId("answer");
  await expect(written).toBeVisible();
  const answerMenu = written.getByTestId("answer-actions");
  await rightClick(window, written.locator(".answer-content p").first());
  const answerByRightClick = await menuCommands(answerMenu);
  expect(answerByRightClick).toEqual(["answer.copy", "answer.regenerate", "—", "block.delete"]);
  await window.keyboard.press("Escape");
  await slideTo(window, written);
  await written.getByTestId("answer-menu").click();
  expect(await menuCommands(answerMenu)).toEqual(answerByRightClick);
  await window.keyboard.press("Escape");

  // A Citation: Open, then Remove; its card's ⋯ opens the same.
  await window.getByTestId("mind-list-item").filter({ hasText: "Cited notes" }).click();
  const citation = window.getByTestId("mind-editor").getByTestId("citation");
  const citationMenu = citation.getByTestId("citation-actions");
  await rightClick(window, citation.getByTestId("citation-chip"));
  expect(await menuCommands(citationMenu)).toEqual(["citation.open", "—", "citation.remove"]);
  await screenshot(window, "menu-citation");
  await window.keyboard.press("Escape");
  await citation.getByTestId("citation-chip").hover();
  const card = window.getByTestId("citation-card");
  await expect(card).toBeVisible();
  await card.getByTestId("citation-menu").click();
  expect(await menuCommands(card.getByTestId("citation-actions"))).toEqual([
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
    // The first item has the focus; the arrows move through the items and wrap round. The focused
    // one shows the blue ring, not only a faint fill (#157).
    await expect(menu.getByTestId("mind.open")).toBeFocused();
    await expect.poll(() => ringShows(menu.getByTestId("mind.open"))).toBe(true);
    await window.keyboard.press("ArrowDown");
    await expect(menu.getByTestId("mind.openSide")).toBeFocused();
    await expect.poll(() => ringShows(menu.getByTestId("mind.openSide"))).toBe(true);
    await expect.poll(() => ringShows(menu.getByTestId("mind.open"))).toBe(false);
    await window.keyboard.press("ArrowUp");
    await window.keyboard.press("ArrowUp");
    await expect(menu.getByTestId("mind.delete")).toBeFocused();
    // Esc closes it and the focus is back on the row, not lost.
    await window.keyboard.press("Escape");
    await expect(menu).toBeHidden();
    await expect(row).toBeFocused();
  }

  // Enter runs the focused item: Export options… opens the options sheet for this Mind.
  await row.focus();
  await window.keyboard.press("Shift+F10");
  for (let step = 0; step < 7; step++) await window.keyboard.press("ArrowDown");
  await expect(menu.getByTestId("mind.exportOptions")).toBeFocused();
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
  await expect(menu.getByTestId("mind.rename")).toHaveText(/^RenameF2$/);
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
    /^大纲(⌥⌘L|Ctrl\+Alt\+L)$/,
    /^导出为 Word（\.docx）(⇧⌘E|Ctrl\+Shift\+E)$/,
    "导出为 PDF",
    "导出为 HTML",
    "导出为 Markdown（.md）",
    "导出选项…",
    "历史版本",
    "引用格式…",
    "做成技能",
    /^移动到…(⇧⌘M|Ctrl\+Shift\+M)$/,
    /^新建文件夹并放入此 Mind(⌃⌘N)?$/,
    /^重命名F2$/,
    "删除",
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

test("a right-click opens the menu at the pointer, the ⋯ at its button, and the keyboard at the row", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await newMind(window);
  const mind = window.getByTestId("mind-row").first();
  const menu = mind.getByTestId("mind-actions");
  const nearBy = async (point: { x: number; y: number }) => {
    await motionSettled(window);
    const box = await menu.boundingBox();
    if (!box) throw new Error("The menu isn't laid out.");
    expect(Math.abs(point.x - box.x)).toBeLessThan(4);
    expect(Math.abs(point.y - box.y)).toBeLessThan(4);
  };

  const point = await rightClick(window, mind.getByTestId("mind-list-item"));
  await expect(menu).toBeVisible();
  await nearBy(point);
  await window.keyboard.press("Escape");
  await expect(menu).toBeHidden();

  // The ⋯ opens it under the button, not at the last pointer.
  await mind.hover();
  const button = mind.getByTestId("mind-menu");
  await button.click();
  await motionSettled(window);
  const buttonBox = await button.boundingBox();
  const menuBox = await menu.boundingBox();
  if (!buttonBox || !menuBox) throw new Error("Not laid out.");
  expect(menuBox.y).toBeGreaterThanOrEqual(buttonBox.y + buttonBox.height - 2);
  await window.keyboard.press("Escape");

  // The keyboard opens it at the focused row.
  await mind.getByTestId("mind-list-item").focus();
  await window.keyboard.press("Shift+F10");
  await motionSettled(window);
  const rowBox = await mind.boundingBox();
  const keyedBox = await menu.boundingBox();
  if (!rowBox || !keyedBox) throw new Error("Not laid out.");
  expect(keyedBox.y).toBeGreaterThanOrEqual(rowBox.y + rowBox.height - 2);
  expect(keyedBox.y).toBeLessThan(rowBox.y + rowBox.height + 16);
  await app.close();
});

test("a Document's Tags…, a Folder's Edit description and a Tag's Rename and Delete start the flows the app already has", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await (await coreOf(window)).evaluate(async (core) => {
    await core.createLibraryGroup({ name: "Finance", description: "Invoices" });
    const tag = await core.createTag({ name: "Tides" });
    const [first] = await core.listDocuments();
    if (first) await core.addDocumentTag(first.id, tag.id);
  });

  const document = window.getByTestId("document-list-item").first();
  await rightClick(window, document.getByTestId("open-document"));
  await document.getByTestId("document.tags").click();
  await expect(document.getByTestId("document-tags-popover")).toBeVisible();
  await window.keyboard.press("Escape");

  const folder = sidebarFolder(window, "Finance");
  await rightClick(window, folder.getByTestId("folder-row"));
  await folder.getByTestId("folder.editDescription").click();
  await expect(window.getByTestId("library")).toBeVisible();
  await expect(window.getByTestId("library").locator("form input").first()).toHaveValue("Finance");
  await expect(window.getByTestId("library").locator("form textarea")).toHaveValue("Invoices");

  await window.getByTestId("browse-views").getByRole("button", { name: "Tags" }).click();
  const tag = window.getByTestId("sidebar-tag").filter({ hasText: "Tides" });
  await rightClick(window, tag.getByTestId("row-text"));
  await tag.getByTestId("tag.rename").click();
  // In its row, with no dialog.
  const dialog = window.getByTestId("tags-dialog");
  await expect(window.getByTestId("sidebar-tag").getByRole("textbox")).toHaveValue("Tides");
  await expect(dialog).toBeHidden();
  await window.keyboard.press("Escape");
  await expect(window.getByTestId("sidebar-tag").getByRole("textbox")).toHaveCount(0);
  await rightClick(window, tag.getByTestId("row-text"));
  await tag.getByTestId("tag.delete").click();
  await expect(window.getByTestId("undo-message")).toHaveText("Moved “Tides” to Recently deleted");
  await app.close();
});

test("a menu opens at once while another is closing, and a right-click on another row replaces the open menu", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await newMind(window);
  await newMind(window);
  const [first, second] = [
    window.getByTestId("mind-row").nth(0),
    window.getByTestId("mind-row").nth(1),
  ];
  const firstMenu = first.getByTestId("mind-actions");
  const secondMenu = second.getByTestId("mind-actions");

  // Esc, then at once a right-click on another row: the new menu opens, the old one is gone.
  await rightClick(window, first.getByTestId("mind-list-item"));
  await expect(firstMenu).toBeVisible();
  await window.keyboard.press("Escape");
  await rightClick(window, second.getByTestId("mind-list-item"));
  await expect(secondMenu).toBeVisible();
  await expect(firstMenu).toBeHidden();

  // With no Esc, a right-click on the row above dismisses it and opens that row's menu at once.
  await rightClick(window, first.getByTestId("mind-list-item"));
  await expect(firstMenu).toBeVisible();
  await expect(secondMenu).toBeHidden();

  // Esc, then at once the ⋯ of the other row: one click opens it, though the first is still leaving.
  await window.keyboard.press("Escape");
  await second.hover();
  await second.getByTestId("mind-menu").click();
  await expect(secondMenu).toBeVisible();
  await expect(firstMenu).toBeHidden();

  // Esc, then at once Shift+F10 on the other row.
  await window.keyboard.press("Escape");
  await first.getByTestId("mind-list-item").focus();
  await window.keyboard.press("Shift+F10");
  await expect(firstMenu).toBeVisible();
  await expect(secondMenu).toBeHidden();
  await app.close();
});
