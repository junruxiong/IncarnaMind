import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  confirmLink,
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  linkedFolderRow,
  openViewer,
  previewLink,
  removeDataFolder,
  showSourceLocations,
  widthOf,
} from "./app";

/*
 * The shell's card (DESIGN.md, Layout): everything right of the 248px sidebar
 * sits in one card, 8px from the sidebar and from the window's top, right and
 * bottom edges, with its 36px tab band inside, and the sidebar header has one
 * "+" menu. Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots
 * of the shell at 1440 and 1000px, in English and in Chinese.
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

async function boxOf(locator: Locator) {
  const box = await locator.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  return box;
}

/** A new Mind the way a person makes one: the "+" in the band, then typing its title. */
async function newMind(window: Page, title: string): Promise<void> {
  await window.getByTestId("new-tab").click();
  await expect(window.getByTestId("mind-title")).toBeFocused();
  await window.keyboard.type(title);
}

test("the Mind sits in a card 8px from the sidebar and the window's edges, with its band inside", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await newMind(window, "Reading notes");
  await newMind(window, "Interview plan");

  for (const [width, requested] of [
    [1440, 860],
    [1000, 700],
  ] as const) {
    await app.evaluate(
      ({ BrowserWindow }, size) =>
        BrowserWindow.getAllWindows()[0]?.setSize(size.width, size.height),
      { width, height: requested },
    );
    await expect.poll(() => window.evaluate(() => globalThis.innerWidth)).toBe(width);
    // The screen may not have room for the height asked for: the card follows the page's.
    await expect.poll(() => window.evaluate(() => globalThis.innerHeight)).toBeGreaterThan(600);
    const height = await window.evaluate(() => globalThis.innerHeight);

    // The sidebar is exactly 248px; the card follows after an 8px gap and ends 8px short of the window.
    const sidebar = await boxOf(window.getByTestId("sidebar"));
    expect(sidebar.width).toBe(248);
    const card = await boxOf(window.getByTestId("card"));
    expect(card).toEqual({ x: 256, y: 8, width: width - 256 - 8, height: height - 16 });
    const style = await window
      .getByTestId("card")
      .evaluate((element) => getComputedStyle(element).borderTopLeftRadius);
    expect(style).toBe("10px");

    // The band is the card's first 36px, so it ends level with the sidebar header's rule at y 44,
    // and the tabs sit inside it, the first flush with the card's left edge.
    const header = await boxOf(window.getByTestId("sidebar-header"));
    const band = await boxOf(window.getByTestId("mind-header"));
    expect(band).toMatchObject({ x: card.x, y: 8, width: card.width, height: 36 });
    expect(header.y + header.height).toBe(band.y + band.height);
    const firstTab = await boxOf(window.getByTestId("mind-tab").first());
    expect(firstTab.x).toBe(card.x);

    if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `shell-${width}.png`) });

    // With the viewer open, it is inside the card too, and its toolbar continues the band.
    await openViewer(window);
    const viewer = await boxOf(window.getByTestId("viewer"));
    expect(viewer.x + viewer.width).toBe(width - 8);
    expect(await boxOf(window.getByTestId("viewer-header"))).toMatchObject({ y: 8, height: 36 });
    if (SCREENSHOTS) {
      await window.screenshot({ path: join(SCREENSHOTS, `shell-${width}-viewer.png`) });
    }
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("viewer")).toHaveCount(0);
  }
  await app.close();
});

test("the divider in the gap before the card is grabbed with the mouse and moved with the arrow keys", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const rod = window.getByTestId("sidebar-resize");
  const sidebar = window.getByTestId("sidebar");
  const card = window.getByTestId("card");

  // The gap itself is the grab strip: 8px wide, between the sidebar and the card.
  const gap = await boxOf(rod);
  expect(gap).toMatchObject({ x: 248, width: 8 });

  // Slid onto and dragged 40px right, it widens the sidebar and keeps the 8px gap.
  const y = gap.y + gap.height / 2;
  await window.mouse.move(gap.x + 40, y);
  await window.mouse.move(gap.x + 4, y, { steps: 6 });
  await window.mouse.down();
  await window.mouse.move(gap.x + 44, y, { steps: 6 });
  await window.mouse.up();
  await expect.poll(() => widthOf(sidebar)).toBe(288);
  expect((await boxOf(card)).x).toBe(296);

  // Focused, the arrow keys move it, as before.
  await rod.focus();
  await window.keyboard.press("ArrowLeft");
  await expect.poll(() => widthOf(sidebar)).toBe(278);
  await app.close();
});

test("the + menu in the sidebar header offers New Mind, Add Documents… and Link a folder…, by mouse and by keyboard", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const plus = window.getByTestId("plus-menu");
  const menu = window.getByRole("menu", { name: "New or add" });

  // It sits at the header's right end, 28px, with a name for a screen reader.
  const header = await boxOf(window.getByTestId("sidebar-header"));
  const button = await boxOf(plus);
  expect(button).toMatchObject({ width: 28, height: 28 });
  expect(button.x + button.width).toBe(header.x + header.width - 8);
  await expect(plus).toHaveAccessibleName("New or add");
  await expect(plus).toHaveAttribute("aria-haspopup", "menu");

  // Opened by the mouse, its four actions are menu items, and the first has the focus.
  await window.mouse.move(button.x - 40, button.y + 14);
  await window.mouse.move(button.x + 14, button.y + 14, { steps: 6 });
  await window.mouse.down();
  await window.mouse.up();
  await expect(menu).toBeVisible();
  const items = menu.getByRole("menuitem");
  await expect(items).toHaveText([/New Mind/, /New Folder…/, /Add Documents…/, /Link a folder…/]);
  await expect(items.first()).toBeFocused();
  await expect(plus).toHaveAttribute("aria-expanded", "true");
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "shell-menu-en.png") });

  // The arrow keys, Home and End move between them; Esc closes the menu and returns to "+".
  await window.keyboard.press("ArrowDown");
  await expect(items.nth(1)).toBeFocused();
  await window.keyboard.press("End");
  await expect(items.nth(3)).toBeFocused();
  await window.keyboard.press("ArrowDown");
  await expect(items.first()).toBeFocused();
  await window.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await expect(plus).toBeFocused();

  // From the keyboard alone: Enter opens it, Enter on New Mind makes a Mind named by typing.
  await window.keyboard.press("Enter");
  await expect(items.first()).toBeFocused();
  await window.keyboard.press("Enter");
  await expect(menu).toBeHidden();
  await expect(window.getByTestId("mind-title")).toBeFocused();
  await window.keyboard.type("From the menu");
  await expect(window.getByTestId("mind-tab-title")).toHaveText(["From the menu"]);

  // Add Documents… opens the system's picker and adds what is chosen.
  const notes = join(sources, "Field notes.txt");
  await writeFile(notes, "Tea is picked by hand in spring.\n");
  await interceptOpenDialog(app, notes);
  await plus.focus();
  await window.keyboard.press("Enter");
  await expect(items.first()).toBeFocused();
  await window.keyboard.press("ArrowDown");
  await window.keyboard.press("ArrowDown");
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("document-list-item")).toHaveCount(1);

  // Link a folder… asks first, as "Add folder…" does.
  const folder = join(sources, "Library");
  await mkdir(folder);
  await writeFile(join(folder, "Brew.txt"), "Water at eighty degrees.\n");
  await plus.focus();
  await window.keyboard.press("Enter");
  await expect(items.first()).toBeFocused();
  const dialog = await previewLink(app, window, folder, items.nth(3));
  await confirmLink(dialog);
  await showSourceLocations(window);
  await expect(linkedFolderRow(window, "Library")).toHaveCount(1);
  await app.close();
});

test("the + menu and the shell speak Chinese too", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.updateSettings({ user: { language: "zh-CN" } });
  });
  const plus = window.getByTestId("plus-menu");
  await expect(plus).toHaveAccessibleName("新建或添加");
  await plus.click();
  await expect(window.getByRole("menuitem")).toHaveText([
    /新建 Mind/,
    /新建文件夹…/,
    /添加文档…/,
    /关联文件夹…/,
  ]);
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "shell-zh-menu.png") });
  await window.keyboard.press("Escape");
  await window.getByTestId("plus-menu").focus();
  await window.keyboard.press("Enter");
  await expect(window.getByRole("menuitem").first()).toBeFocused();
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("mind-title")).toBeFocused();
  await window.keyboard.type("茶叶的产地");
  await window.getByTestId("new-tab").click();
  await expect(window.getByTestId("mind-title")).toBeFocused();
  await window.keyboard.type("访谈提纲");
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "shell-zh.png") });
  await app.close();
});
