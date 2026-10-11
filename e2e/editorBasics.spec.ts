import { expect, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  interceptOpenExternal,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  urlsOpened,
} from "./app";

/*
 * The editor's basics (#115): lists, a checklist, a quote and a divider from
 * "/", links, the outline, find in the Mind, "@" for a Document, each by
 * keyboard and by mouse. Set INCARNAMIND_SCREENSHOTS to a folder to also
 * save screenshots.
 */
let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

/** Opens a new Mind with the cursor on its first line, in a window of a fixed size. */
async function startMind(language: "en" | "zh-CN" = "en", width = 1280) {
  const running = await launchApp(dataDir);
  const { app, window } = running;
  await dismissChatSetup(window);
  await setLanguage(window, language);
  await setWindowSize(app, window, width, 700);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await pointAndClick(window, editor);
  return { ...running, editor };
}

const BLOCKS = [
  { id: "bulleted-list", selector: "ul > li > p", typed: "bullet" },
  { id: "numbered-list", selector: "ol > li > p", typed: "numbered" },
  { id: "checklist", selector: "ul:has(> li[data-checked]) > li > div > p", typed: "check" },
  { id: "quote", selector: "blockquote > p", typed: "quote" },
] as const;

test('"/" offers lists, a checklist, a quote and a divider, chosen by keyboard or mouse and undone', async () => {
  const { window, editor } = await startMind();

  for (const [index, block] of BLOCKS.entries()) {
    await window.keyboard.type("/");
    for (const other of BLOCKS) {
      await expect(window.getByTestId(`slash-item-${other.id}`)).toBeVisible();
    }
    const item = window.getByTestId(`slash-item-${block.id}`);
    if (index % 2 === 0) {
      // By keyboard: type the name, Enter.
      await window.keyboard.type(block.typed);
      await expect(item).toHaveAttribute("aria-selected", "true");
      await window.keyboard.press("Enter");
    } else {
      // By mouse: point at the item and click it.
      await item.hover();
      await item.click();
    }
    await expect(editor.locator(block.selector)).toHaveCount(1);
    await window.keyboard.type(`Item ${index}`);
    await expect(editor.locator(block.selector)).toHaveText(`Item ${index}`);
    // Enter twice leaves the block, to a plain line for the next one.
    await window.keyboard.press("Enter");
    await window.keyboard.press("Enter");
  }

  // A checklist item is ticked with its box.
  await window.keyboard.type("/check");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Buy milk");
  const box = editor.locator("li[data-checked] input").last();
  await box.hover();
  await box.click();
  await expect(editor.locator('li[data-checked="true"]')).toHaveCount(1);
  await window.keyboard.press("Enter");
  await window.keyboard.press("Enter");

  // A divider, and the cursor goes on to the line after it.
  await window.keyboard.type("/divider");
  await expect(window.getByTestId("slash-item-divider")).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("Enter");
  await expect(editor.locator("hr")).toHaveCount(1);
  await window.keyboard.type("After the rule");
  await expect(editor.locator("hr + p")).toHaveText("After the rule");

  // Undo takes the typing back, then the divider.
  await window.keyboard.press("ControlOrMeta+Z");
  await window.keyboard.press("ControlOrMeta+Z");
  await expect(editor.locator("hr")).toHaveCount(0);
});

test('"/" finds the new items by their Chinese names', async () => {
  const { window, editor } = await startMind("zh-CN");
  const names = [
    ["列表", ["bulleted-list", "numbered-list"]],
    ["待办", ["checklist"]],
    ["引用", ["quote"]],
    ["分割", ["divider"]],
  ] as const;
  for (const [typed, ids] of names) {
    await window.keyboard.type(`/${typed}`);
    for (const id of ids) await expect(window.getByTestId(`slash-item-${id}`)).toBeVisible();
    await window.keyboard.press("Escape");
    for (let count = 0; count <= typed.length; count++) await window.keyboard.press("Backspace");
  }
  await window.keyboard.type("/待办");
  await window.keyboard.press("Enter");
  await expect(editor.locator("ul:has(> li[data-checked])")).toHaveCount(1);
});

test("a link: added with the button or ⌘K on a selection, edited, opened and removed in place, each undone", async () => {
  const { app, window, editor } = await startMind();
  await interceptOpenExternal(app);
  await window.keyboard.type("Read the Tides paper today and the Moon one tomorrow");
  const line = editor.locator("p").first();
  const menu = window.getByTestId("selection-menu");
  const field = window.getByTestId("link-field");

  // ⌘K on a selection opens the address field; Enter makes the link.
  await slideSelect(window, line, "Tides paper");
  await expect(menu).toBeVisible();
  await window.keyboard.press("ControlOrMeta+K");
  await expect(field).toBeFocused();
  await window.keyboard.type("example.com/tides");
  await window.keyboard.press("Enter");
  await expect(line.locator("a")).toHaveText("Tides paper");
  await expect(line.locator("a")).toHaveAttribute("href", "https://example.com/tides");
  // Undo takes the link back as one step.
  await window.keyboard.press("ControlOrMeta+Z");
  await expect(line.locator("a")).toHaveCount(0);
  await expect(line).toHaveText("Read the Tides paper today and the Moon one tomorrow");

  // The selection menu's link button does the same, by mouse.
  await slideSelect(window, line, "Moon one");
  await expect(menu).toBeVisible();
  await window.getByTestId("format-link").hover();
  await window.getByTestId("format-link").click();
  await expect(field).toBeFocused();
  await window.keyboard.type("https://example.com/moon");
  await window.keyboard.press("Enter");
  const link = line.locator("a");
  await expect(link).toHaveAttribute("href", "https://example.com/moon");

  // A click in the link shows where it goes, without leaving the page.
  await link.hover();
  await link.click();
  const bubble = window.getByTestId("link-bubble-address");
  await expect(bubble).toHaveText("https://example.com/moon");
  await window.getByTestId("link-open").hover();
  await window.getByTestId("link-open").click();
  await expect.poll(() => urlsOpened(app)).toEqual(["https://example.com/moon"]);

  // Edit in place: the field has the address; a new one replaces it on the whole link.
  await window.getByTestId("link-edit").hover();
  await window.getByTestId("link-edit").click();
  await expect(field).toHaveValue("https://example.com/moon");
  await field.fill("example.com/moons");
  await field.press("Enter");
  await expect(link).toHaveAttribute("href", "https://example.com/moons");
  await expect(link).toHaveText("Moon one");

  // Remove it from the bubble; Undo puts it back.
  await link.click();
  await window.getByTestId("link-remove").hover();
  await window.getByTestId("link-remove").click();
  await expect(line.locator("a")).toHaveCount(0);
  await window.keyboard.press("ControlOrMeta+Z");
  await expect(link).toHaveAttribute("href", "https://example.com/moons");

  // By keyboard: the cursor into the link, ⌘K edits it, Esc leaves it as it was.
  await window.keyboard.press("End");
  for (let count = 0; count < 12; count++) await window.keyboard.press("ArrowLeft");
  await expect(bubble).toBeVisible();
  await window.keyboard.press("ControlOrMeta+K");
  await expect(field).toBeFocused();
  await window.keyboard.press("Escape");
  await expect(field).toHaveCount(0);
  await expect(link).toHaveAttribute("href", "https://example.com/moons");
  await expect(editor).toBeFocused();

  // Cleared and confirmed, the field removes the link.
  await window.keyboard.press("ControlOrMeta+K");
  await field.fill("");
  await field.press("Enter");
  await expect(line.locator("a")).toHaveCount(0);

  // With nothing selected and the cursor outside a link, ⌘K does nothing here.
  await window.keyboard.press("ControlOrMeta+K");
  await expect(field).toHaveCount(0);
});
