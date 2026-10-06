import { expect, test } from "@playwright/test";
import { createDataFolder, launchApp, removeDataFolder } from "./app";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("a heading and a formula inserted from the slash menu are still there, rendered, after reopening the app", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;
  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();

  // "/" opens the slash menu, and Esc closes it.
  const menu = window.getByTestId("slash-menu");
  await window.keyboard.type("/");
  await expect(menu).toBeVisible();
  await window.keyboard.press("Escape");
  await expect(menu).toHaveCount(0);
  await window.keyboard.press("Backspace");

  // The arrows choose an entry and Enter inserts it: Text, then Heading 1.
  await window.keyboard.type("/");
  await expect(window.getByTestId("slash-item-text")).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("ArrowDown");
  await expect(window.getByTestId("slash-item-heading-1")).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Results");
  const heading = editor.locator("h1");
  await expect(heading).toHaveText("Results");
  const headingId = await heading.getAttribute("data-id");
  expect(headingId).toMatch(UUID);

  // Typing after the slash finds an entry: a math block, whose LaTeX field opens on insert.
  await window.keyboard.press("Enter");
  await window.keyboard.type("/math");
  await expect(window.getByTestId("slash-item-math")).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("Enter");
  const latexField = window.getByTestId("math-editor");
  await expect(latexField).toBeFocused();
  await window.keyboard.type("E = mc^2");
  await window.keyboard.press("Enter");
  await expect(latexField).toHaveCount(0);
  const formula = editor.locator('[data-type="block-math"]');
  await expect(formula).toHaveAttribute("data-latex", "E = mc^2");
  await expect(formula.locator(".katex")).toBeVisible();
  await first.app.close();

  // After a restart: the same heading Block, and the formula rendered by KaTeX.
  const second = await launchApp(dataDir);
  await second.window.getByTestId("mind-list-item").click();
  const reopened = second.window.getByTestId("mind-editor");
  await expect(reopened.locator("h1")).toHaveText("Results");
  await expect(reopened.locator("h1")).toHaveAttribute("data-id", `${headingId}`);
  const restored = reopened.locator('[data-type="block-math"]');
  await expect(restored).toHaveAttribute("data-latex", "E = mc^2");
  await expect(restored.locator(".katex")).toBeVisible();
  await second.app.close();
});

test("a Block dragged above another stays there after reopening the app", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;
  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  for (const [index, text] of ["First", "Second", "Third"].entries()) {
    if (index > 0) await window.keyboard.press("Enter");
    await window.keyboard.type(text);
  }
  const paragraphs = editor.locator("p");
  await expect(paragraphs).toHaveText(["First", "Second", "Third"]);

  // Hovering a Block shows its handle. Dropping it on the top of "First" moves the Block above it.
  await paragraphs.nth(2).hover();
  const handle = window.getByTestId("block-handle");
  await expect(handle).toBeVisible();
  await handle.dragTo(paragraphs.nth(0), { targetPosition: { x: 4, y: 2 } });
  await expect(paragraphs).toHaveText(["Third", "First", "Second"]);
  await first.app.close();

  const second = await launchApp(dataDir);
  await second.window.getByTestId("mind-list-item").click();
  await expect(second.window.getByTestId("mind-editor").locator("p")).toHaveText([
    "Third",
    "First",
    "Second",
  ]);
  await second.app.close();
});

test("a Block is deleted from its handle's menu, and with the keyboard", async () => {
  const { app, window } = await launchApp(dataDir);
  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  for (const [index, text] of [
    "Keep",
    "Delete from the menu",
    "Delete with the keyboard",
  ].entries()) {
    if (index > 0) await window.keyboard.press("Enter");
    await window.keyboard.type(text);
  }
  const paragraphs = editor.locator("p");

  // Clicking the handle opens the Block's menu.
  await paragraphs.nth(1).hover();
  await window.getByTestId("block-handle").click();
  await expect(window.getByTestId("block-menu")).toBeVisible();
  await window.getByTestId("block-menu-delete").click();
  await expect(window.getByTestId("block-menu")).toHaveCount(0);
  await expect(paragraphs).toHaveText(["Keep", "Delete with the keyboard"]);

  // Cmd/Ctrl+Shift+Backspace deletes the Block the cursor is in.
  await paragraphs.nth(1).click();
  await window.keyboard.press("ControlOrMeta+Shift+Backspace");
  await expect(paragraphs).toHaveText(["Keep"]);
  await app.close();
});
