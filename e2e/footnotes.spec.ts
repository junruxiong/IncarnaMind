import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import {
  createDataFolder,
  dismissChatSetup,
  dragBlock,
  launchApp,
  newMind,
  removeDataFolder,
  setWindowSize,
  slideSelect,
} from "./app";

/*
 * Footnotes the User writes (#284): inserted by keyboard, from the "/" menu,
 * from the selection menu and by typing "^[…]"; written and edited in place,
 * in the popover at the marker and in the footnote area; numbered in the
 * order they appear, as Blocks move; deleted, and back with Undo.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

/** Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots of footnotes. */
async function screenshot(target: Page | Locator, name: string): Promise<void> {
  const folder = process.env.INCARNAMIND_SCREENSHOTS;
  if (!folder) return;
  await mkdir(folder, { recursive: true });
  await target.screenshot({ path: join(folder, `${name}.png`) });
}

/**
 * Puts the cursor at the end of a line of text as a person does: slides to
 * the right of its words and clicks there (Home and End don't move the
 * cursor on macOS).
 */
async function clickEndOf(window: Page, line: Locator): Promise<void> {
  const box = await line.boundingBox();
  if (!box) throw new Error("The line isn't visible.");
  const x = box.x + box.width - 4;
  const y = box.y + box.height - 10;
  await window.mouse.move(x, y, { steps: 6 });
  await window.mouse.click(x, y);
  // A key pressed next acts where the editor's cursor is: wait until it is in the line.
  await expect
    .poll(() =>
      line.evaluate((element) => {
        const root = element.closest(".ProseMirror") as (Element & { editor?: Editor }) | null;
        const view = root?.editor?.view;
        if (!view) return false;
        const { from, empty } = view.state.selection;
        const at = view.domAtPos(from).node;
        return empty && element.contains(at);
      }),
    )
    .toBe(true);
}

/** The footnotes' texts in the footnote area, in their order. */
const areaTexts = (window: Page) =>
  window.getByTestId("footnote-area").getByTestId("footnote-row").getByTestId("footnote-text");

test("footnotes by keyboard, the / menu, the selection menu and ^[…], edited in place, numbered as Blocks move, deleted and undone", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 800);
  await dismissChatSetup(window);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type("Neap tides are weaker");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Spring tides come at new moon");

  // ⌥⌘F (Ctrl+Alt+F) puts a footnote after the cursor and opens its text at the marker.
  await window.keyboard.press("ControlOrMeta+Alt+F");
  const popover = window.getByTestId("footnote-popover");
  await expect(popover).toBeVisible();
  const field = popover.getByTestId("footnote-text");
  await expect(field).toBeFocused();
  await window.keyboard.type("Twice a month.");
  const marker = editor.getByTestId("footnote-marker").first();
  const markerBox = await marker.boundingBox();
  const popoverBox = await popover.boundingBox();
  if (!markerBox || !popoverBox) throw new Error("The marker or its popover isn't visible.");
  expect(popoverBox.y).toBeGreaterThanOrEqual(markerBox.y + markerBox.height - 2);
  await screenshot(window, "footnote-popover-en");

  // Enter goes back to the text, right after the marker.
  await window.keyboard.press("Enter");
  await expect(popover).toHaveCount(0);
  await window.keyboard.type(" and at full moon.");
  const paragraphs = editor.locator(":scope > p");
  await expect(paragraphs).toHaveText([
    "Neap tides are weaker",
    "Spring tides come at new moon1 and at full moon.",
  ]);
  await expect(editor.getByTestId("footnote-marker")).toHaveText(["1"]);
  await expect(marker).toHaveAttribute("aria-label", "Footnote 1: Twice a month.");

  // The footnote area at the end of the Mind lists it.
  const area = window.getByTestId("footnote-area");
  await expect(area).toBeVisible();
  await expect(areaTexts(window)).toHaveText(["Twice a month."]);

  // The "/" menu, in the line above: its footnote is 1 now, and the first one 2.
  await clickEndOf(window, paragraphs.nth(0));
  await window.keyboard.type(" /footnote");
  await expect(window.getByTestId("slash-item-footnote")).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("Enter");
  await expect(popover).toBeVisible();
  await window.keyboard.type("At the quarter moons.");
  await window.keyboard.press("Escape");
  await expect(popover).toHaveCount(0);
  await expect(areaTexts(window)).toHaveText(["At the quarter moons.", "Twice a month."]);
  await expect(editor.getByTestId("footnote-marker")).toHaveText(["1", "2"]);
  await expect(paragraphs).toHaveText([
    "Neap tides are weaker 1",
    "Spring tides come at new moon2 and at full moon.",
  ]);

  // The selection menu, by mouse: a footnote after the selected words.
  await slideSelect(window, paragraphs.nth(0), "Neap tides");
  await expect(window.getByTestId("selection-menu")).toBeVisible();
  await window.getByTestId("format-footnote").hover();
  await window.getByTestId("format-footnote").click();
  await expect(popover).toBeVisible();
  await window.keyboard.type("Less pull.");
  await window.keyboard.press("Enter");
  await expect(areaTexts(window)).toHaveText([
    "Less pull.",
    "At the quarter moons.",
    "Twice a month.",
  ]);
  await expect(paragraphs).toHaveText([
    "Neap tides1 are weaker 2",
    "Spring tides come at new moon3 and at full moon.",
  ]);

  // Typing "^[…]" makes one, in a new line at the end.
  await clickEndOf(window, paragraphs.nth(1));
  await window.keyboard.press("Enter");
  await window.keyboard.type("Tide tables^[Published each year.]");
  await expect(paragraphs.nth(2)).toHaveText("Tide tables4");
  await expect(areaTexts(window)).toHaveText([
    "Less pull.",
    "At the quarter moons.",
    "Twice a month.",
    "Published each year.",
  ]);
  await screenshot(window, "footnote-area-en");

  // Edited in place in the area: the marker's text follows.
  await clickEndOf(window, areaTexts(window).nth(1));
  await expect(areaTexts(window).nth(1)).toBeFocused();
  await window.keyboard.type(" Roughly.");
  await expect(editor.getByTestId("footnote-marker").nth(1)).toHaveAttribute(
    "aria-label",
    "Footnote 2: At the quarter moons. Roughly.",
  );

  // Moving a Block renumbers: "Tide tables" dragged to the top takes 1.
  await dragBlock(window, paragraphs.nth(2), paragraphs.nth(0));
  await expect(paragraphs.nth(0)).toHaveText("Tide tables1");
  await expect(areaTexts(window)).toHaveText([
    "Published each year.",
    "Less pull.",
    "At the quarter moons. Roughly.",
    "Twice a month.",
  ]);
  // Undo puts it back, numbers and all.
  await window.keyboard.press("ControlOrMeta+Z");
  await expect(paragraphs.nth(2)).toHaveText("Tide tables4");
  await expect(areaTexts(window).first()).toHaveText("Less pull.");

  // By keyboard: from a footnote's number in the area to its marker; Backspace deletes it.
  await window.getByTestId("footnote-area").getByTestId("footnote-row-number").nth(2).click();
  await expect(editor).toBeFocused();
  await window.keyboard.press("Backspace");
  await expect(editor.getByTestId("footnote-marker")).toHaveText(["1", "2", "3"]);
  await expect(areaTexts(window)).toHaveText([
    "Less pull.",
    "At the quarter moons. Roughly.",
    "Published each year.",
  ]);
  await window.keyboard.press("ControlOrMeta+Z");
  await expect(areaTexts(window)).toHaveText([
    "Less pull.",
    "At the quarter moons. Roughly.",
    "Twice a month.",
    "Published each year.",
  ]);

  // The arrows select a marker, and Enter opens its text; "Remove footnote" deletes it, Undo too.
  await window.getByTestId("footnote-area").getByTestId("footnote-row-number").nth(2).click();
  await expect(editor).toBeFocused();
  await window.keyboard.press("ArrowLeft");
  await expect(editor.getByTestId("footnote").nth(2)).toHaveClass(/ProseMirror-selectednode/);
  await window.keyboard.press("Enter");
  await expect(popover).toBeVisible();
  await expect(field).toHaveText("Twice a month.");
  await popover.getByTestId("footnote-remove").hover();
  await popover.getByTestId("footnote-remove").click();
  await expect(popover).toHaveCount(0);
  await expect(editor.getByTestId("footnote-marker")).toHaveCount(3);
  await window.keyboard.press("ControlOrMeta+Z");
  await expect(editor.getByTestId("footnote-marker")).toHaveCount(4);

  // Clicking a marker opens it too; Undo while writing takes the typing back.
  await editor.getByTestId("footnote-marker").first().hover();
  await editor.getByTestId("footnote-marker").first().click();
  await expect(field).toHaveText("Less pull.");
  await expect(field).toBeFocused();
  await window.keyboard.type(" Much less.");
  await expect(areaTexts(window).first()).toHaveText("Less pull. Much less.");
  await window.keyboard.press("ControlOrMeta+Z");
  await expect(field).toHaveText("Less pull.");
  await expect(areaTexts(window).first()).toHaveText("Less pull.");
  await window.keyboard.press("Escape");
  await expect(popover).toHaveCount(0);
  await app.close();

  // After a restart the footnotes are there, in the same order, with their texts.
  const again = await launchApp(dataDir);
  await again.window.getByTestId("mind-list-item").click();
  await expect(again.window.getByTestId("mind-editor").getByTestId("footnote-marker")).toHaveText([
    "1",
    "2",
    "3",
    "4",
  ]);
  await expect(areaTexts(again.window)).toHaveText([
    "Less pull.",
    "At the quarter moons. Roughly.",
    "Twice a month.",
    "Published each year.",
  ]);
  await again.app.close();
});
