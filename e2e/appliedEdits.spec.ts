import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/*
 * An edit the User asks for on the text applies at once (#880): ⌘J, the
 * selection menu's AI actions, and a change asked for in the composer with
 * text selected. A line over it says how many places changed, with Undo and
 * Review changes. The scripted model (src/main/fakeChat/edit.ts and
 * src/main/fakeChat/appliedEdits.ts) does what "replace X with Y" says, shortens
 * by dropping "very" and "really", gives back only the second paragraph
 * made formal, as Claude Haiku 5.5 did, deletes the second paragraph, and
 * writes a haiku like none of the text. Set INCARNAMIND_SCREENSHOTS to a
 * folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const TIDES = "The tides here are very strong and really regular.";
const FIRST = "Tides rise twice a day along this coast.";
const SECOND = "It is a big harbour and a lot of boats do not leave at low water.";
const THIRD = "The harbour office posts the times each week.";
const FORMAL = "It is a considerable harbour and a great many boats do not leave at low water.";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

async function launch(width = 1440) {
  const running = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(running.window);
  await setWindowSize(running.app, running.window, width, 900);
  return running;
}

/** A new Mind with these paragraphs; its editor and paragraphs. */
async function mindWith(window: Page, lines: readonly string[]) {
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  for (const [index, line] of lines.entries()) {
    if (index > 0) await window.keyboard.press("Enter");
    await window.keyboard.type(line);
  }
  const paragraphs = editor.locator(":scope > p");
  await expect(paragraphs).toHaveText([...lines]);
  return { editor, paragraphs };
}

/** ⌘J over what is selected, saying what to change. */
async function askToChange(window: Page, instruction: string) {
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type(instruction);
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("edit-pending")).toHaveCount(0);
}

async function selectAll(window: Page, editor: Locator) {
  await editor.click();
  await window.keyboard.press("ControlOrMeta+a");
}

const lineOf = (editor: Locator) => editor.getByTestId("applied-line");

test("⌘J applies the change at once as AI text, with Changed 1 place · Undo ⌘Z · Review changes over it; ⌘Z takes it back and ⌘⇧Z does it again, in Chinese too", async () => {
  const { window } = await launch();
  await useLocalChatModel(window);
  const { editor, paragraphs } = await mindWith(window, [TIDES]);
  const note = paragraphs.first();

  await slideSelect(window, note, TIDES);
  await askToChange(window, "replace strong with powerful");

  await expect(note).toHaveText("The tides here are very powerful and really regular.");
  await expect(note.locator("del.proposal-delete, ins.proposal-insert")).toHaveCount(0);
  await expect(note.locator(".generated-text")).toHaveText("powerful");
  await expect(window.getByTestId("proposals-bar")).toHaveCount(0);
  const line = lineOf(editor);
  await expect(line.getByTestId("applied-count")).toHaveText("Changed 1 place");
  await expect(line).toContainText(/Undo(⌘Z|Ctrl\+Z)/);
  await expect(line.getByTestId("applied-review")).toHaveText("Review changes");
  await expect(line).toHaveAttribute("title", "Why: Changed as you asked.");
  await expect(window.getByTestId("undo-message")).toHaveText("Changed 1 place");
  // Over the Block it changed.
  const lineBox = await line.boundingBox();
  const noteBox = await note.boundingBox();
  expect((lineBox?.y ?? 0) + (lineBox?.height ?? 0)).toBeLessThanOrEqual((noteBox?.y ?? 0) + 1);
  await motionSettled(window);
  await screenshot(window, "applied-en");

  await setLanguage(window, "zh-CN");
  await expect(line.getByTestId("applied-count")).toHaveText("改了 1 处");
  await expect(line.getByTestId("applied-undo")).toHaveText("撤销");
  await expect(line.getByTestId("applied-review")).toHaveText("查看改动");
  await screenshot(window, "applied-zh");
  await setLanguage(window, "en");

  await editor.focus();
  await window.keyboard.press("ControlOrMeta+z");
  await expect(note).toHaveText(TIDES);
  await expect(line).toHaveCount(0);
  await window.keyboard.press("ControlOrMeta+Shift+z");
  await expect(note).toHaveText("The tides here are very powerful and really regular.");
  await expect(note.locator("del.proposal-delete, ins.proposal-insert")).toHaveCount(0);
});

test("Shorten from the menu applies at once; Review changes shows what it did, struck and underlined, each with Restore original; Done keeps the rest, and Undo all takes all back", async () => {
  const { window } = await launch();
  await useLocalChatModel(window);
  const { editor, paragraphs } = await mindWith(window, [TIDES]);
  const note = paragraphs.first();
  const line = lineOf(editor);
  const removed = note.locator("del.proposal-delete");

  await slideSelect(window, note, TIDES);
  await pointAndClick(window, window.getByTestId("selection-action-shorten"));
  await expect(note).toHaveText("The tides here are strong and regular.");
  await expect(line.getByTestId("applied-count")).toHaveText("Changed 2 places");

  await pointAndClick(window, line.getByTestId("applied-review"));
  await expect(removed).toHaveText(["very ", "really "]);
  await expect(line.getByTestId("applied-count")).toHaveText("2 changes made by Edit with AI");
  await expect(line.getByTestId("applied-undo-all")).toHaveText("Undo all");
  await expect(line.getByTestId("applied-done")).toHaveText("Done");
  await expect(window.getByTestId("proposals-bar")).toHaveCount(0);
  await motionSettled(window);
  await screenshot(window, "applied-review-en");

  // Restoring one leaves the other.
  await pointAndClick(window, removed.first());
  const changeBar = window.getByTestId("change-bar");
  await expect(changeBar.getByTestId("change-restore")).toHaveText("Restore original");
  await expect(changeBar.getByTestId("change-accept")).toHaveCount(0);
  await pointAndClick(window, changeBar.getByTestId("change-restore"));
  await expect(removed).toHaveText(["really "]);
  await expect(line.getByTestId("applied-count")).toHaveText("1 change made by Edit with AI");
  await pointAndClick(window, line.getByTestId("applied-done"));
  await expect(note).toHaveText("The tides here are very strong and regular.");
  await expect(removed).toHaveCount(0);
  await expect(line).toHaveCount(0);

  // Undo all, in Chinese.
  await setLanguage(window, "zh-CN");
  await slideSelect(window, note, "The tides here are very strong and regular.");
  await pointAndClick(window, window.getByTestId("selection-action-shorten"));
  await expect(note).toHaveText("The tides here are strong and regular.");
  await pointAndClick(window, line.getByTestId("applied-review"));
  await expect(line.getByTestId("applied-count")).toHaveText("用 AI 编辑做的 1 处改动");
  await expect(line.getByTestId("applied-done")).toHaveText("完成");
  await pointAndClick(window, removed.first());
  await expect(window.getByTestId("change-restore")).toHaveText("恢复原文");
  await pointAndClick(window, line.getByTestId("applied-undo-all"));
  await expect(note).toHaveText("The tides here are very strong and regular.");
  await expect(line).toHaveCount(0);
});

test("whole note selected, “make the second paragraph more formal”, the model gives back only that paragraph: it changes, the two it left out are kept, and Remove them takes them out; ⌘Z brings them back, and the line's Undo takes back the whole edit, at 1,100 px", async () => {
  const { window } = await launch(1100);
  await useLocalChatModel(window);
  const { editor, paragraphs } = await mindWith(window, [FIRST, SECOND, THIRD]);
  const line = lineOf(editor);
  const kept = editor.getByTestId("applied-kept");

  await selectAll(window, editor);
  await askToChange(window, "make the second paragraph more formal; leave the others as they are");

  await expect(paragraphs).toHaveText([FIRST, FORMAL, THIRD]);
  await expect(line.getByTestId("applied-count")).toHaveText(/^Changed \d+ places$/);
  await expect(kept).toHaveText("Kept 2 paragraphs the edit left out·Remove them");
  // The line over the paragraph it changed, the kept one's over the first it kept.
  const keptBox = await kept.boundingBox();
  const firstBox = await paragraphs.first().boundingBox();
  expect((keptBox?.y ?? 0) + (keptBox?.height ?? 0)).toBeLessThanOrEqual((firstBox?.y ?? 0) + 1);
  const lineBox = await line.boundingBox();
  expect(lineBox?.y ?? 0).toBeGreaterThan((firstBox?.y ?? 0) + (firstBox?.height ?? 0) - 1);
  await motionSettled(window);
  await screenshot(window, "applied-kept-en-1100");
  await setLanguage(window, "zh-CN");
  await expect(kept).toHaveText("保留了这次修改漏掉的 2 个段落·删除它们");
  await screenshot(window, "applied-kept-zh-1100");
  await setLanguage(window, "en");

  await pointAndClick(window, kept.getByTestId("applied-remove-kept"));
  await expect(paragraphs).toHaveText([FORMAL]);
  await expect(kept).toHaveCount(0);
  await expect(window.getByTestId("undo-message")).toHaveText("Removed what the edit left out");
  await editor.focus();
  await window.keyboard.press("ControlOrMeta+z");
  await expect(paragraphs).toHaveText([FIRST, FORMAL, THIRD]);

  await pointAndClick(window, line.getByTestId("applied-undo"));
  await expect(paragraphs).toHaveText([FIRST, SECOND, THIRD]);
  await expect(line).toHaveCount(0);
});

test("an edit that would remove most of what was selected, and can't be mapped onto it, waits for the User: Discard leaves the text, Apply applies it", async () => {
  const { window } = await launch();
  await useLocalChatModel(window);
  const { editor, paragraphs } = await mindWith(window, [FIRST, SECOND, THIRD]);
  const line = lineOf(editor);

  await selectAll(window, editor);
  await askToChange(window, "replace it all with a haiku");
  await expect(line.getByTestId("applied-count")).toHaveText(
    "This edit would remove most of what you selected, so it's waiting for you",
  );
  await expect(editor.locator("del.proposal-delete").first()).toBeVisible();
  await expect(window.getByTestId("proposals-bar")).toHaveCount(0);
  await motionSettled(window);
  await screenshot(window, "applied-held-en");
  await setLanguage(window, "zh-CN");
  await expect(line.getByTestId("applied-count")).toHaveText(
    "这次修改会删掉你选中的大部分内容，所以先等你确认",
  );
  await expect(line.getByTestId("applied-apply")).toHaveText("应用");
  await expect(line.getByTestId("applied-discard")).toHaveText("放弃");
  await setLanguage(window, "en");

  await pointAndClick(window, line.getByTestId("applied-discard"));
  await expect(paragraphs).toHaveText([FIRST, SECOND, THIRD]);
  await expect(line).toHaveCount(0);

  await selectAll(window, editor);
  await askToChange(window, "replace it all with a haiku");
  await pointAndClick(window, line.getByTestId("applied-apply"));
  await expect(paragraphs).toHaveText([
    "Gulls wheel, crying",
    "salt wind on stone",
    "night falls soft",
  ]);
  await expect(line.getByTestId("applied-count")).toHaveText(/^Changed \d+ places$/);
  await editor.focus();
  await window.keyboard.press("ControlOrMeta+z");
  await expect(paragraphs).toHaveText([FIRST, SECOND, THIRD]);
});

test("“delete the second paragraph” applies at once; a change asked for in the composer with text selected applies at once too", async () => {
  const { window } = await launch();
  await useLocalChatModel(window);
  const { editor, paragraphs } = await mindWith(window, [FIRST, SECOND, THIRD]);
  const line = lineOf(editor);

  await selectAll(window, editor);
  await askToChange(window, "delete the second paragraph");
  await expect(paragraphs).toHaveText([FIRST, THIRD]);
  await expect(line.getByTestId("applied-count")).toHaveText("Changed 1 place");
  await expect(editor.getByTestId("applied-kept")).toHaveCount(0);

  await slideSelect(window, paragraphs.first(), FIRST);
  const input = window.getByTestId("composer-input");
  await pointAndClick(window, input);
  await window.keyboard.type("replace coast with shore");
  await window.keyboard.press("Enter");
  await expect(paragraphs.first()).toHaveText("Tides rise twice a day along this shore.");
  await expect(line.getByTestId("applied-count")).toHaveText("Changed 1 place");
  await expect(input).toHaveValue("");
});
