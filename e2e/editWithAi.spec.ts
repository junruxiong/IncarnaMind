import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/*
 * Edit with AI (#93): select text, ask for a change from the selection menu
 * or with ⌘J, and review what comes back as changes in the note, each with
 * its own bar, by mouse and keyboard, with Undo. The scripted model (see
 * src/main/fakeChatModel.ts) shortens by dropping "very" and "really" (非常),
 * and does what "replace X with Y" (把 X 换成 Y) says. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);

const COMMAND = process.platform === "darwin" ? "Command" : "Control";

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

const setLanguage = (window: Page, language: "en" | "zh-CN") =>
  window.evaluate(
    (value) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: value },
      }),
    language,
  );

/** Selects a line's `words` with the mouse, then `more` characters after them with the keyboard. */
async function selectWords(window: Page, line: Locator, words: string, more = 0) {
  await slideSelect(window, line, words);
  for (let index = 0; index < more; index++) await window.keyboard.press("Shift+ArrowRight");
}

test("Shorten comes back as changes with their own bars: one accepted by mouse, one rejected by keyboard, Undo, and the Citation is still checked", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();

  // An Answer with a checked Citation, whose sentence is copied into a Note.
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("When do spring tides happen?");
  await window.keyboard.press("Enter");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await editor.evaluate((dom) => {
    const { view, commands } = (dom as unknown as { editor: Editor }).editor;
    let at = -1;
    view.state.doc.descendants((node, pos) => {
      if (at < 0 && node.type.name === "citation") at = pos;
    });
    const sentence = view.state.doc.resolve(at);
    commands.setTextSelection({ from: sentence.start(), to: sentence.end() });
    const { dom: copied } = view.serializeForClipboard(view.state.selection.content());
    commands.setTextSelection(view.state.doc.content.size - 1);
    view.pasteHTML(copied.innerHTML);
  });
  const note = editor.locator(":scope > p").last();
  await expect(note.getByTestId("citation")).toHaveAttribute("data-check", "found");
  await slideSelect(window, note, "Your Documents answer this");
  await window.keyboard.type("The tides here are very strong and really regular");
  await expect(note).toContainText("The tides here are very strong and really regular");

  // Selected, the menu offers the AI's actions; Shorten comes back as two changes.
  // The words, then the space, the Citation and the full stop after them.
  await selectWords(window, note, "The tides here are very strong and really regular", 3);
  const menu = window.getByTestId("selection-menu");
  await expect(menu.getByTestId("selection-action-shorten")).toBeVisible();
  await expect(menu.getByTestId("selection-action-edit")).toContainText("Edit with AI…");
  await screenshot(window, "edit-menu-en");
  await pointAndClick(window, menu.getByTestId("selection-action-shorten"));

  const bar = window.getByTestId("proposals-bar");
  await expect(bar.getByTestId("proposals-count")).toHaveText("2 changes from the assistant");
  const removed = note.locator("del.proposal-delete");
  await expect(removed).toHaveText(["very ", "really "]);
  await expect(removed.first()).toHaveCSS("text-decoration-line", "line-through");
  // Nothing is applied yet: the words are still there, struck.
  await expect(note).toContainText("very strong and really regular");
  // The cursor went into the first change: its bar, and what a screen reader hears.
  const changeBar = window.getByTestId("change-bar");
  await expect(changeBar).toBeVisible();
  await expect(changeBar).toContainText("Change 1 of 2");
  await expect(window.getByTestId("proposal-spoken")).toHaveText(
    `Suggested change: removes “very”. Accept with ${COMMAND} Enter, reject with ${COMMAND} Backspace.`,
  );
  await screenshot(window, "edit-changes-en");
  // Its bar sits over its line.
  const barBox = await changeBar.boundingBox();
  const wordBox = await removed.first().boundingBox();
  expect((barBox?.y ?? 0) + (barBox?.height ?? 0)).toBeLessThanOrEqual((wordBox?.y ?? 0) + 1);

  // Accept by mouse.
  await pointAndClick(window, changeBar.getByTestId("change-accept"));
  await expect(removed).toHaveText(["really "]);
  await expect(note).toContainText("The tides here are strong and really regular");
  await expect(bar.getByTestId("proposals-count")).toHaveText("1 change from the assistant");
  await expect(window.getByTestId("undo-message")).toHaveText("Accepted 1 change");

  // Reject by keyboard, with the cursor in the other change.
  await pointAndClick(window, removed.first());
  await expect(changeBar).toContainText("Change 1 of 1");
  // The change the cursor is in is washed.
  await expect(note.locator(".proposal-current")).toHaveText("really ");
  await window.keyboard.press("ControlOrMeta+Backspace");
  await expect(removed).toHaveCount(0);
  await expect(bar).toHaveCount(0);
  await expect(note).toContainText("The tides here are strong and really regular");

  // Undo brings the rejected change back; Accept all takes it.
  await window.keyboard.press("ControlOrMeta+z");
  await expect(removed).toHaveText(["really "]);
  await pointAndClick(window, bar.getByTestId("proposals-accept-all"));
  await expect(bar).toHaveCount(0);
  await expect(note).toContainText("The tides here are strong and regular");
  // The Citation in the sentence is kept, and its quote still found.
  await expect(note.getByTestId("citation")).toHaveCount(1);
  await expect(note.getByTestId("citation")).toHaveAttribute("data-check", "found");
});

test("⌘J with text selected says what to change in the composer, in Chinese; ⌘↵ accepts, Undo, and Reject all", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 900, 660);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type("潮水每天非常规律地涨落两次。");
  const note = editor.locator(":scope > p").first();

  // ⌘J with the line selected: the composer quotes it and asks what to change.
  await selectWords(window, note, "潮水每天非常规律地涨落两次。");
  await window.keyboard.press("ControlOrMeta+j");
  const composer = window.getByTestId("composer");
  const input = composer.getByTestId("composer-input");
  await expect(input).toBeFocused();
  await expect(composer.getByTestId("composer-edit-quote")).toContainText("正在编辑“潮水每天");
  await expect(input).toHaveAttribute("placeholder", "说说要怎么改…");
  await expect(note.locator(".proposal-target")).toHaveText("潮水每天非常规律地涨落两次。");
  await window.keyboard.type("把规律换成准时");
  await screenshot(window, "edit-composer-zh");
  await window.keyboard.press("Enter");

  const bar = window.getByTestId("proposals-bar");
  await expect(bar.getByTestId("proposals-count")).toHaveText("来自助手的 1 处修订");
  await expect(note.locator("del.proposal-delete")).toHaveText("规律");
  await expect(note.locator("ins.proposal-insert")).toHaveText("准时");
  await expect(note.locator("ins.proposal-insert")).toHaveCSS("text-decoration-line", "underline");
  const changeBar = window.getByTestId("change-bar");
  await expect(changeBar).toContainText("第 1 处修订，共 1 处");
  await expect(changeBar.getByTestId("change-accept")).toHaveText("接受");
  await expect(window.getByTestId("proposal-spoken")).toContainText(
    "建议的修订：将“规律”改为“准时”",
  );
  await screenshot(window, "edit-changes-zh");
  // The bar fits in the narrow window.
  const box = await changeBar.boundingBox();
  expect((box?.x ?? 0) + (box?.width ?? 0)).toBeLessThanOrEqual(900);

  // ⌘↵ in the change accepts it; Undo brings it back; Reject all leaves the text as it was.
  await editor.focus();
  await window.keyboard.press("ControlOrMeta+Enter");
  await expect(note).toHaveText("潮水每天非常准时地涨落两次。");
  await expect(bar).toHaveCount(0);
  await window.keyboard.press("ControlOrMeta+z");
  await expect(note.locator("ins.proposal-insert")).toHaveText("准时");
  await pointAndClick(window, bar.getByTestId("proposals-reject-all"));
  await expect(note).toHaveText("潮水每天非常规律地涨落两次。");
  await expect(window.getByTestId("undo-message")).toHaveText("已拒绝 1 处修订");

  // Shorten from the menu, in Chinese: 非常 goes.
  await selectWords(window, note, "潮水每天非常规律地涨落两次。");
  await pointAndClick(window, window.getByTestId("selection-action-shorten"));
  await expect(note.locator("del.proposal-delete")).toHaveText("非常");
  await window.keyboard.press("ControlOrMeta+Enter");
  await expect(note).toHaveText("潮水每天规律地涨落两次。");

  // Translate… asks for the language beside it. A text the model gives back as it was (the
  // scripted model only capitalises) says so in the composer, and changes nothing.
  await selectWords(window, note, "潮水每天规律地涨落两次。");
  await pointAndClick(window, window.getByTestId("selection-action-translate"));
  const languages = window.getByTestId("selection-choices-translate");
  await expect(languages).toBeVisible();
  await pointAndClick(window, languages.getByTestId("selection-choice-en"));
  const problem = window.getByTestId("edit-problem");
  await expect(problem).toContainText("没有要修改的地方");
  await expect(bar).toHaveCount(0);
  await pointAndClick(window, problem.getByTestId("edit-problem-close"));
  await expect(problem).toHaveCount(0);
});
