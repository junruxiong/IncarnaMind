import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  askInAssistant,
  assistantAnswers,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  enterContent,
  launchApp,
  pointAndClick,
  pointAt,
  removeDataFolder,
  setWindowSize,
  slideToHandle,
  stubClipboard,
} from "./app";

/**
 * Answers in the assistant (#201): a Question asked in the assistant is
 * answered there, never in the note; Add to note keeps an Answer (or a part
 * of it) at the note's cursor, its Citations checked again there, with Undo;
 * the ▾ adds it to another Mind or a new one, with a link back that opens the
 * conversation at that Answer; ⌘J, Tab, ⌘⇧A, ⌘⇧C and Esc reach it all.
 * Driven as a person would, with the mouse slid onto what it clicks. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(target: Page | Locator, name: string): Promise<void> {
  if (SCREENSHOTS) await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
  { lines: ["Tide tables", "Harbours publish the times of high water every year."] },
]);

let dataDir: string;
let sources: string;
let home: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
  home = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
  await removeDataFolder(home);
});

/** The note as the editor holds it, as JSON: what is stored for it. */
const noteJSON = (editor: Locator) =>
  editor.evaluate((dom) => JSON.stringify((dom as unknown as { editor: Editor }).editor.getJSON()));

/** Each top-level Block of the note: its text, without Citations. */
const blocksOf = (editor: Locator) =>
  editor.locator(":scope > *").evaluateAll((all) =>
    all.map((block) => {
      const copy = block.cloneNode(true) as Element;
      for (const citation of copy.querySelectorAll('[data-testid="citation"]')) citation.remove();
      return (copy.textContent ?? "").trim();
    }),
  );

async function addTides(window: Page) {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
}

/** A new Mind, from the sidebar's +, with the mouse. */
async function newMindByMouse(window: Page) {
  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-new-mind"));
}

/** A Mind with a title and two Notes, the cursor on the empty line between them. */
async function mindWithNotes(window: Page, title: string) {
  await newMindByMouse(window);
  await window.getByTestId("mind-title").fill(title);
  await enterContent(window);
  const editor = window.getByTestId("mind-editor");
  await window.keyboard.type("Spring tides come twice a month.");
  await window.keyboard.press("Enter");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Check the tide tables next.");
  await pointAndClick(window, editor.locator(":scope > p").nth(1));
  await expect(editor.locator(":scope > p").nth(1)).toHaveClass(/has-focus/);
  return editor;
}

async function start(): Promise<{
  app: Awaited<ReturnType<typeof launchApp>>["app"];
  window: Page;
}> {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await connectLocalChatModel(window);
  await addTides(window);
  return { app, window };
}

test("an Answer streams into the assistant and leaves the Mind as it was; Add to note puts it at the cursor with its Citation found, and Undo takes it out", async () => {
  const { app, window } = await start();
  const editor = await mindWithNotes(window, "Reading notes: tides");
  const before = await noteJSON(editor);
  const assistant = window.getByTestId("assistant");

  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("When do spring tides happen?");
  await window.keyboard.press("Enter");
  await expect(assistant.getByTestId("assistant-question")).toContainText(
    "When do spring tides happen?",
  );
  const answer = assistantAnswers(window).first();
  await expect(answer).toHaveAttribute("data-status", "streaming");
  await expect(answer.getByTestId("answer-stop")).toBeVisible();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("citation-chip")).toHaveAttribute("data-check", "found");
  await expect(answer.getByTestId("margin-check")).toHaveAttribute("data-check", "found");
  await expect(answer.getByTestId("answer-model")).toHaveText("fake-model");
  // The conversation is named by its first Question.
  await expect(assistant.getByTestId("assistant-title")).toContainText(
    "When do spring tides happen",
  );
  // The Answer's text is smaller than the note's: the note stays the largest on screen.
  const answerSize = await answer
    .locator(".assistant-answer-text p")
    .first()
    .evaluate((p) => Number.parseFloat(getComputedStyle(p).fontSize));
  const noteSize = await editor
    .locator(":scope > p")
    .first()
    .evaluate((p) => Number.parseFloat(getComputedStyle(p).fontSize));
  expect(answerSize).toBeLessThan(noteSize);
  await screenshot(window, "assistant-answer");

  // A Question never changes a Mind on its own.
  expect(await noteJSON(editor)).toBe(before);
  await expect(editor.getByTestId("question")).toHaveCount(0);
  await expect(editor.getByTestId("answer")).toHaveCount(0);

  // A follow-up goes into the same conversation.
  const second = await askInAssistant(window, "And neap tides?");
  await expect(assistant.getByTestId("assistant-question")).toHaveCount(2);
  await expect(second).toContainText("That is all.");
  expect(await noteJSON(editor)).toBe(before);

  // Copy takes the Answer with its Citation: as HTML that pastes back as a Citation, and as
  // text that names its source.
  const copied = await stubClipboard(window);
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  await pointAndClick(window, answer.getByTestId("answer-copy"));
  await expect(answer.getByTestId("answer-copy")).toHaveText("Copied");
  const clipboard = await copied();
  expect(clipboard["text/html"]).toContain('data-type="citation"');
  expect(clipboard["text/plain"]).toContain("Your Documents answer this");
  expect(clipboard["text/plain"]).toContain("[Tides, p. 2]");

  // Add to note: at the note's cursor, named in its tooltip; the Citation is found there too.
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  const add = answer.getByTestId("answer-add");
  await expect(add).toHaveAttribute("title", /Add to “Reading notes: tides”/);
  await pointAndClick(window, add);
  await expect(answer.getByTestId("answer-added")).toContainText("Added to “Reading notes: tides”");
  await expect
    .poll(() => blocksOf(editor))
    .toEqual([
      "Spring tides come twice a month.",
      "Your Documents answer this .",
      "That is all.",
      "Check the tide tables next.",
    ]);
  const added = editor.getByTestId("citation-chip");
  await expect(added).toHaveCount(1);
  await expect(added).toHaveAttribute("data-check", "found");
  // The link back shows in the margin.
  await expect(window.getByTestId("conversation-link")).toHaveCount(1);
  await screenshot(window, "assistant-added");

  // Undo, on the line under the Answer: the note is as it was.
  await pointAndClick(window, answer.getByTestId("answer-added-undo"));
  await expect.poll(() => noteJSON(editor)).toBe(before);
  await expect(answer.getByTestId("answer-added")).toHaveCount(0);

  // Added again, ⌘Z in the note takes it back too.
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  await pointAndClick(window, answer.getByTestId("answer-add"));
  await expect.poll(() => blocksOf(editor)).toContain("Your Documents answer this .");
  await pointAndClick(window, editor.locator(":scope > p").first());
  await window.keyboard.press("ControlOrMeta+z");
  await expect.poll(() => blocksOf(editor)).not.toContain("Your Documents answer this .");
  await app.close();
});

test("an Answer goes to another Mind through the ▾, and to a new Mind as changes to accept; the link back opens the conversation at that Answer, and says so once it is deleted", async () => {
  const { app, window } = await start();
  await newMindByMouse(window);
  await window.getByTestId("mind-title").fill("Chapter 2 outline");
  await enterContent(window);
  await window.keyboard.type("Outline.");
  const editor = await mindWithNotes(window, "Reading notes: tides");
  const before = await noteJSON(editor);
  const answer = await askInAssistant(window, "When do spring tides happen?");

  // ▾: the Mind in focus first, then the recent ones, a search, and New Mind.
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  await pointAndClick(window, answer.getByTestId("answer-add-to"));
  const picker = window.getByTestId("add-to-picker");
  await expect(picker).toBeVisible();
  await expect(picker.getByTestId("add-to-search")).toBeFocused();
  const options = picker.getByTestId("add-to-option");
  await expect(options.first()).toContainText("Reading notes: tides");
  await expect(options.filter({ hasText: "Chapter 2 outline" })).toHaveCount(1);
  await expect(options.last()).toContainText("New Mind");
  await screenshot(window, "assistant-add-to-picker");
  await window.keyboard.type("chapter");
  await expect(options).toHaveCount(2);
  await window.keyboard.press("Enter");
  await expect(picker).toBeHidden();
  await expect(answer.getByTestId("answer-added")).toContainText("Added to “Chapter 2 outline”");
  // The Mind in focus is as it was.
  expect(await noteJSON(editor)).toBe(before);

  // In the other Mind, at its end, with its link back.
  await pointAndClick(
    window,
    window.getByTestId("mind-list-item").filter({ hasText: "Chapter 2 outline" }),
  );
  const chapter = window.getByTestId("mind-editor");
  await expect
    .poll(() => blocksOf(chapter))
    .toEqual(["Outline.", "Your Documents answer this .", "That is all."]);
  await expect(chapter.getByTestId("citation-chip")).toHaveAttribute("data-check", "found");

  // Hovering the added text names the conversation and the Question.
  await pointAt(window, chapter.locator(":scope > p").nth(1));
  const card = window.getByTestId("origin-card");
  await expect(card).toBeVisible();
  await expect(card.getByTestId("origin-open")).toContainText(
    "From the conversation “When do spring tides happen”",
  );
  await expect(card.getByTestId("origin-question")).toContainText(
    "You asked: When do spring tides happen?",
  );

  // A new chat; the link back opens the conversation again, at that Answer, flashed.
  await window.keyboard.press("ControlOrMeta+Shift+o");
  await expect(window.getByTestId("assistant-empty")).toBeVisible();
  await pointAt(window, chapter.locator(":scope > p").nth(1));
  await pointAndClick(window, card.getByTestId("origin-open"));
  await expect(assistantAnswers(window)).toHaveCount(1);
  await expect(assistantAnswers(window).first()).toHaveAttribute("data-sought", "true");

  // The Block menu has the same.
  await window.keyboard.press("ControlOrMeta+Shift+o");
  await (await slideToHandle(window, chapter.locator(":scope > p").nth(1))).click();
  await pointAndClick(window, window.getByTestId("block-menu-open-conversation"));
  await expect(assistantAnswers(window)).toHaveCount(1);

  // ▾ → New Mind: named as the conversation, in a new tab, its content as changes to accept.
  const shown = assistantAnswers(window).first();
  await pointAt(window, shown.getByTestId("assistant-answer-text"));
  await pointAndClick(window, shown.getByTestId("answer-add-to"));
  await pointAndClick(window, window.getByTestId("add-to-option").last());
  await expect(shown.getByTestId("answer-added")).toContainText(
    "Made the Mind “When do spring tides happen” from this Answer",
  );
  const tab = window.getByTestId("mind-tab").filter({ hasText: "When do spring tides happen" });
  await expect(tab).toHaveAttribute("aria-selected", "true");
  const made = window.getByTestId("mind-editor");
  await expect(made.locator('[data-proposal="insert"]')).toHaveCount(2);
  await expect(window.getByTestId("proposals-bar")).toBeVisible();
  await screenshot(window, "assistant-new-mind");
  // Undo removes it, and its tab.
  await pointAndClick(window, shown.getByTestId("answer-added-undo"));
  await expect(tab).toHaveCount(0);

  // Deleted, the conversation's link says so.
  await pointAndClick(
    window,
    window.getByTestId("mind-list-item").filter({ hasText: "Chapter 2 outline" }),
  );
  await window.evaluate(async () => {
    const bridge = (
      globalThis as unknown as {
        incarnamind: {
          listConversations(): Promise<{ id: string }[]>;
          deleteConversation(id: string): Promise<void>;
        };
      }
    ).incarnamind;
    for (const conversation of await bridge.listConversations()) {
      await bridge.deleteConversation(conversation.id);
    }
  });
  // Onto the added text from outside it, so its card is asked for afresh.
  await pointAt(window, window.getByTestId("mind-title"));
  await pointAt(window, window.getByTestId("mind-editor").locator(":scope > p").nth(1));
  await expect(window.getByTestId("origin-gone")).toContainText(
    "From the conversation “When do spring tides happen”, now in Recently deleted",
  );
  await app.close();
});

test("with no Mind in focus, Add to note opens the ▾, and adds through it", async () => {
  const { app, window } = await start();
  await mindWithNotes(window, "Reading notes: tides");
  const answer = await askInAssistant(window, "When do spring tides happen?");
  // The Library in front: no Mind in focus.
  await pointAndClick(window, window.getByTestId("open-library"));
  await expect(window.getByTestId("assistant-reads")).toContainText("Searches All Documents");
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  const add = answer.getByTestId("answer-add");
  await expect(add).toHaveAttribute("title", "Choose a Mind to add it to");
  await pointAndClick(window, add);
  const picker = window.getByTestId("add-to-picker");
  await expect(picker).toBeVisible();
  await pointAndClick(
    window,
    picker.getByTestId("add-to-option").filter({ hasText: "Reading notes: tides" }),
  );
  await expect(answer.getByTestId("answer-added")).toContainText("Added to “Reading notes: tides”");
  await app.close();
});

test("the keyboard reaches it all: ⌘J, Tab through an Answer's actions, ⌘⇧C, ⌘⇧A, and Esc folds, then goes back to the Mind", async () => {
  const { app, window } = await start();
  const editor = await mindWithNotes(window, "Reading notes: tides");
  const answer = await askInAssistant(window, "When do spring tides happen?");
  const copied = await stubClipboard(window);

  // Tab from the composer goes back through the Answer's actions, by name.
  await window.keyboard.press("Shift+Tab");
  const named = async () =>
    window.evaluate(() => document.activeElement?.getAttribute("aria-label") ?? "");
  const seen: string[] = [];
  for (let i = 0; i < 12 && !seen.some((label) => label.startsWith("Copy the Answer")); i++) {
    seen.push(await named());
    await window.keyboard.press("Shift+Tab");
  }
  expect(seen).toEqual(
    expect.arrayContaining([
      "Regenerate",
      "Add to another Mind or a new Mind",
      expect.stringMatching(/^Add to “Reading notes: tides”/),
    ]),
  );
  await expect(answer.getByTestId("answer-meta")).toHaveAttribute("aria-label", "Answer actions");

  // ⌘⇧C copies the newest Answer; ⌘⇧A adds it to the Mind in focus, at its cursor. The keys
  // as a keyboard reports them with Shift held: "C" and "A" (Linux's editor reads a lowercase
  // letter with Shift as the same letter without it, and Ctrl+A selects all).
  await window.keyboard.press("ControlOrMeta+Shift+C");
  await expect.poll(async () => (await copied())["text/plain"] ?? "").toContain("[Tides, p. 2]");
  await pointAndClick(window, editor.locator(":scope > p").nth(1));
  await window.keyboard.press("ControlOrMeta+Shift+A");
  await expect
    .poll(() => blocksOf(editor))
    .toEqual([
      "Spring tides come twice a month.",
      "Your Documents answer this .",
      "That is all.",
      "Check the tide tables next.",
    ]);
  // The focus stayed in the note.
  await expect(editor).toBeFocused();

  // Esc in the assistant folds it, the focus on what shows it again; a second Esc, back to the Mind.
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.press("Escape");
  const assistant = window.getByTestId("assistant");
  await expect(assistant).toHaveAttribute("data-folded", "true");
  await expect(assistant.getByTestId("assistant-fold")).toBeFocused();
  await window.keyboard.press("Escape");
  await expect(editor).toBeFocused();
  // ⌘J unfolds it, into the composer.
  await window.keyboard.press("ControlOrMeta+j");
  await expect(assistant).not.toHaveAttribute("data-folded");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await app.close();
});

test("the ⋯ menu adds the last Answer to a new Mind and clears the conversation, with Undo; in Chinese too", async () => {
  const { app, window } = await start();
  await mindWithNotes(window, "Reading notes: tides");
  const more = window.getByTestId("assistant-more");
  // Empty, they say why they can't be used.
  await pointAndClick(window, more);
  await expect(window.getByTestId("assistant-menu-clear")).toHaveAttribute("aria-disabled", "true");
  await window.keyboard.press("Escape");

  await askInAssistant(window, "When do spring tides happen?");
  await pointAndClick(window, more);
  await pointAndClick(window, window.getByTestId("assistant-menu-last-to-new-mind"));
  const tab = window.getByTestId("mind-tab").filter({ hasText: "When do spring tides happen" });
  await expect(tab).toHaveAttribute("aria-selected", "true");
  await expect(window.getByTestId("mind-editor").locator('[data-proposal="insert"]')).toHaveCount(
    2,
  );

  await pointAndClick(window, more);
  await pointAndClick(window, window.getByTestId("assistant-menu-clear"));
  await expect(window.getByTestId("assistant-empty")).toBeVisible();
  await expect(window.getByTestId("assistant-title")).toContainText("New chat");
  await expect(window.getByTestId("undo-message")).toHaveText(
    "Cleared “When do spring tides happen”",
  );
  await pointAndClick(window, window.getByTestId("undo-action"));
  await expect(assistantAnswers(window)).toHaveCount(1);

  // In Chinese.
  await window.evaluate(async () => {
    const bridge = (
      globalThis as unknown as { incarnamind: { updateSettings(patch: unknown): Promise<unknown> } }
    ).incarnamind;
    await bridge.updateSettings({ user: { language: "zh-CN" } });
  });
  const answer = assistantAnswers(window).first();
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  await expect(answer.getByTestId("answer-add")).toHaveText("添加到笔记");
  await expect(answer.getByTestId("answer-copy")).toHaveText("复制");
  await expect(answer.getByTestId("answer-regenerate")).toHaveText("重新生成");
  await screenshot(window, "assistant-answer-zh");
  await app.close();
});
