import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  askInThread,
  createDataFolder,
  dismissChatSetup,
  dragBy,
  enterContent,
  launchApp,
  pointAndClick,
  pointAt,
  removeDataFolder,
  setWindowSize,
  slideToHandle,
  useLocalChatModel,
} from "./app";

/**
 * The thread above the composer (#201): a Question asked in a Mind is
 * answered in the thread, never in the note; Add to note keeps an Answer (or
 * a part of it) at the cursor, its Citations checked again there, with Undo;
 * the ▾ adds it to another Mind, with a link back that opens the thread at
 * that Answer; ⌘J, Esc and Tab reach it all. Driven as a person would, with
 * the mouse slid onto what it clicks and the keys typed. Set
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

/** Each top-level Block of the note: a Question, an Answer, or a Note's text (without Citations). */
const blocksOf = (editor: Locator) =>
  editor.locator(":scope > *").evaluateAll((all) =>
    all.map((block) => {
      if (block.matches(".node-question")) return "question";
      if (block.matches(".node-answer")) return "answer";
      const copy = block.cloneNode(true) as Element;
      for (const citation of copy.querySelectorAll('[data-testid="citation"]')) citation.remove();
      return (copy.textContent ?? "").trim();
    }),
  );

async function addTides(window: Page) {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
}

/** A Mind with a title and two Notes, the cursor on the empty line between them. */
async function mindWithNotes(window: Page, title: string) {
  await pointAndClick(window, window.getByTestId("new-mind"));
  const titleField = window.getByTestId("mind-title");
  await titleField.fill(title);
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

test("an Answer streams into the thread and leaves the note as it was; Add to note puts it at the cursor with its Citation found, and Undo takes it out", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addTides(window);
  const editor = await mindWithNotes(window, "Reading notes: tides");
  const before = await noteJSON(editor);

  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("When do spring tides happen?");
  await window.keyboard.press("Enter");
  const thread = window.getByTestId("thread");
  await expect(thread).toBeVisible();
  await expect(thread.getByTestId("thread-question").locator(".thread-question-text")).toHaveText(
    "When do spring tides happen?",
  );
  const answer = thread.getByTestId("answer").first();
  await expect(answer).toHaveAttribute("data-status", "streaming");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("citation-chip")).toHaveAttribute("data-check", "found");
  await expect(thread.getByTestId("margin-check")).toHaveAttribute("data-check", "found");
  // The thread sits on the composer, and the composer stays where it was, uncovered.
  const threadBox = await thread.boundingBox();
  const composerBox = await window.getByTestId("composer").boundingBox();
  if (!threadBox || !composerBox) throw new Error("The thread or the composer isn't visible.");
  expect(Math.abs(threadBox.y + threadBox.height - composerBox.y)).toBeLessThanOrEqual(1);
  expect(Math.abs(threadBox.x - composerBox.x)).toBeLessThanOrEqual(1);
  expect(Math.abs(threadBox.width - composerBox.width)).toBeLessThanOrEqual(1);
  await expect(window.getByTestId("composer-input")).toHaveAttribute(
    "placeholder",
    "Ask a follow-up…",
  );
  await screenshot(window, "thread-answer");

  expect(await noteJSON(editor)).toBe(before);
  await expect(editor.getByTestId("question")).toHaveCount(0);
  await expect(editor.getByTestId("answer")).toHaveCount(0);

  await window.keyboard.type("And neap tides?");
  await window.keyboard.press("Enter");
  await expect(thread.getByTestId("thread-question")).toHaveCount(2);
  await expect(thread.getByTestId("answer").nth(1)).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  expect(await noteJSON(editor)).toBe(before);

  // Copy takes the Answer with its Citations: as HTML that pastes back as Citations, and as text
  // that names their sources. (The page's clipboard is stubbed: the machine's own stays as it was.)
  await window.evaluate(() => {
    const copied: Record<string, string> = {};
    (globalThis as { copied?: unknown }).copied = copied;
    navigator.clipboard.write = async (items: ClipboardItem[]) => {
      for (const item of items) {
        for (const type of item.types) copied[type] = await (await item.getType(type)).text();
      }
    };
  });
  await pointAt(window, answer.locator(".answer-content p").first());
  await pointAndClick(window, answer.getByTestId("answer-copy"));
  await expect(answer.getByTestId("answer-copy")).toHaveText("Copied");
  const copied = await window.evaluate(
    () => (globalThis as { copied?: Record<string, string> }).copied ?? {},
  );
  expect(copied["text/html"]).toContain('data-type="citation"');
  expect(copied["text/plain"]).toContain("Your Documents answer this");
  expect(copied["text/plain"]).toContain("Tides, p. 2");

  await pointAt(window, answer.locator(".answer-content p").first());
  const add = answer.getByTestId("answer-add");
  await expect(add).toHaveCSS("opacity", "1");
  await pointAndClick(window, add);
  await expect(answer.getByTestId("thread-added")).toContainText("Added to note");
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
  await expect(window.locator(".mind-editor-frame").getByTestId("margin-check")).toHaveCount(1);
  await screenshot(window, "thread-added");

  await pointAndClick(window, answer.getByTestId("thread-added-undo"));
  await expect.poll(() => noteJSON(editor)).toBe(before);
  await expect(answer.getByTestId("thread-added")).toHaveCount(0);
  await app.close();
});

test("a selection of an Answer goes to another Mind through the ▾ picker, with a link back that opens the thread at that Answer", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addTides(window);
  await pointAndClick(window, window.getByTestId("new-mind"));
  await window.getByTestId("mind-title").fill("Chapter 2 outline");
  await enterContent(window);
  await window.keyboard.type("Outline.");
  await mindWithNotes(window, "Reading notes: tides");
  const answer = await askInThread(window, "When do spring tides happen?");

  // Select the Answer's first paragraph with the mouse (a triple-click), Citation and all.
  const first = answer.locator(".answer-content p").first();
  const box = await first.boundingBox();
  if (!box) throw new Error("The Answer isn't visible.");
  await window.mouse.move(box.x + 20, box.y + box.height / 2, { steps: 6 });
  await window.mouse.click(box.x + 20, box.y + box.height / 2, { clickCount: 3 });
  await expect
    .poll(() => window.evaluate(() => getSelection()?.toString() ?? ""))
    .toContain("Your Documents answer this");
  expect(await window.evaluate(() => getSelection()?.toString() ?? "")).not.toContain(
    "That is all",
  );

  // ▾ opens the picker: This Mind first, then the Minds edited last, then New Mind….
  await pointAndClick(window, answer.getByTestId("answer-add-to"));
  const picker = window.getByTestId("add-to-picker");
  await expect(picker).toBeVisible();
  await expect(picker).not.toHaveCSS("background-color", "rgba(0, 0, 0, 0)");
  await expect(picker.getByTestId("add-to-search")).toBeFocused();
  const options = picker.getByTestId("add-to-option");
  await expect(options.first()).toContainText("This Mind");
  await expect(options.filter({ hasText: "Chapter 2 outline" })).toHaveCount(1);
  await expect(options.last()).toContainText("New Mind…");
  await screenshot(window, "thread-picker");
  await window.keyboard.type("chap");
  await expect(options.first()).toContainText("Chapter 2 outline");
  await window.keyboard.press("Enter");
  await expect(picker).toBeHidden();
  await expect(answer.getByTestId("thread-added")).toContainText("Added to “Chapter 2 outline”");
  await expect(answer.getByTestId("thread-added-undo")).toBeFocused();

  const sidebarRow = window.getByTestId("mind-list-item").filter({ hasText: "Chapter 2 outline" });
  await pointAndClick(window, sidebarRow);
  const other = window.getByTestId("mind-editor");
  await expect.poll(() => blocksOf(other)).toEqual(["Outline.", "Your Documents answer this ."]);
  await expect(other.getByTestId("citation-chip")).toHaveAttribute("data-check", "found");
  const added = other.locator(":scope > p").last();
  await expect(added).toHaveAttribute("title", "From the chat “When do spring tides happen?”");

  // Its link back, in its Block menu, opens the chat beside it, at that Answer.
  await (await slideToHandle(window, added)).click();
  const openChat = window.getByTestId("block-menu-open-chat");
  await expect(openChat).toHaveText("Open the chat it came from");
  await pointAndClick(window, openChat);
  await expect(window.getByTestId("mind-title")).toHaveValue("Reading notes: tides");
  const thread = window.getByTestId("thread");
  await expect(thread).toHaveAttribute("data-open", "true");
  await expect(thread.getByTestId("answer")).toBeFocused();
  // It opened in a tab of its own: Chapter 2 outline's tab is still there.
  await expect(window.getByRole("tab", { name: /Chapter 2 outline/ })).toBeVisible();
  await app.close();
});

test("the keyboard reaches the thread: ⌘J focuses the composer, Tab reaches the Answer's actions by name, Esc folds the thread and then goes to the note; in Chinese too", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addTides(window);
  const editor = await mindWithNotes(window, "Reading notes: tides");
  await askInThread(window, "When do spring tides happen?");
  const thread = window.getByTestId("thread");

  // Back from the composer, the Answer's actions come first, each named for a screen reader.
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  const names: string[] = [];
  for (let step = 0; step < 6; step++) {
    await window.keyboard.press("Shift+Tab");
    names.push(
      await window.evaluate(() => document.activeElement?.getAttribute("aria-label") ?? ""),
    );
  }
  expect(names).toEqual(
    expect.arrayContaining([
      "Regenerate this Answer",
      "Add to another Mind…",
      expect.stringMatching(/^Add this Answer to this note, at the cursor/),
      expect.stringMatching(/^Copy this Answer with its Citations/),
    ]),
  );
  await expect(thread.getByRole("group", { name: "For this Answer" })).toBeVisible();

  // ⌘⇧A adds the focused Answer, at the cursor; ⌘Z in the thread takes it back.
  await thread.getByTestId("answer-copy").focus();
  await window.keyboard.press("ControlOrMeta+Shift+a");
  await expect(thread.getByTestId("thread-added")).toBeVisible();
  await expect(editor.getByTestId("citation-chip")).toHaveCount(1);
  await thread.getByTestId("thread-added-undo").focus();
  await window.keyboard.press("ControlOrMeta+z");
  await expect(editor.getByTestId("citation-chip")).toHaveCount(0);

  // Esc in the thread folds it to its header and goes to the composer; ⌘J then Esc, to the note.
  await thread.getByTestId("thread-fold").focus();
  await window.keyboard.press("Escape");
  await expect(thread).not.toHaveAttribute("data-open");
  await expect(thread.getByTestId("thread-subtitle")).toHaveText("When do spring tides happen?");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.press("Escape");
  await expect(editor).toBeFocused();

  // Unfolded again from its header; Esc from the composer folds it first.
  await pointAndClick(window, thread.getByTestId("thread-fold"));
  await expect(thread).toHaveAttribute("data-open", "true");
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.press("Escape");
  await expect(thread).not.toHaveAttribute("data-open");
  await expect(window.getByTestId("composer-input")).toBeFocused();

  await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "zh-CN" },
    }),
  );
  await pointAndClick(window, thread.getByTestId("thread-fold"));
  await expect(thread.locator(".thread-title")).toHaveText("就这个 Mind 提问");
  await expect(thread.getByTestId("thread-subtitle")).toHaveText(
    "回答留在这里，加入笔记后才会写进去",
  );
  await expect(thread.getByTestId("answer-add")).toHaveText("加入笔记");
  await expect(window.getByTestId("composer-input")).toHaveAttribute("placeholder", "继续追问…");
  await screenshot(window, "thread-zh");
  await app.close();
});

test("each Mind keeps its own chats, with the note, after a restart; New chat starts another", async () => {
  const first = await launchApp(dataDir, { fakeChat: true, home });
  await setWindowSize(first.app, first.window, 1440, 900);
  await dismissChatSetup(first.window);
  await useLocalChatModel(first.window);
  await mindWithNotes(first.window, "Tides");
  await askInThread(first.window, "What is a spring tide?");
  // New chat, from the thread's history: the next Question starts it.
  const thread = first.window.getByTestId("thread");
  await pointAndClick(first.window, thread.getByTestId("thread-history"));
  await pointAndClick(first.window, first.window.getByTestId("thread-new-chat"));
  await expect(thread.getByTestId("thread-empty")).toBeVisible();
  await expect(first.window.getByTestId("composer-input")).toBeFocused();
  await askInThread(first.window, "What is a neap tide?");
  await expect(thread.getByTestId("thread-question").locator(".thread-question-text")).toHaveText([
    "What is a neap tide?",
  ]);
  await mindWithNotes(first.window, "Empty");
  await expect(first.window.getByTestId("thread")).toHaveCount(0);
  await first.app.close();

  const second = await launchApp(dataDir, { fakeChat: true, home });
  const row = second.window.getByTestId("mind-list-item").filter({ hasText: /^Tides/ });
  await pointAndClick(second.window, row);
  const reopened = second.window.getByTestId("thread");
  // Folded to its header, naming the newest chat; its history lists both, newest first.
  await expect(reopened).not.toHaveAttribute("data-open");
  await expect(reopened.getByTestId("thread-subtitle")).toHaveText("What is a neap tide?");
  await pointAndClick(second.window, reopened.getByTestId("thread-history"));
  const chats = second.window.getByTestId("thread-history-chat");
  await expect(chats.locator(".thread-history-title")).toHaveText([
    "What is a neap tide?",
    "What is a spring tide?",
  ]);
  await pointAndClick(second.window, chats.last());
  await expect(reopened).toHaveAttribute("data-open", "true");
  await expect(reopened.getByTestId("thread-question").locator(".thread-question-text")).toHaveText(
    ["What is a spring tide?"],
  );
  await expect(reopened.getByTestId("answer")).toHaveAttribute("data-status", "done");
  await second.app.close();
});

async function heights(window: Page) {
  const thread = await window.getByTestId("thread").boundingBox();
  const mind = await window.getByTestId("composer-dock").locator("xpath=..").boundingBox();
  if (!thread || !mind) throw new Error("The thread or the Mind isn't visible.");
  return { thread: thread.height, mind: mind.height };
}

test("the thread opens at no more than a third of the Mind; its top edge makes it taller, by mouse or keys, remembered after a restart, and a double-click puts it back", async () => {
  const first = await launchApp(dataDir, { fakeChat: true, home });
  await setWindowSize(first.app, first.window, 1440, 900);
  await dismissChatSetup(first.window);
  await useLocalChatModel(first.window);
  await addTides(first.window);
  await mindWithNotes(first.window, "Tides");
  for (const question of [
    "When do spring tides happen?",
    "And neap tides?",
    "Why twice a month?",
  ]) {
    await askInThread(first.window, question);
  }
  const opened = await heights(first.window);
  expect(opened.thread).toBeLessThanOrEqual(opened.mind / 3 + 1);

  const rod = first.window.getByTestId("thread-resize");
  await dragBy(first.window, rod, 0, -150);
  await expect
    .poll(async () => (await heights(first.window)).thread)
    .toBeGreaterThan(opened.thread + 140);
  const dragged = (await heights(first.window)).thread;
  expect(dragged).toBeLessThan(opened.thread + 160);
  await rod.focus();
  await first.window.keyboard.press("ArrowUp");
  await expect.poll(async () => (await heights(first.window)).thread).toBeGreaterThan(dragged + 5);
  const chosen = (await heights(first.window)).thread;
  await first.app.close();

  const second = await launchApp(dataDir, { fakeChat: true, home });
  await setWindowSize(second.app, second.window, 1440, 900);
  const thread = second.window.getByTestId("thread");
  await pointAndClick(second.window, thread.getByTestId("thread-fold"));
  await expect(thread).toHaveAttribute("data-open", "true");
  await expect
    .poll(async () => Math.round((await heights(second.window)).thread))
    .toBe(Math.round(chosen));
  const again = second.window.getByTestId("thread-resize");
  const box = await again.boundingBox();
  if (!box) throw new Error("The thread's top edge isn't visible.");
  await second.window.mouse.move(box.x + box.width / 2 - 40, box.y - 2);
  await second.window.mouse.move(box.x + box.width / 2, box.y, { steps: 6 });
  await second.window.mouse.dblclick(box.x + box.width / 2, box.y);
  await expect
    .poll(async () => {
      const now = await heights(second.window);
      return now.thread <= now.mind / 3 + 1;
    })
    .toBe(true);
  await second.app.close();
});
