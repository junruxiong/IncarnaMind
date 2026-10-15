import { access, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import {
  addToNote,
  assistantAnswers,
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  linkedFolderRow,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  showSourceLocations,
  useLocalChatModel,
} from "./app";

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

/**
 * Moves the mouse onto an element as a person does, in steps, checks that it
 * is what is under the pointer there (nothing covers it), and clicks.
 * `atX`: that far into it from its left edge, instead of its middle;
 * `through`: it lets the pointer through to what is under it, which is clicked.
 */
async function pointAndClick(
  window: Page,
  target: Locator,
  { atX, through = false }: { atX?: number; through?: boolean } = {},
): Promise<void> {
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  const x = box.x + (atX ?? box.width / 2);
  const y = box.y + box.height / 2;
  await window.mouse.move(x, y, { steps: 8 });
  const hit = await target.evaluate(
    (element, [px, py]) => {
      const under = document.elementFromPoint(px as number, py as number);
      return under !== null && (under === element || element.contains(under));
    },
    [x, y],
  );
  expect(hit).toBe(!through);
  await window.mouse.click(x, y);
}

/** The "Get started" card's count, e.g. "· 1 of 3". */
const progress = (window: Page) => window.getByTestId("get-started-count");

/** Words a newcomer shouldn't meet on the first screens: the left column of #763's glossary. */
const JARGON = [
  /\bindex(?:es|ed|ing)?\b/i,
  /\bPassages?\b/,
  /\bMCP\b/,
  /\bchat model/i,
  /\bembedding/i,
  /\brerank/i,
  /\bproviders?\b/i,
  /\blocal mode\b/i,
  /\bOCR\b/,
  /\btokens?\b/i,
  /\bstdio\b/,
  /\bendpoint/i,
  /\bOAuth\b/,
  /\b(?:claude|gpt|qwen|llama|gemini|deepseek)-[\w.:-]+/i,
  /索引|嵌入|重排序|重新排序|对话模型|提供商|本地模式/,
];

/** Each region's text, as shown, has none of the glossary's words. */
async function expectPlainWords(regions: Locator[]): Promise<void> {
  for (const region of regions) {
    const text = await region.innerText();
    for (const word of JARGON) expect(text, `${word} on the first screens`).not.toMatch(word);
  }
}

test("a first run opens on the example Mind: its Answer is written in advance, its Citations are checked against its Documents, and the chat setup waits", async () => {
  const { window } = await launchApp(dataDir, { examples: true });

  const pane = window.getByTestId("mind-pane");
  await expect(window.getByTestId("mind-title")).toHaveValue("Where tea comes from");
  await expect(pane.getByTestId("example-banner")).toContainText("This is an example Mind.");
  // "Example" on the Mind and on its Documents' Linked folder; both Documents are listed.
  await expect(window.getByTestId("mind-list-item").getByTestId("example-chip")).toHaveText(
    "Example",
  );
  await expect(window.getByTestId("document-list-item")).toHaveText([
    /Tea · Wikipedia|茶 · 维基百科/,
    /Tea · Wikipedia|茶 · 维基百科/,
  ]);
  // No model needed: no setup over it, and no "set up a chat model" notice in it.
  await expect(window.getByTestId("chat-setup")).toBeHidden();
  await expect(pane.getByTestId("chat-readiness")).toHaveCount(0);

  const answer = pane.getByTestId("answer");
  await expect(answer.getByTestId("answer-example")).toHaveText(
    "Example Answer, written in advance: no model needed to try it",
  );
  // Both Citations are checked against the Documents' text: the quotes are found.
  const checks = pane.getByTestId("margin-check");
  await expect(checks).toHaveCount(2);
  await expect(checks.nth(0)).toHaveAttribute("data-check", "found", { timeout: 20_000 });
  await expect(checks.nth(1)).toHaveAttribute("data-check", "found", { timeout: 20_000 });

  // The Linked folder's chip, as on disk, once its status (indexing) leaves the room for it.
  await showSourceLocations(window);
  await expect(linkedFolderRow(window, "Examples").getByTestId("example-chip")).toBeVisible({
    timeout: 30_000,
  });

  await expect(progress(window)).toHaveText("· 0 of 3");
  // Clicking a green check opens its card and the Document at the quote, and ticks the first step.
  await pointAndClick(window, checks.nth(0));
  await expect(window.getByTestId("citation-card")).toBeVisible();
  await expect(window.getByTestId("viewer")).toBeVisible();
  await expect(progress(window)).toHaveText("· 1 of 3");
  await expect(
    window.locator('[data-testid="get-started-step"][data-step="citation"] [data-done]'),
  ).toContainText("Check a Citation in the example");

  // The banner's way on: adding one's own Documents, in place, as Get started's step does (#766),
  // asking for no model: that waits for the first own Question.
  await window.keyboard.press("Escape");
  const addOwn = pane.getByTestId("example-add-own");
  await expect(addOwn).toHaveText("Add your Documents");
  await pointAndClick(window, addOwn);
  const ways = window.getByTestId("example-add-items");
  await expect(ways.getByRole("menuitem")).toHaveText(["Add files…", "Link a folder…"]);
  await expect(ways.getByRole("menuitem").first()).toBeFocused();
  await expect(window.getByTestId("mind-title")).toHaveValue("Where tea comes from");
  await expect(window.getByTestId("chat-setup")).toHaveCount(0);
  await window.keyboard.press("Escape");
  await expect(addOwn).toBeFocused();
  await setLanguage(window, "zh-CN");
  await expect(addOwn).toHaveText("添加你的文档");
  await pointAndClick(window, addOwn);
  await expect(ways.getByRole("menuitem")).toHaveText(["添加文件…", "添加同步文件夹…"]);
});

test("removing the examples deletes the example Mind, its Documents and the copies of their files, and they aren't made again", async () => {
  const first = await launchApp(dataDir, { examples: true });
  const window = first.window;
  await expect(window.getByTestId("example-banner")).toBeVisible();

  await pointAndClick(window, window.getByTestId("example-remove"));
  await expect(window.getByTestId("mind-list-item")).toHaveCount(0);
  await expect(window.getByTestId("document-list-item")).toHaveCount(0);
  await expect(linkedFolderRow(window, "Examples")).toHaveCount(0);
  await expect(window.getByTestId("mind-none-open")).toBeVisible();
  // Moving on from the example: setting up a chat model is offered now.
  await dismissChatSetup(window);
  await expect
    .poll(() =>
      access(join(dataDir, "Examples")).then(
        () => "there",
        () => "gone",
      ),
    )
    .toBe("gone");
  await first.app.close();

  // A first run happens once: the next launch doesn't make them again.
  const again = await launchApp(dataDir, { examples: true });
  await expect(again.window.getByTestId("mind-none-open")).toBeVisible();
  await expect(again.window.getByTestId("mind-list-item")).toHaveCount(0);
  // "Check a Citation in the example" makes them again, and opens them.
  await pointAndClick(
    again.window,
    again.window.locator('[data-testid="get-started-step"][data-step="citation"] button'),
  );
  await expect(again.window.getByTestId("example-banner")).toBeVisible();
  await again.app.close();
});

test("with only the example Documents, the footer doesn't ask for a model to tag them; a Document of one's own without a model does", async () => {
  const own = join(sources, "Harbour notes.md");
  await writeFile(own, "# Harbour notes\n\nThe harbour master publishes the tide tables.\n");
  const { app, window } = await launchApp(dataDir, { examples: true });
  const items = window.getByTestId("document-list-item");
  const notice = window.getByTestId("sidebar-footer").getByTestId("tagging-waiting");

  // The examples are ready and their tagging waits for a model, but they need none: no notice.
  await expect(items).toHaveCount(2);
  for (const item of await items.all()) {
    await expect(item).toHaveAttribute("data-tagging", "waiting-for-provider", {
      timeout: 30_000,
    });
  }
  await expect(notice).toHaveCount(0);

  // A Document of one's own, with no model: the notice, as always.
  await window.getByTestId("add-documents-input").setInputFiles([own]);
  const added = items.filter({ hasText: "Harbour notes" });
  await expect(added).toHaveAttribute("data-tagging", "waiting-for-provider");
  await expect(notice).toBeVisible();
  await expect(notice).toContainText("No AI model for Tags");
  await expect(notice.getByTestId("tagging-waiting-setup")).toHaveText("Choose…");
  await setLanguage(window, "zh-CN");
  await expect(notice).toContainText("打标签需要 AI 模型");
  await expect(notice.getByTestId("tagging-waiting-setup")).toHaveText("选择…");
  await notice.getByTestId("tagging-waiting-setup").click();
  await expect(window.getByTestId("settings")).toHaveAttribute("data-page", "models");
  await app.close();
});

test("an empty Mind of one's own shows the three steps; Get started ticks itself as Documents are added and a Question is asked, and goes once all are done", async () => {
  await writeFile(
    join(sources, "Harbour notes.md"),
    "# Harbour notes\n\nThe harbour master publishes the tide tables every January.\n",
  );
  const { window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  await expect(progress(window)).toHaveText("· 0 of 3");

  // A Mind of one's own shows the three steps.
  await newMind(window);
  await dismissChatSetup(window);
  const guide = window.getByTestId("start-guide");
  await expect(guide).toBeVisible();
  await expect(guide.getByTestId("start-step")).toHaveCount(3);
  await expect(guide).toContainText("Only your Documents get checked Citations.");

  // Adding a file of one's own ticks "Add your Documents".
  await window
    .getByTestId("add-documents-input")
    .setInputFiles([join(sources, "Harbour notes.md")]);
  await expect(progress(window)).toHaveText("· 1 of 3");
  await expect(
    window.locator('[data-testid="get-started-step"][data-step="documents"] [data-done]'),
  ).toBeVisible();

  // The empty Mind's hint focuses the composer, for a Question about one's Documents.
  await useLocalChatModel(window);
  const hint = window.getByTestId("end-hint");
  await expect(hint).toHaveAttribute("data-place", "empty");
  // A first click puts the cursor on the line; then its link is armed.
  // (The first empty Mind's hint runs to two lines, so the click is on the line itself.)
  await pointAndClick(window, window.getByTestId("mind-editor").locator("p").first());
  await expect(hint).toHaveAttribute("data-armed", "true");
  await pointAndClick(window, hint.getByTestId("end-hint-ask"));
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("When are the tide tables published?");
  await window.keyboard.press("Enter");
  // It is asked in the assistant, and answered there: the Mind is left as it was, the steps
  // still under its line for the one left to do.
  await expect(window.getByTestId("assistant-question")).toContainText(
    "When are the tide tables published?",
  );
  await expect(assistantAnswers(window)).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  await expect(progress(window)).toHaveText("· 2 of 3");
  const editor = window.getByTestId("mind-editor");
  await expect(editor.getByTestId("question")).toHaveCount(0);
  await expect(guide).toBeVisible();

  // The last step, from the card: the example Mind opens, and checking a Citation there ends it.
  await pointAndClick(
    window,
    window.locator('[data-testid="get-started-step"][data-step="citation"] button'),
  );
  const checks = window.getByTestId("mind-pane").getByTestId("margin-check");
  await expect(checks.nth(0)).toHaveAttribute("data-check", "found", { timeout: 20_000 });
  await pointAndClick(window, checks.nth(0));
  await expect(window.getByTestId("get-started")).toHaveCount(0);
});

test('Get started\'s "Add your Documents" opens Add files… and Link a folder… at once, and the first screens and the start guide speak plainly, in English and Chinese (#766)', async () => {
  const notes = join(sources, "Harbour notes.md");
  await writeFile(notes, "# Harbour notes\n\nThe harbour master publishes the tide tables.\n");
  const { app, window } = await launchApp(dataDir, { examples: true });
  await setWindowSize(app, window, 1440, 900);
  const pane = window.getByTestId("mind-pane");
  const checks = pane.getByTestId("margin-check");
  await expect(checks.nth(0)).toHaveAttribute("data-check", "found", { timeout: 20_000 });
  const firstScreen = [
    window.getByTestId("sidebar"),
    pane.getByTestId("example-banner"),
    window.getByTestId("assistant"),
  ];
  await expectPlainWords(firstScreen);

  // Once the example's Citation is checked, adding Documents is the step to do.
  await pointAndClick(window, checks.nth(0));
  await window.keyboard.press("Escape");
  const step = window.locator('[data-testid="get-started-step"][data-step="documents"] button');
  await expect(step).toHaveAccessibleName("Add your Documents");
  await pointAndClick(window, step);
  const ways = window.getByTestId("get-started-add-items");
  await expect(ways.getByRole("menuitem")).toHaveText(["Add files…", "Link a folder…"]);
  await expect(ways.getByRole("menuitem").first()).toBeFocused();
  // No Mind of its own first, and no model setup over it.
  await expect(window.getByTestId("mind-title")).toHaveValue("Where tea comes from");
  await expect(window.getByTestId("chat-setup")).toBeHidden();
  await window.keyboard.press("Escape");
  await expect(ways).toBeHidden();
  await expect(step).toBeFocused();

  // In Chinese, at the narrowest window, the menu still opens whole, inside the window.
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 900, 760);
  await expect(step).toHaveAccessibleName("添加你的文档");
  await expectPlainWords(firstScreen);
  await pointAndClick(window, step);
  await expect(ways.getByRole("menuitem")).toHaveText(["添加文件…", "添加同步文件夹…"]);
  const menuBox = await ways.boundingBox();
  const width = await window.evaluate(() => globalThis.innerWidth);
  expect(menuBox?.x ?? -1).toBeGreaterThanOrEqual(0);
  expect((menuBox?.x ?? 0) + (menuBox?.width ?? 0)).toBeLessThanOrEqual(width);
  await interceptOpenDialog(app, notes);
  await pointAndClick(window, ways.getByRole("menuitem").first());
  await expect(
    window.getByTestId("document-list-item").filter({ hasText: "Harbour notes" }),
  ).toHaveCount(1);
  await expect(
    window.locator('[data-testid="get-started-step"][data-step="documents"] [data-done]'),
  ).toBeVisible();

  // The start guide in a Mind of one's own says where the Answer goes.
  await newMind(window);
  await dismissChatSetup(window);
  const guide = window.getByTestId("start-guide");
  await expect(guide).toContainText("回答会出现在助手里，一键加入笔记。");
  await expectPlainWords([guide]);
  await setLanguage(window, "en");
  await expect(guide).toContainText(
    "The Answer appears in the assistant; add it to your note with one click.",
  );
  await expectPlainWords([guide, window.getByTestId("sidebar")]);
});

test('"Ask your own Question" is ticked by an Answer that arrives, not by one that fails (#766)', async () => {
  const { app, window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  await setWindowSize(app, window, 1440, 900);
  await useLocalChatModel(window);
  const asked = window.locator(
    '[data-testid="get-started-step"][data-step="question"] [data-done]',
  );
  await expect(progress(window)).toHaveText("· 0 of 3");

  await window.getByTestId("mind-editor").click();
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("Send a blank reply");
  await window.keyboard.press("Enter");
  await expect(assistantAnswers(window).last()).toHaveAttribute("data-status", "failed", {
    timeout: 15_000,
  });
  await expect(asked).toHaveCount(0);
  await expect(progress(window)).toHaveText("· 0 of 3");

  await window.getByTestId("composer-input").click();
  await window.keyboard.type("Where does tea come from?");
  await window.keyboard.press("Enter");
  await expect(assistantAnswers(window).last()).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  await expect(asked).toBeVisible();
  await expect(progress(window)).toHaveText("· 1 of 3");
});

test("Get started can be hidden, and stays hidden", async () => {
  const first = await launchApp(dataDir, { examples: true });
  await pointAndClick(first.window, first.window.getByTestId("get-started-hide"));
  await expect(first.window.getByTestId("get-started")).toHaveCount(0);
  await first.app.close();

  const again = await launchApp(dataDir, { examples: true });
  await expect(again.window.getByTestId("example-banner")).toBeVisible();
  await expect(again.window.getByTestId("get-started")).toHaveCount(0);
  await again.app.close();
});

test("the hint at the end of a Mind focuses the composer when its link is clicked once the cursor is on its line, and is otherwise the line to write on", async () => {
  const { window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  const editor = window.getByTestId("mind-editor");
  const hint = window.getByTestId("end-hint");
  await expect(hint).toHaveAttribute("data-place", "afterAnswer");
  const shortcut = process.platform === "darwin" ? "⌘J" : "Ctrl+J";
  await expect(hint).toHaveText(
    `Try it: write a line here, or press ${shortcut} to ask your own Question`,
  );

  // Before the cursor is on the line, a click on the link puts the cursor there, to write.
  await expect(hint).not.toHaveAttribute("data-armed");
  await pointAndClick(window, hint.getByTestId("end-hint-ask"), { through: true });
  await expect(hint).toHaveAttribute("data-armed", "true");
  await expect(editor.getByTestId("question")).toHaveCount(1);
  await window.keyboard.type("Green and black tea come from the same plant.");
  const last = editor.locator(":scope > *").last();
  await expect(last).toHaveText("Green and black tea come from the same plant.");
  await expect(editor.getByTestId("question")).toHaveCount(1);
  // A Note at the end: no hint, as after an Answer only.
  await expect(hint).toHaveCount(0);

  // An empty line after a Note has none.
  await window.keyboard.press("Enter");
  await expect(hint).toHaveCount(0);
  await window.keyboard.press("Backspace");
  // Back on the Note: put the cursor on a new empty line right after the Answer.
  await pointAndClick(window, last, { atX: 2 });
  // The editor takes the click's place from the browser's selectionchange: wait, as a person would.
  await expect
    .poll(() =>
      editor.evaluate(
        (dom) => (dom as unknown as { editor: Editor }).editor.state.selection.$from.parentOffset,
      ),
    )
    .toBe(0);
  await window.keyboard.press("Enter");
  await window.keyboard.press("ArrowUp");
  // An empty line after an Answer, not at the end, shows the hint while it is written in.
  await expect(hint).toHaveAttribute("data-place", "afterAnswer");

  // Its link focuses the composer; the Question asked there is answered in the assistant, the
  // Mind left as it was.
  await pointAndClick(window, hint.getByTestId("end-hint-ask"));
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await useLocalChatModel(window);
  await window.keyboard.type("Which tea is drunk most in Britain?");
  await window.keyboard.press("Enter");
  const answer = assistantAnswers(window).last();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(editor.getByTestId("question")).toHaveCount(1);
  await expect(editor.locator('[data-origin="conversation"]')).toHaveCount(0);
  // Added to the note, it goes on the hint's line: right after the example's Answer, before the Note.
  await addToNote(window, answer);
  const order = await editor
    .locator(":scope > *")
    .evaluateAll((all) =>
      all.map((block) =>
        block.matches(".node-question")
          ? "question"
          : block.matches(".node-answer")
            ? "answer"
            : block.matches('[data-origin="conversation"]')
              ? "added"
              : (block.textContent ?? "").trim(),
      ),
    );
  const at = order.lastIndexOf("answer");
  expect(order.slice(at - 1, at + 2)).toEqual(["question", "answer", "added"]);
  expect(order.at(-1)).toBe("Green and black tea come from the same plant.");
  expect(order.slice(at + 1, -1).every((block) => block === "added")).toBe(true);
});

/** The banner's links to the other examples, by the group each opens, in the order shown. */
const moreLinks = (window: Page) =>
  window.getByTestId("example-banner").getByTestId("example-more-link");
const groupsShown = (window: Page) =>
  moreLinks(window).evaluateAll((links) => links.map((link) => link.getAttribute("data-group")));

test("the banner offers the other examples in an order picked once per install; each opens in a new tab and is made once (#123)", async () => {
  const first = await launchApp(dataDir, { examples: true });
  const { app, window } = first;
  await setWindowSize(app, window, 1100, 720);
  const banner = window.getByTestId("example-banner");
  await expect(banner).toBeVisible();

  // Nothing is asked before the example; its last line offers the three others.
  await expect(banner.getByTestId("example-more")).toContainText(
    "Try an example closer to your work:",
  );
  const order = await groupsShown(window);
  expect([...order].sort()).toEqual(["contracts", "meetings", "papers"]);
  const tabs = window.getByTestId("tab-title");
  await expect(tabs).toHaveText(["Where tea comes from"]);

  // Contracts opens beside the first example, which stays in its own tab.
  await moreLinks(window).and(window.locator('[data-group="contracts"]')).click();
  await expect(tabs).toHaveCount(2);
  await expect(tabs.first()).toHaveText("Where tea comes from");
  await expect(window.getByTestId("mind-list-item")).toHaveCount(2);
  // Its banner lists the others, the first example included.
  const contracts = await groupsShown(window);
  expect(contracts[0]).toBe("tea");
  expect([...contracts].sort()).toEqual(["meetings", "papers", "tea"]);

  // From the first example again, Contracts shows the one made, not a second.
  await tabs.first().click();
  await expect(moreLinks(window).first()).toBeVisible();
  await moreLinks(window).and(window.locator('[data-group="contracts"]')).click();
  await expect(window.getByTestId("mind-list-item")).toHaveCount(2);
  await expect(tabs).toHaveCount(2);
  await app.close();

  // The same order after a restart.
  const again = await launchApp(dataDir, { examples: true });
  const tea = again.window.getByTestId("tab-title").first();
  await expect(tea).toHaveText("Where tea comes from");
  await tea.click();
  await expect(again.window.getByTestId("example-banner")).toBeVisible();
  expect(await groupsShown(again.window)).toEqual(order);
  await again.app.close();
});

test("Remove the examples removes every example Mind made, and Undo brings them back (#123)", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true });
  await moreLinks(window).and(window.locator('[data-group="papers"]')).click();
  await expect(window.getByTestId("mind-list-item")).toHaveCount(2);

  await window.getByTestId("example-remove").click();
  await expect(window.getByTestId("mind-list-item")).toHaveCount(0);
  await dismissChatSetup(window);
  await window.getByTestId("undo-line").getByTestId("undo-action").click();
  await expect(window.getByTestId("mind-list-item")).toHaveCount(2);
  await app.close();
});
