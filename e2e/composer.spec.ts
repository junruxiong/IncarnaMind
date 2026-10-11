import { writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import * as Y from "yjs";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  assistantAnswers,
  createDataFolder,
  dismissChatSetup,
  enterContent,
  launchApp,
  pointAndClick,
  pointAt,
  removeDataFolder,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/**
 * The composer at the foot of the assistant (DESIGN.md, Composer; #253, #201):
 * asking shows the Question in the assistant's conversation and streams its
 * Answer under it, with checked Citations, and leaves the Mind as it was;
 * Stop keeps what streamed; ⌘J, Enter, Shift+Enter and Esc; the model chip,
 * remembered per Mind; Minds written before. Driven as a person would, with
 * the mouse slid onto what it clicks and the keys typed. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(target: Page | Locator, name: string): Promise<void> {
  if (SCREENSHOTS) await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

/** Three short pages: the line about spring tides is on page 2. */
const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
  { lines: ["Tide tables", "Harbours publish the times of high water every year."] },
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

/**
 * The Mind as wide as the approved mockups drew it (a 1440px window's card),
 * beside the assistant docked on the right (#253), which takes 368px of it.
 */
async function atMockupSize(app: ElectronApplication, window: Page) {
  await setWindowSize(app, window, 1600, 900);
  await window.evaluate(() => document.fonts.ready.then(() => undefined));
}

/** Each top-level Block of the Mind: "question", "answer", or a Note's text. */
const order = (editor: Locator) =>
  editor
    .locator(":scope > *")
    .evaluateAll((all) =>
      all.map((block) =>
        block.matches(".node-question")
          ? "question"
          : block.matches(".node-answer")
            ? "answer"
            : (block.textContent ?? "").trim(),
      ),
    );

const opacityOf = (locator: Locator) =>
  locator.evaluate((element) => Number(getComputedStyle(element).opacity));

const bridge = (window: Page) => ({
  language: (language: "en" | "zh-CN") =>
    window.evaluate(
      (value) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
          user: { language: value },
        }),
      language,
    ),
});

test("typing in the composer and pressing Enter shows the Question in the assistant as a bubble, and its Answer streams in under it with checked Citations, the Mind as it was", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await atMockupSize(app, window);

  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-new-mind"));
  const title = window.getByTestId("mind-title");
  await title.fill("Reading notes: tides");
  await enterContent(window);
  await window.keyboard.type("Spring tides come twice a month.");
  const editor = window.getByTestId("mind-editor");

  // The composer, at the foot of the Mind: its placeholder, the model and the Search scope.
  const composer = window.getByTestId("composer");
  const input = composer.getByTestId("composer-input");
  await expect(input).toHaveAttribute("placeholder", "Ask, or say what to change…");
  await expect(composer.getByTestId("composer-model")).toHaveText("fake-model");
  await expect(composer.getByTestId("composer-scope-all")).toHaveText("All Documents");
  const ask = composer.getByTestId("composer-ask");
  await expect(ask).toBeDisabled();
  await expect(ask).toHaveAccessibleName("Ask (Enter)");
  // At the foot of the assistant, beside the card (#253), and as wide as the assistant lets it be.
  const inputBox = await composer.boundingBox();
  const assistantBox = await window.getByTestId("assistant").boundingBox();
  const cardBox = await window.getByTestId("card").boundingBox();
  expect(inputBox?.x ?? 0).toBeGreaterThan((cardBox?.x ?? 0) + (cardBox?.width ?? 0));
  expect(inputBox?.x ?? 0).toBeGreaterThan(assistantBox?.x ?? 0);
  expect((inputBox?.x ?? 0) + (inputBox?.width ?? 0)).toBeLessThan(
    (assistantBox?.x ?? 0) + (assistantBox?.width ?? 0),
  );
  expect(inputBox?.width ?? 0).toBeGreaterThan((assistantBox?.width ?? 0) - 32);
  await screenshot(window, "composer-empty-en");

  // Clicked into, then typed in: the Ask button turns primary; Enter asks.
  await pointAndClick(window, input);
  await window.keyboard.type("When do spring tides happen?");
  await expect(ask).toBeEnabled();
  await expect(composer).toHaveCSS("border-color", "rgb(42, 91, 215)");
  await screenshot(window, "composer-typed-en");
  await window.keyboard.press("Enter");

  // A bubble at the right of the assistant, in the compact sizes; its Answer under it.
  const question = window.getByTestId("assistant-question");
  await expect(question).toHaveCount(1);
  await expect(question).toContainText("When do spring tides happen?");
  const bubble = question.locator(".assistant-question-bubble");
  await expect(bubble).toHaveCSS("font-size", "13px");
  const answer = assistantAnswers(window).first();
  await expect(answer).toHaveAttribute("data-status", "streaming");
  // While it is written: the pulsing dot and Stop show; it is read out in a polite live region.
  await expect(answer.getByTestId("answer-writing")).toBeVisible();
  await expect.poll(() => opacityOf(answer.getByTestId("answer-meta"))).toBe(1);
  const live = composer.getByTestId("composer-live");
  await expect(live).toHaveAttribute("aria-live", "polite");
  await expect(live).toHaveText(/Writing the Answer…|Your Documents answer this/);
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(live).toHaveText("Your Documents answer this. That is all. The Answer is done.");
  // The Mind is as it was written.
  await expect.poll(() => order(editor)).toEqual(["Spring tides come twice a month."]);

  // The Citation is checked: found on page 2, its check in the margin beside it.
  const citation = answer.getByTestId("citation");
  await expect(citation).toHaveAttribute("data-check", "found");
  await expect(citation.getByTestId("citation-chip")).toHaveAttribute(
    "aria-label",
    "Citation 1: Tides, p. 2. Quote found on p. 2",
  );
  await expect(answer.getByTestId("margin-check")).toHaveAttribute("data-check", "found");
  // The composer is empty and keeps the focus, for a follow-up.
  await expect(input).toHaveValue("");
  await expect(input).toBeFocused();
  await expect(ask).toBeDisabled();

  // The newest Answer shows its actions at rest: Copy, Add to note, Regenerate, its model and
  // what it searched.
  await window.mouse.move(10, 10);
  await expect.poll(() => opacityOf(answer.getByTestId("answer-meta"))).toBe(1);
  await expect(answer.getByTestId("answer-model")).toHaveText("fake-model");
  await expect(answer.getByTestId("answer-scope")).toHaveText("All Documents");
  await expect(answer.getByTestId("answer-tools")).toContainText("Searched your Documents");
  await expect(answer.getByTestId("answer-regenerate")).toBeVisible();
  await screenshot(window, "composer-answered-en");

  // An earlier one shows them on hover only.
  await window.keyboard.type("And neap tides?");
  await window.keyboard.press("Enter");
  await expect(assistantAnswers(window).nth(1)).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  await window.mouse.move(10, 10);
  await expect.poll(() => opacityOf(answer.getByTestId("answer-meta"))).toBe(0);
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  await expect.poll(() => opacityOf(answer.getByTestId("answer-meta"))).toBe(1);
  await screenshot(window, "answer-hover-en");
  await expect.poll(() => order(editor)).toEqual(["Spring tides come twice a month."]);
  await app.close();
});

test("Stop keeps what streamed, and Regenerate writes the Answer again", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-new-mind"));
  const composer = window.getByTestId("composer");
  await pointAndClick(window, composer.getByTestId("composer-input"));
  // The scripted Answer repeats the Question: a long one keeps it streaming while the
  // pointer slides to Stop, even on a slow runner.
  await window.keyboard.type(
    "What is IncarnaMind, how does it read my Documents, where does it keep them, which models can it use, how does it cite a page, and what happens to my files when I close it?",
  );
  await window.keyboard.press("Enter");

  const answer = assistantAnswers(window).first();
  await expect(answer).toContainText("scripted");
  await expect(answer).toHaveAttribute("data-status", "streaming");
  await pointAndClick(window, answer.getByTestId("answer-stop"));
  await expect(answer).toHaveAttribute("data-status", "stopped");
  const kept = await answer.locator(".assistant-answer-text").textContent();
  expect(kept).toContain("scripted");
  expect(kept).not.toContain("That is all.");
  // Stopped: it says so under its last line, at rest too, with Regenerate.
  await window.mouse.move(10, 10);
  await expect(answer.getByTestId("answer-stopped")).toHaveText("Stopped");
  await expect.poll(() => opacityOf(answer.getByTestId("answer-meta"))).toBe(1);
  await expect(composer.getByTestId("composer-live")).toContainText("The Answer was stopped.");
  // What streamed stays: a moment later it is still all there is.
  await window.waitForTimeout(800);
  await expect(answer.locator(".assistant-answer-text")).toHaveText(kept ?? "");
  await screenshot(window, "answer-stopped-en");

  // Regenerate, under it, writes it again in full, in its place.
  await pointAndClick(window, answer.getByTestId("answer-regenerate"));
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer).toContainText("That is all.");
  await expect(assistantAnswers(window)).toHaveCount(1);
  await expect(window.getByTestId("mind-editor").getByTestId("answer")).toHaveCount(0);
  await app.close();
});

test("⌘J focuses the composer from the note; Shift+Enter breaks the Question's line; Esc folds the assistant and a second Esc goes back to the note, where the cursor was", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-new-mind"));
  await enterContent(window);
  await window.keyboard.type("First thought");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Second thought");
  const editor = window.getByTestId("mind-editor");
  const input = window.getByTestId("composer-input");

  // The cursor at the end of the first line: ⌘J, two lines, Enter.
  const first = editor.locator(":scope > p").first();
  await pointAndClick(window, first);
  // The editor takes the click's place from the browser's selectionchange: wait, as a person would.
  await expect(first).toHaveClass(/has-focus/);
  await window.keyboard.press("End");
  await window.keyboard.press("ControlOrMeta+j");
  await expect(input).toBeFocused();
  await window.keyboard.type("Which thought");
  await window.keyboard.press("Shift+Enter");
  await window.keyboard.type("comes first?");
  await expect(input).toHaveValue("Which thought\ncomes first?");
  await window.keyboard.press("Enter");

  // Its two lines, in one bubble.
  const bubble = window.getByTestId("assistant-question").locator(".assistant-question-bubble");
  await expect.poll(() => bubble.textContent()).toContain("Which thought\ncomes first?");
  const answer = assistantAnswers(window).first();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect.poll(() => order(editor)).toEqual(["First thought", "Second thought"]);

  // Esc folds the assistant, the focus on what shows it again; a second Esc goes back to the
  // note, where the cursor was, to write there.
  await expect(input).toBeFocused();
  await window.keyboard.press("Escape");
  const assistant = window.getByTestId("assistant");
  await expect(assistant).toHaveAttribute("data-folded", "true");
  await window.keyboard.press("Escape");
  await expect(editor).toBeFocused();
  await window.keyboard.type(" noted");
  await expect.poll(() => order(editor)).toEqual(["First thought noted", "Second thought"]);

  // From the title, ⌘J too: the assistant unfolds into its composer.
  await pointAndClick(window, window.getByTestId("mind-title"));
  await window.keyboard.press("ControlOrMeta+j");
  await expect(assistant).not.toHaveAttribute("data-folded");
  await expect(input).toBeFocused();
  await window.keyboard.type("And the last?");
  await window.keyboard.press("Enter");
  await expect(assistantAnswers(window).nth(1)).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  await expect.poll(() => order(editor)).toEqual(["First thought noted", "Second thought"]);
  await app.close();
});

/**
 * An Ollama server that lists models, for the model chip's menu, after
 * `delayMs` as a slow provider does. It answers nothing else.
 */
async function ollamaListing(models: string[], { delayMs = 0 }: { delayMs?: number } = {}) {
  const server: Server = createServer((request, response) => {
    request.resume();
    if (request.url === "/api/tags") {
      setTimeout(() => {
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify({ models: models.map((name) => ({ name, model: name })) }));
      }, delayMs);
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return { server, url: `http://127.0.0.1:${port}` };
}

test("the model chip chooses the model for one Mind, which remembers it after a restart, while new Minds take the default", async () => {
  // Listed a second late, so the menu opens before its models are there, as on a slow machine
  // or with a provider far away: the current one still takes the focus once it is listed.
  const ollama = await ollamaListing(["fake-model", "other-model"], { delayMs: 1000 });
  try {
    const first = await launchApp(dataDir, { fakeChat: true });
    let { window } = first;
    await dismissChatSetup(window);
    await window.evaluate(async (baseUrl) => {
      await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.saveChatProvider({
        kind: "ollama",
        baseUrl,
        modelId: "fake-model",
      });
    }, ollama.url);
    await atMockupSize(first.app, window);

    // A Mind that chooses another model, with the mouse.
    await pointAndClick(window, window.getByTestId("plus-menu"));
    await pointAndClick(window, window.getByTestId("plus-new-mind"));
    await window.getByTestId("mind-title").fill("Other model");
    const chip = window.getByTestId("composer-model");
    await expect(chip).toHaveText("fake-model");
    await pointAndClick(window, chip);
    const menu = window.getByTestId("composer-model-menu");
    await expect(menu).toBeVisible();
    await expect(menu.getByRole("group", { name: "On this computer" })).toBeVisible();
    const options = menu.getByTestId("composer-model-option");
    await expect(options).toHaveText([/fake-model/, /other-model/]);
    await expect(options.first()).toHaveAttribute("aria-checked", "true");
    await expect(options.first()).toBeFocused();
    await expect(menu).toContainText(
      "For this Mind’s next Questions. It’s remembered for the Mind.",
    );
    await expect(menu).toContainText("Default for new Minds: fake-model · Change in Settings");
    await screenshot(window, "composer-model-menu-en");
    await pointAndClick(window, options.nth(1));
    await expect(menu).toBeHidden();
    await expect(chip).toHaveText("other-model");
    await expect(window.getByTestId("composer-input")).toBeFocused();
    await window.keyboard.type("Which model answers?");
    await window.keyboard.press("Enter");
    const answer = assistantAnswers(window).first();
    await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
    await expect(answer.getByTestId("answer-model")).toHaveText("other-model");

    // A new Mind takes the default; by keyboard, the menu opens and closes again.
    await pointAndClick(window, window.getByTestId("plus-menu"));
    await pointAndClick(window, window.getByTestId("plus-new-mind"));
    await window.getByTestId("mind-title").fill("Default model");
    await expect(chip).toHaveText("fake-model");
    await chip.focus();
    await window.keyboard.press("Enter");
    await expect(menu).toBeVisible();
    await expect(options.first()).toBeFocused();
    await window.keyboard.press("ArrowDown");
    await expect(options.nth(1)).toBeFocused();
    await window.keyboard.press("Escape");
    await expect(menu).toBeHidden();
    await expect(chip).toBeFocused();
    await expect(chip).toHaveText("fake-model");
    await first.app.close();

    // After a restart: each Mind still asks with its own.
    const second = await launchApp(dataDir, { fakeChat: true });
    window = second.window;
    const minds = window.getByTestId("mind-list-item");
    await pointAndClick(window, minds.filter({ hasText: "Other model" }));
    await expect(window.getByTestId("composer-model")).toHaveText("other-model");
    await pointAndClick(window, minds.filter({ hasText: "Default model" }));
    await expect(window.getByTestId("composer-model")).toHaveText("fake-model");
    await pointAndClick(window, window.getByTestId("composer-input"));
    await window.keyboard.type("And this one?");
    await window.keyboard.press("Enter");
    const other = assistantAnswers(window).first();
    await expect(other).toHaveAttribute("data-status", "done", { timeout: 15_000 });
    await expect(other.getByTestId("answer-model")).toHaveText("fake-model");

    // In Chinese.
    await bridge(window).language("zh-CN");
    await expect(window.getByTestId("composer-input")).toHaveAttribute(
      "placeholder",
      "提问，或说明要修改的内容…",
    );
    await pointAndClick(window, window.getByTestId("composer-model"));
    const menuZh = window.getByTestId("composer-model-menu");
    await expect(menuZh.getByRole("group", { name: "在本机运行" })).toBeVisible();
    await expect(menuZh).toContainText("新 Mind 的默认模型：fake-model · 在设置中更改");
    await screenshot(window, "composer-model-menu-zh");
    await window.keyboard.press("Escape");
    await screenshot(window, "composer-zh");
    await second.app.close();
  } finally {
    ollama.server.close();
  }
});

/**
 * A Mind written before the composer, in its stored shape: a Note, a Question
 * with a model, a Search scope and a forced Skill, and its finished Answer.
 */
function oldMind(): Uint8Array {
  const doc = new Y.Doc();
  const blocks = doc.getXmlFragment("blocks");
  const paragraph = (text: string) => {
    const element = new Y.XmlElement("paragraph");
    element.setAttribute("id", crypto.randomUUID());
    element.insert(0, [new Y.XmlText(text)]);
    return element;
  };
  const question = new Y.XmlElement("question");
  question.setAttribute("id", "question-1");
  question.setAttribute("providerId", "provider-gone");
  question.setAttribute("modelId", "an-older-model");
  question.setAttribute("scopeTagIds", ["tag-gone"] as unknown as string);
  question.insert(0, [new Y.XmlText("What did the tide tables say?")]);
  const answer = new Y.XmlElement("answer");
  answer.setAttribute("id", "answer-1");
  answer.setAttribute("questionId", "question-1");
  answer.setAttribute("modelId", "an-older-model");
  answer.setAttribute("status", "done");
  answer.insert(0, [paragraph("They said high water comes later each day.")]);
  blocks.insert(0, [paragraph("Written before the composer."), question, answer, paragraph("")]);
  return Y.encodeStateAsUpdate(doc);
}

test("a Mind written before opens with its Questions and Answers in the new style, and its stored Blocks unchanged", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  const mindId = await window.evaluate(async (update) => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const mind = await core.createMind({ title: "From before" });
    await core.applyMindUpdate(mind.id, new Uint8Array(update));
    await core.closeMind(mind.id);
    return mind.id;
  }, Array.from(oldMind()));
  const stored = () =>
    window.evaluate(async (id) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      return Array.from((await core.openMind(id)).state);
    }, mindId);
  const beforeState = new Uint8Array(await stored());

  await pointAndClick(
    window,
    window.getByTestId("mind-list-item").filter({ hasText: "From before" }),
  );
  const editor = window.getByTestId("mind-editor");
  const question = editor.getByTestId("question");
  await expect(question.locator(".question-text")).toHaveText("What did the tide tables say?");
  await expect(question.locator(".question-text")).toHaveCSS("font-size", "14px");
  await expect(question.getByTestId("question-model")).toHaveText("an-older-model");
  const chip = question.getByTestId("scope-chip");
  await expect(chip).toHaveAttribute("data-deleted", "true");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done");
  await expect(answer).toContainText("They said high water comes later each day.");
  await expect(answer.getByTestId("answer-model")).toHaveText("an-older-model");
  await window.mouse.move(10, 10);
  await expect.poll(() => opacityOf(answer.getByTestId("answer-meta"))).toBe(0);
  await screenshot(window, "old-mind-en");

  // Its Blocks are as they were stored: opening and showing them wrote nothing.
  const afterState = new Uint8Array(await stored());
  const read = (state: Uint8Array) => {
    const doc = new Y.Doc();
    Y.applyUpdate(doc, state);
    return { vector: Y.encodeStateVector(doc), blocks: doc.getXmlFragment("blocks").toString() };
  };
  expect(read(afterState)).toEqual(read(beforeState));
  await app.close();
});

test("a follow-up asked straight away goes into the same conversation, without taking the focus from the composer or touching the Mind", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-new-mind"));
  const input = window.getByTestId("composer-input");
  await pointAndClick(window, input);
  await window.keyboard.type("A first Question?");
  await window.keyboard.press("Enter");
  const editor = window.getByTestId("mind-editor");
  await expect(assistantAnswers(window).first()).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  await window.keyboard.type("A follow-up?");
  await window.keyboard.press("Enter");
  await expect(assistantAnswers(window).nth(1)).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  await expect(window.getByTestId("assistant-question")).toHaveCount(2);
  await expect(window.getByTestId("assistant-title")).toContainText("A first Question");
  const blocks = await editor.evaluate(
    (dom) => (dom as unknown as { editor: Editor }).editor.state.doc.childCount,
  );
  expect(blocks).toBe(1);
  await expect(editor.getByTestId("question")).toHaveCount(0);
  await expect(input).toBeFocused();
  await app.close();
});

test("in Chinese: the composer with a Search scope and a Skill chosen from its pickers, the Answer's meta line, and a narrow window", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await bridge(window).language("zh-CN");
  await atMockupSize(app, window);

  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-new-mind"));
  await window.getByTestId("mind-title").fill("潮汐笔记");
  await enterContent(window);
  await window.keyboard.type("大潮每月出现两次。");
  const composer = window.getByTestId("composer");
  const input = composer.getByTestId("composer-input");
  await expect(input).toHaveAttribute("placeholder", "提问，或说明要修改的内容…");
  await expect(input).toHaveAccessibleName(/提问，或说明要修改的内容（(⌘J|Ctrl\+J)）/);
  await expect(composer.getByTestId("composer-ask")).toHaveAccessibleName("提问（Enter）");
  await expect(composer.getByTestId("composer-scope-all")).toHaveText("全部文档");

  // "@" chooses what it searches; "/" a Skill: each a chip in the composer.
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("@");
  const picker = window.getByTestId("scope-picker");
  await expect(picker.getByRole("group", { name: "文档" })).toBeVisible();
  await screenshot(window, "composer-scope-picker-zh");
  await window.keyboard.type("Tid");
  await window.keyboard.press("Enter");
  await expect(composer.getByTestId("scope-chip")).toHaveText("Tides");
  await window.keyboard.type("/");
  const skills = window.getByTestId("composer-skill-picker");
  await expect(skills.getByTestId("slash-item-skill-literature-review")).toBeVisible();
  await screenshot(window, "composer-skill-picker-zh");
  await window.keyboard.type("lit");
  await window.keyboard.press("Enter");
  await expect(composer.getByTestId("composer-skill")).toHaveText("literature-review");
  await window.keyboard.type("大潮什么时候出现？");
  await screenshot(window, "composer-chips-zh");

  // Asked: the Question carries both, its Answer says what it searched, in Chinese.
  await window.keyboard.press("Enter");
  const answer = assistantAnswers(window).first();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(composer.getByTestId("composer-live")).toContainText("回答已生成。");
  const question = window.getByTestId("assistant-question");
  await expect(question.getByTestId("assistant-question-skill")).toHaveText("literature-review");
  await expect(answer.getByTestId("answer-scope")).toHaveText("“Tides”");
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  await expect(answer.getByTestId("answer-regenerate")).toHaveText("重新生成");
  await expect(answer.getByTestId("answer-add")).toHaveText("添加到笔记");
  await screenshot(window, "answer-hover-zh");

  // A narrow window: the assistant goes to the bottom (#253), and the composer spans it there,
  // up to the Mind's column.
  await setWindowSize(app, window, 960, 640);
  const assistant = window.getByTestId("assistant");
  await expect(assistant).toHaveAttribute("data-dock", "bottom");
  const card = await window.getByTestId("card").boundingBox();
  const below = await composer.boundingBox();
  expect(below?.y ?? 0).toBeGreaterThan((card?.y ?? 0) + (card?.height ?? 0));
  expect(below?.width ?? 0).toBeGreaterThan((card?.width ?? 0) - 40);
  await screenshot(window, "composer-narrow-zh");
  await app.close();
});

test("after the consent dialog closes, the caret is back in the composer, and both Answers are in the assistant, the Mind as it was", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  // A model on a server elsewhere, so the first Question asks to send first. The scripted model answers.
  await window.evaluate(async () => {
    await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.saveChatProvider({
      kind: "openai-compatible",
      baseUrl: "https://llm.example.com/v1",
      modelId: "example-model",
    });
  });
  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-new-mind"));
  await enterContent(window);
  await window.keyboard.type("Alpha");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Omega");
  const editor = window.getByTestId("mind-editor");
  // The cursor in the note: the Mind stays as it is, whatever is asked.
  const first = editor.locator(":scope > p").first();
  await pointAndClick(window, first);
  await expect(first).toHaveClass(/has-focus/);
  await window.keyboard.press("End");
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("A first Question?");
  await window.keyboard.press("Enter");

  const consent = window.getByTestId("consent-dialog");
  await expect(consent).toBeVisible();
  await consent.getByTestId("consent-allow").click();
  await expect(consent).toBeHidden();

  // Typed straight away, with no click: it reaches the composer.
  const input = window.getByTestId("composer-input");
  await expect(input).toBeFocused();
  await window.keyboard.type("A follow-up?");
  await expect(input).toHaveValue("A follow-up?");
  await window.keyboard.press("Enter");
  await expect(assistantAnswers(window).nth(1)).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  await expect(window.getByTestId("assistant-question")).toHaveCount(2);
  await expect.poll(() => order(editor)).toEqual(["Alpha", "Omega"]);
  await app.close();
});
