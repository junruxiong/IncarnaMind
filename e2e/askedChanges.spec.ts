import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  askInAssistant,
  assistantAnswers,
  createDataFolder,
  dismissChatSetup,
  enterContent,
  frontMindName,
  launchApp,
  nameMind,
  newMind,
  openSteps,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/*
 * What the User asks the assistant to write or change applies at once
 * (#880): the owner's own two requests, "write a summary page", then "make it
 * longer". The scripted model (src/main/fakeChat/minds.ts) makes a page with
 * a short summary, then has Edit with AI add a sentence to each paragraph;
 * asked to rewrite the User's notes "to be clearer", it writes two new
 * paragraphs in place of them, which wait.
 * Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

const SPRING = "Spring tides come at new and full moon, when the Sun and the Moon pull together.";
const NEAP = "Neap tides come in between, at the quarter moons.";

test("“write a summary page” makes the page with the summary applied, “make it longer” changes it applied too, and each request is one step of Undo", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 1440, 900);
  await newMind(window);

  const first = await askInAssistant(window, "write a summary page");
  await expect(window.getByTestId("mind-tab").filter({ hasText: "Summary" })).toBeVisible();
  const editor = window.getByTestId("mind-editor");
  const paragraphs = editor.locator(":scope > p");
  await expect(paragraphs).toHaveText([SPRING, NEAP]);
  await expect(editor.locator("ins.proposal-insert, [data-proposal]")).toHaveCount(0);
  const line = editor.getByTestId("applied-line");
  await expect(line.getByTestId("applied-count")).toHaveText("Wrote 2 paragraphs");
  await expect(first.getByTestId("mind-changes-title")).toHaveText(
    "Created “Summary” and wrote 2 paragraphs",
  );
  await expect(first.getByTestId("mind-changes-undo")).toBeVisible();
  await screenshot(window, "asked-write-en");

  const second = await askInAssistant(window, "make it longer");
  await expect(paragraphs).toHaveText([`${SPRING} It says more.`, `${NEAP} It says more.`]);
  await expect(editor.locator("ins.proposal-insert, del.proposal-delete")).toHaveCount(0);
  await expect(line.getByTestId("applied-count")).toHaveText("Changed 2 paragraphs");
  await expect(second.getByTestId("mind-changes-title")).toHaveText(
    "Changed 2 paragraphs in “Summary”",
  );
  await screenshot(window, "asked-edit-en");
  await setLanguage(window, "zh-CN");
  await expect(line.getByTestId("applied-count")).toHaveText("修改了 2 个段落");
  await expect(second.getByTestId("mind-changes-title")).toHaveText("在“Summary”中已修改 2 个段落");
  await setLanguage(window, "en");

  // Review changes from the Answer shows what was added, the text still applied.
  await pointAndClick(window, second.getByTestId("mind-changes-review"));
  await expect(line.getByTestId("applied-count")).toHaveText(
    "2 paragraphs changed by the assistant",
  );
  await expect(editor.locator(".applied-new")).toHaveText([" It says more.", " It says more."]);
  await pointAndClick(window, line.getByTestId("applied-done"));
  await expect(paragraphs).toHaveText([`${SPRING} It says more.`, `${NEAP} It says more.`]);

  // Review on the first card shows the summary that Answer wrote, not the later change.
  await pointAndClick(window, assistantAnswers(window).first().getByTestId("mind-changes-review"));
  await expect(line.getByTestId("applied-count")).toHaveText(/changed by the assistant$/);
  await expect(editor.locator(".applied-new").filter({ hasText: "Spring tides" })).toHaveCount(1);
  await expect(editor.locator(".applied-new").filter({ hasText: /^ It says more\.$/ })).toHaveCount(
    0,
  );
  await expect(line.getByTestId("applied-undo-all")).toHaveAttribute("aria-disabled", "true");
  await pointAndClick(window, line.getByTestId("applied-undo-all"));
  await expect(paragraphs).toHaveText([`${SPRING} It says more.`, `${NEAP} It says more.`]);
  await setLanguage(window, "zh-CN");
  await expect(line.getByTestId("applied-count")).toHaveText("助手修改的 3 个段落");
  await expect(line.getByTestId("applied-undo-all")).toHaveAttribute("title", /Summary.*查看改动/);
  await setLanguage(window, "en");
  await screenshot(window, "asked-review-earlier-en");
  await pointAndClick(window, second.getByTestId("mind-changes-review"));
  await expect(line.getByTestId("applied-undo-all")).not.toHaveAttribute("aria-disabled");
  await expect(editor.locator(".applied-new")).toHaveText([" It says more.", " It says more."]);
  await pointAndClick(window, line.getByTestId("applied-done"));

  const longer = [`${SPRING} It says more.`, `${NEAP} It says more.`];
  // The Answer `askInAssistant` gave is the last one, whichever that is now.
  const summary = assistantAnswers(window).first();
  const firstUndo = summary.getByTestId("mind-changes-undo");
  const secondUndo = second.getByTestId("mind-changes-undo");
  const changedSince = "“Summary” changed since. Restore each change in Review changes.";

  // Each card undoes its own request: the summary's can't while the longer text is on top of it.
  await expect(firstUndo).toHaveAttribute("aria-disabled", "true");
  await expect(firstUndo).toHaveAttribute("title", changedSince);
  await expect(secondUndo).not.toHaveAttribute("aria-disabled");
  await screenshot(window, "asked-undo-since-en");
  await pointAndClick(window, firstUndo);
  await expect(paragraphs).toHaveText(longer);
  await pointAndClick(window, secondUndo);
  await expect(paragraphs).toHaveText([SPRING, NEAP]);
  await expect(second.getByTestId("mind-changes-undone")).toHaveText("Undone");
  await expect(firstUndo).not.toHaveAttribute("aria-disabled");

  // The review's Undo follows the step created by Redo in the Mind.
  await editor.focus();
  await window.keyboard.press("ControlOrMeta+Shift+z");
  await expect(paragraphs).toHaveText(longer);
  await expect(secondUndo).not.toHaveAttribute("aria-disabled");
  await expect(firstUndo).toHaveAttribute("aria-disabled", "true");
  await pointAndClick(window, second.getByTestId("mind-changes-review"));
  await pointAndClick(window, line.getByTestId("applied-undo-all"));
  await expect(paragraphs).toHaveText([SPRING, NEAP]);
  await expect(second.getByTestId("mind-changes-undone")).toBeVisible();

  // Typed after the summary, its card says why it can't undo it, until the typing is taken back.
  await paragraphs.last().click();
  await window.keyboard.press("End");
  await window.keyboard.type(" Typed.");
  await expect(paragraphs.last()).toHaveText(`${NEAP} Typed.`);
  await expect(firstUndo).toHaveAttribute("aria-disabled", "true");
  await expect(async () => {
    if ((await paragraphs.last().textContent())?.includes("Typed")) {
      await window.keyboard.press("ControlOrMeta+z");
    }
    await expect(paragraphs).toHaveText([SPRING, NEAP]);
  }).toPass();
  await expect(firstUndo).not.toHaveAttribute("aria-disabled");
  await pointAndClick(window, summary.getByTestId("mind-changes-review"));
  await expect(line.getByTestId("applied-undo-all")).not.toHaveAttribute("aria-disabled");
  await pointAndClick(window, line.getByTestId("applied-undo-all"));
  await expect(editor).not.toContainText("Spring tides");
  await expect(summary.getByTestId("mind-changes-undone")).toHaveText("Undone");
});

test("an Answer written again replaces what it applied with what it writes now, and its card can undo it", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 1440, 900);
  await newMind(window);
  await askInAssistant(window, "write a summary page");
  const editor = window.getByTestId("mind-editor");
  const paragraphs = editor.locator(":scope > p");
  await expect(paragraphs).toHaveText([SPRING, NEAP]);

  const answer = await askInAssistant(window, "make it longer");
  await expect(paragraphs).toHaveText([`${SPRING} It says more.`, `${NEAP} It says more.`]);
  const firstWrite = await changeRequestOf(window);
  await openSteps(window, answer);
  await pointAndClick(window, answer.getByTestId("answer-regenerate"));
  await expect
    .poll(async () => {
      const now = await changeRequestOf(window);
      return now !== null && now !== firstWrite;
    })
    .toBe(true);
  await expect(answer.getByTestId("mind-changes-undo")).not.toHaveAttribute("aria-disabled");
  await expect(paragraphs).toHaveText([`${SPRING} It says more.`, `${NEAP} It says more.`]);
  await expect(editor.locator("ins.proposal-insert, del.proposal-delete")).toHaveCount(0);
  await pointAndClick(window, answer.getByTestId("mind-changes-undo"));
  await expect(paragraphs).toHaveText([SPRING, NEAP]);
  await expect(answer.getByTestId("mind-changes-undone")).toHaveText("Undone");

  // Written again after the Mind changed, the new change comes on top, and the card says so.
  await editor.focus();
  await window.keyboard.press("ControlOrMeta+Shift+z");
  await expect(paragraphs).toHaveText([`${SPRING} It says more.`, `${NEAP} It says more.`]);
  await paragraphs.last().click();
  await window.keyboard.press("End");
  await window.keyboard.type(" Typed.");
  await expect(paragraphs.last()).toHaveText(`${NEAP} It says more. Typed.`);
  const secondWrite = await changeRequestOf(window);
  await openSteps(window, answer);
  await pointAndClick(window, answer.getByTestId("answer-regenerate"));
  await expect
    .poll(async () => {
      const now = await changeRequestOf(window);
      return now !== null && now !== secondWrite;
    })
    .toBe(true);
  await expect(answer.getByTestId("mind-changes-on-top")).toHaveText(
    "· on top of the earlier version, which stays",
  );
  await expect(paragraphs.first()).toHaveText(`${SPRING} It says more. It says more.`);
  await screenshot(window, "asked-on-top-en");
});

const NOTES = [
  "Tides rise twice a day along this coast.",
  "The harbour office posts the times each week.",
];

test("a rewrite that would take out most of the User's own text waits, in English and Chinese: Apply applies it as one step of Undo, and Discard leaves the text", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 1100, 860);
  await newMind(window);
  await nameMind(window, "Harbour notes");
  await enterContent(window);
  await window.keyboard.type(NOTES[0] ?? "");
  await window.keyboard.press("Enter");
  await window.keyboard.type(NOTES[1] ?? "");
  const editor = window.getByTestId("mind-editor");
  const paragraphs = editor.locator(":scope > p");
  await expect(paragraphs).toHaveText(NOTES);

  const answer = await askInAssistant(window, "Rewrite the intro to be clearer");
  await expect(answer.getByTestId("mind-changes-left")).toHaveText("· 2 waiting");
  const line = editor.getByTestId("applied-line");
  await expect(line.getByTestId("applied-count")).toHaveText(
    "This would rewrite most of your text, so it's waiting for you",
  );
  // Each paragraph written anew reads whole, its old text one line above it, struck on a click (#1028).
  const was = editor.getByTestId("change-was-line").first();
  await expect(was).toBeVisible();
  await expect(editor.locator("del.proposal-delete").first()).toBeHidden();
  await pointAndClick(window, was.getByTestId("change-was"));
  await expect(editor.locator("del.proposal-delete").first()).toBeVisible();
  await screenshot(window, "asked-held-en");
  await setLanguage(window, "zh-CN");
  await expect(line.getByTestId("applied-count")).toHaveText(
    "这会改写你自己写的大部分内容，所以先等你确认",
  );
  await setLanguage(window, "en");

  await pointAndClick(window, line.getByTestId("applied-discard"));
  await expect(paragraphs).toHaveText(NOTES);
  await expect(answer.getByTestId("mind-changes-left")).toHaveText("· none waiting");
  // The text keeps the focus, a frame later: ⌘J before that would lose the composer's.
  await expect(editor).toBeFocused();

  await askInAssistant(window, "Rewrite the intro to be clearer");
  await pointAndClick(window, line.getByTestId("applied-apply"));
  await expect(paragraphs).toHaveText([
    "Gulls wheel over the quay at dawn.",
    "Boats rest on the mud at low water.",
  ]);
  await expect(line.getByTestId("applied-count")).toHaveText(/^Changed \d+ paragraphs?$/);
  await editor.focus();
  await window.keyboard.press("ControlOrMeta+z");
  await expect(paragraphs).toHaveText(NOTES);
});

/** The request the newest Answer's change made, once that Answer is done; null before. */
function changeRequestOf(window: Page): Promise<string | null> {
  return window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const [conversation] = await core.listConversations();
    if (!conversation) return null;
    const { messages } = await core.getConversation(conversation.id);
    const answer = messages.filter((message) => message.kind === "answer").at(-1);
    if (answer?.kind !== "answer" || answer.status !== "done") return null;
    return answer.toolCalls.find((call) => call.mindChange)?.mindChange?.request ?? null;
  });
}

/** A new Mind, named, and its Notes typed as a person types them. */
async function writeMind(window: Page, title: string, notes: readonly string[]): Promise<void> {
  await newMind(window);
  await nameMind(window, title);
  await enterContent(window);
  for (const [index, note] of notes.entries()) {
    if (index > 0) await window.keyboard.press("Enter");
    await window.keyboard.type(note);
  }
}

test("a request to write applies what it wrote, and a rewrite of the User's text beside it waits, counted apart", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 1440, 900);
  await writeMind(window, "Harbour notes", NOTES);
  const editor = window.getByTestId("mind-editor");

  const answer = await askInAssistant(window, "Add a section on the risks");
  await expect(editor.getByTestId("applied-count")).toHaveText("Wrote 1 paragraph");
  await expect(answer.getByTestId("mind-changes-title")).toHaveText(
    "Wrote 1 paragraph in “Harbour notes”",
  );
  await expect(answer.getByTestId("mind-changes-left")).toHaveText("· 2 waiting");
  await expect(editor.locator("del.proposal-delete, ins.proposal-insert").first()).toBeVisible();
  await expect(editor).toContainText("The supplier is late.");
  await setLanguage(window, "zh-CN");
  await expect(answer.getByTestId("mind-changes-left")).toHaveText("· 2 个段落待确认");
  await setLanguage(window, "en");
});

test("asked to write in a Mind not in view, the assistant opens it in a tab beside the one in view and writes there, applied", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 1440, 900);
  await writeMind(window, "Project plan", ["Hire a designer."]);
  await writeMind(window, "Q3 review", ["The quarter was good."]);
  const tabs = window.getByTestId("mind-tab");
  await expect(tabs).toHaveCount(2);

  const answer = await askInAssistant(window, "Add a note about the meeting to Project plan");
  await expect(answer.getByTestId("mind-changes-title")).toHaveText(
    "Wrote 1 paragraph in “Project plan”",
  );
  await expect(tabs.filter({ hasText: "Q3 review" })).toHaveCount(1);
  await expect(tabs.filter({ hasText: "Project plan" })).toHaveCount(1);
  await expect(frontMindName(window)).toHaveText("Project plan");
  const editor = window.getByTestId("mind-editor");
  await expect(editor.locator(":scope > p")).toHaveText([
    "Hire a designer.",
    "a note about the meeting",
  ]);
  await expect(editor.getByTestId("applied-count")).toHaveText("Wrote 1 paragraph");
});
