import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import {
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  enterContent,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/*
 * What the User asks the assistant to write or change applies at once (#880,
 * D224): the owner's own two requests, "write a summary page", then "make it
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
  await expect(line.getByTestId("applied-count")).toHaveText("Changed 2 places");
  await expect(second.getByTestId("mind-changes-title")).toHaveText(
    "Changed 2 places in “Summary”",
  );
  await screenshot(window, "asked-edit-en");

  // One step of Undo for each request: first the longer text goes, then the summary.
  await editor.focus();
  await window.keyboard.press("ControlOrMeta+z");
  await expect(paragraphs).toHaveText([SPRING, NEAP]);
  await window.keyboard.press("ControlOrMeta+z");
  await expect(editor).not.toContainText("Spring tides");
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
  await expect(window.getByTestId("mind-title")).toBeFocused();
  await window.keyboard.type("Harbour notes");
  await enterContent(window);
  await window.keyboard.type(NOTES[0] ?? "");
  await window.keyboard.press("Enter");
  await window.keyboard.type(NOTES[1] ?? "");
  const editor = window.getByTestId("mind-editor");
  const paragraphs = editor.locator(":scope > p");
  await expect(paragraphs).toHaveText(NOTES);

  await askInAssistant(window, "Rewrite the intro to be clearer");
  const line = editor.getByTestId("applied-line");
  await expect(line.getByTestId("applied-count")).toHaveText(
    "This would rewrite most of your text, so it's waiting for you",
  );
  await expect(editor.locator("del.proposal-delete").first()).toBeVisible();
  await screenshot(window, "asked-held-en");
  await setLanguage(window, "zh-CN");
  await expect(line.getByTestId("applied-count")).toHaveText(
    "这会改写你自己写的大部分内容，所以先等你确认",
  );
  await setLanguage(window, "en");

  await pointAndClick(window, line.getByTestId("applied-discard"));
  await expect(paragraphs).toHaveText(NOTES);

  await askInAssistant(window, "Rewrite the intro to be clearer");
  await pointAndClick(window, line.getByTestId("applied-apply"));
  await expect(paragraphs).toHaveText([
    "Gulls wheel over the quay at dawn.",
    "Boats rest on the mud at low water.",
  ]);
  await expect(line.getByTestId("applied-count")).toHaveText(/^Changed \d+ places?$/);
  await editor.focus();
  await window.keyboard.press("ControlOrMeta+z");
  await expect(paragraphs).toHaveText(NOTES);
});
