import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import {
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  enterContent,
  launchApp,
  newMind,
  openSteps,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/**
 * The assistant changes Minds and makes them (#202): what the User types
 * decides. The scripted model (src/main/fakeChatModel.ts) calls edit_mind for
 * a request to shorten, which drops "very" and "really" (非常), and
 * create_mind for a new note; anything else is answered. Each change waits in
 * its Mind as changes to accept or reject. Set INCARNAMIND_SCREENSHOTS to a
 * folder to also save screenshots.
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

async function start(language: "en" | "zh-CN") {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setLanguage(window, language);
  await setWindowSize(app, window, 1440, 900);
  return { app, window };
}

/** A new Mind, its title and its Notes typed as a person types them. */
async function writeMind(window: Page, title: string, notes: string[]): Promise<void> {
  await newMind(window);
  await expect(window.getByTestId("mind-title")).toBeFocused();
  await window.keyboard.type(title);
  await enterContent(window);
  for (const [index, line] of notes.entries()) {
    if (index > 0) await window.keyboard.press("Enter");
    await window.keyboard.type(line);
  }
  await expect(window.getByTestId("mind-editor").locator("p")).toHaveText(notes);
}

test("a Question is answered and changes nothing; a request to shorten comes back as changes in the Mind, which the steps name", async () => {
  const { window } = await start("en");
  const note = "The quarter was very good, and sales were really strong.";
  await writeMind(window, "Q3 review", [note]);
  const editor = window.getByTestId("mind-editor");

  const answered = await askInAssistant(window, "How was the quarter?");
  await expect(answered.getByTestId("assistant-answer-text")).toContainText("scripted");
  await expect(window.getByTestId("proposals-bar")).toHaveCount(0);
  await expect(editor.locator("p")).toHaveText([note]);

  const changed = await askInAssistant(window, "Shorten the summary");
  const bar = window.getByTestId("proposals-bar");
  await expect(bar.getByTestId("proposals-count")).toHaveText("2 changes from the assistant");
  await expect(changed.getByTestId("assistant-answer-text")).toHaveText(
    "Proposed 2 changes in “Q3 review” (from 10 to 8 words).",
  );
  const timeline = await openSteps(window, changed);
  await expect(timeline.getByTestId("answer-mind-call")).toHaveText(
    /Edited “Q3 review” · 2 changes/,
  );
  await screenshot(window, "assistant-changed-mind");

  await pointAndClick(window, bar.getByTestId("proposals-accept-all"));
  await expect(editor.locator("p")).toHaveText(["The quarter was good, and sales were strong."]);
});

test("a request for a new note makes a Mind beside the one in focus, its text as changes; in Chinese", async () => {
  const { window } = await start("zh-CN");
  await writeMind(window, "第三季度", ["销售非常好。"]);

  const answer = await askInAssistant(window, "新建一篇叫“田野计划”的笔记");

  const tab = window.getByTestId("mind-tab").filter({ hasText: "田野计划" });
  await expect(tab).toBeVisible();
  await expect(window.getByTestId("mind-tab").filter({ hasText: "第三季度" })).toBeVisible();
  await expect(window.getByTestId("mind-title")).toHaveValue("田野计划");
  await expect(window.getByTestId("proposals-count")).toHaveText("来自助手的 2 处修订");
  const timeline = await openSteps(window, answer);
  await expect(timeline.getByTestId("answer-mind-call")).toHaveText(/新建了“田野计划” · 2 处修订/);
  await screenshot(window, "assistant-made-mind-zh");
});
