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
 * When changes wait in a Mind that the assistant can't revise, it changes
 * nothing and says so: the step names why, and a line under the Answer says
 * how many wait, with Review, which brings the Mind up at its first change.
 * The scripted model (src/main/fakeChat) calls edit_mind for a request to
 * shorten, and its changes wait in the Mind, so a second request to shorten
 * meets them.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

async function writeMind(window: Page, title: string, note: string): Promise<void> {
  await newMind(window);
  await expect(window.getByTestId("mind-title")).toBeFocused();
  await window.keyboard.type(title);
  await enterContent(window);
  await window.keyboard.type(note);
  await expect(window.getByTestId("mind-editor").locator("p")).toHaveText([note]);
}

test("over changes it can't revise, the assistant changes nothing, says why, and Review shows them; in English and Chinese; the line goes once none wait", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setLanguage(window, "en");
  await setWindowSize(app, window, 1100, 800);
  await writeMind(window, "Q3 review", "The quarter was very good, and sales were really strong.");
  await askInAssistant(window, "Shorten the summary");
  const bar = window.getByTestId("proposals-bar");
  await expect(bar.getByTestId("proposals-count")).toHaveText("2 changes from the assistant");

  const refused = await askInAssistant(window, "Shorten the summary");

  const line = refused.getByTestId("answer-waiting-changes");
  await expect(line).toHaveText(
    "2 changes are waiting in “Q3 review”. Accept or reject them first? · Review",
  );
  await expect(bar.getByTestId("proposals-count")).toHaveText("2 changes from the assistant");
  const timeline = await openSteps(window, refused);
  await expect(timeline.getByTestId("answer-mind-call")).toHaveText(
    /Couldn't change “Q3 review”: 2 changes are waiting there/,
  );
  // The line sits under the Answer's text, inside the assistant.
  const text = await refused.getByTestId("assistant-answer-text").boundingBox();
  const shown = await line.boundingBox();
  expect(shown && text && shown.y >= text.y + text.height).toBe(true);

  await pointAndClick(window, line.getByTestId("answer-waiting-review"));
  await expect(window.getByTestId("mind-editor")).toBeFocused();

  await setLanguage(window, "zh-CN");
  await expect(line).toHaveText("“Q3 review”中有 2 处修订待确认。先接受或拒绝它们？· 审阅");
  await expect(timeline.getByTestId("answer-mind-call")).toHaveText(
    /无法修改“Q3 review”：其中有 2 处修订待确认/,
  );

  await pointAndClick(window, bar.getByTestId("proposals-accept-all"));
  await expect(bar).toHaveCount(0);
  await expect(line).toHaveCount(0);
});
