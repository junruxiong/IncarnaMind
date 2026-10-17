import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import {
  askInAssistant,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  dragBy,
  enterContent,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  widthOf,
} from "./app";

/**
 * A model that can't call Tools, asked to write into a Mind or change one,
 * answers in the assistant and says it can't write there itself, with Add to
 * Mind, which puts the Answer in the Mind in one step with Undo. The scripted
 * model "tiny-no-tools" refuses Tools, as a small server model does. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots there.
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

async function start(
  language: "en" | "zh-CN",
  title: string,
  text: string,
  size = { width: 1440, height: 900 },
) {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await connectLocalChatModel(window, "tiny-no-tools");
  await setLanguage(window, language);
  await setWindowSize(app, window, size.width, size.height);
  await newMind(window);
  await expect(window.getByTestId("mind-title")).toBeFocused();
  await window.keyboard.type(title);
  await enterContent(window);
  await window.keyboard.type(text);
  await expect(window.getByTestId("mind-editor").locator("p")).toHaveText([text]);
  return { app, window, editor: window.getByTestId("mind-editor") };
}

async function expectFits(window: Page, answer: Locator) {
  const line = answer.getByTestId("answer-no-mind-tools");
  const [lineBox, answerBox, buttonBox, textBox, actionsBox] = await Promise.all([
    line.boundingBox(),
    answer.boundingBox(),
    line.getByTestId("answer-no-mind-tools-add").boundingBox(),
    answer.getByTestId("assistant-answer-text").boundingBox(),
    answer.getByTestId("answer-actions").boundingBox(),
  ]);
  if (!lineBox || !answerBox || !buttonBox || !textBox || !actionsBox) {
    throw new Error("The line isn't on screen.");
  }
  expect(lineBox.y).toBeGreaterThanOrEqual(textBox.y + textBox.height - 0.5);
  expect(lineBox.y + lineBox.height).toBeLessThanOrEqual(actionsBox.y + 0.5);
  expect(lineBox.x).toBeGreaterThanOrEqual(answerBox.x - 0.5);
  expect(lineBox.x + lineBox.width).toBeLessThanOrEqual(answerBox.x + answerBox.width + 0.5);
  expect(buttonBox.x).toBeGreaterThanOrEqual(lineBox.x - 0.5);
  expect(buttonBox.x + buttonBox.width).toBeLessThanOrEqual(lineBox.x + lineBox.width + 0.5);
  expect(await line.evaluate((each) => each.scrollWidth <= each.clientWidth + 1)).toBe(true);
  expect(
    await window
      .getByTestId("assistant-log")
      .evaluate(
        (log) => (log.parentElement?.scrollWidth ?? 0) <= (log.parentElement?.clientWidth ?? 0),
      ),
  ).toBe(true);
}

async function narrowestAssistant(window: Page) {
  const assistant = window.getByTestId("assistant");
  await dragBy(window, window.getByTestId("assistant-resize"), 1200);
  await expect.poll(() => widthOf(assistant)).toBe(280);
}

test("a request to write is answered in the assistant with the line; Add to Mind puts the Answer in the Mind in one step, and Undo takes it back", async () => {
  const { app, window, editor } = await start("en", "Q3 review", "The quarter was good.");

  const question = await askInAssistant(window, "How was the quarter?");
  await expect(question.getByTestId("answer-no-mind-tools")).toHaveCount(0);

  const answer = await askInAssistant(window, "write a summary page");
  await expect(answer.getByTestId("assistant-answer-text")).toContainText(
    "Answer to: write a summary page",
  );
  const line = answer.getByTestId("answer-no-mind-tools");
  await expect(line).toHaveText("This model can't write into your Mind itself. · Add to Mind");
  await expect(editor.locator("p")).toHaveText(["The quarter was good."]);
  await screenshot(window, "no-mind-tools-line");

  await narrowestAssistant(window);
  await expectFits(window, answer);
  await screenshot(window, "no-mind-tools-line-narrow");

  await pointAndClick(window, line.getByTestId("answer-no-mind-tools-add"));
  const added = answer.getByTestId("answer-added");
  await expect(added).toContainText("Added to “Q3 review”");
  await expect(line).toHaveCount(0);
  await expect(editor).toContainText("Answer to: write a summary page");

  await pointAndClick(window, answer.getByTestId("answer-added-undo"));
  await expect(editor).not.toContainText("Answer to: write a summary page");
  await expect(editor.locator("p")).toHaveText(["The quarter was good."]);
  await expect(line).toBeVisible();
  await app.close();
});

test("in Chinese, “把它改长一点” after a request to write shows the line again, and the line fits the narrowest assistant", async () => {
  const { app, window, editor } = await start("zh-CN", "第三季度", "销售很好。");

  const first = await askInAssistant(window, "写一个总结页面");
  const line = first.getByTestId("answer-no-mind-tools");
  await expect(line).toContainText("这个模型不能自己写进你的 Mind。");
  await expect(line.getByTestId("answer-no-mind-tools-add")).toHaveText("添加到 Mind");

  const second = await askInAssistant(window, "把它改长一点");
  await expect(second.getByTestId("answer-no-mind-tools")).toContainText(
    "这个模型不能自己写进你的 Mind。",
  );

  await narrowestAssistant(window);
  await expectFits(window, first);
  await expectFits(window, second);
  await screenshot(window, "no-mind-tools-line-narrow-zh");

  await pointAndClick(window, second.getByTestId("answer-no-mind-tools-add"));
  await expect(second.getByTestId("answer-added")).toContainText("已添加到“第三季度”");
  await expect(editor).toContainText("Answer to: 把它改长一点");
  await pointAndClick(window, second.getByTestId("answer-added-undo"));
  await expect(editor.locator("p")).toHaveText(["销售很好。"]);
  await app.close();
});

for (const language of ["en", "zh-CN"] as const) {
  test(`at 900 px the assistant is docked at the bottom and the line fits it, in ${language}`, async () => {
    const english = language === "en";
    const { app, window } = await start(
      language,
      english ? "Q3 review" : "第三季度",
      english ? "The quarter was good." : "销售很好。",
      { width: 900, height: 760 },
    );

    const answer = await askInAssistant(
      window,
      english ? "write a summary page" : "写一个总结页面",
    );

    await expect(window.getByTestId("assistant")).toHaveAttribute("data-dock", "bottom");
    await expect(answer.getByTestId("answer-no-mind-tools")).toContainText(
      english ? "This model can't write into your Mind itself." : "这个模型不能自己写进你的 Mind。",
    );
    await expectFits(window, answer);
    await app.close();
  });
}
