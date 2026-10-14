import { expect, test } from "@playwright/test";
import {
  askInAssistant,
  assistantAnswers,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  enterContent,
  launchApp,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/**
 * Esc and the quiet actions in the assistant (#612): Esc with the focus in it
 * and nothing in it open returns the focus to the Mind and leaves the
 * assistant as it is, its draft kept; an open menu closes first, by its own
 * Esc; only the chevron folds it. Copy · Add to note under an Answer are in
 * weight 400 and the meta grey, as the steps line above them is.
 */
let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

async function start(language: "en" | "zh-CN" = "en") {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await connectLocalChatModel(window);
  await setLanguage(window, language);
  await newMind(window);
  await window.getByTestId("mind-title").fill("Tides");
  await enterContent(window);
  await window.keyboard.type("Spring tides follow the Moon.");
  return { app, window };
}

test("Esc returns the focus to the Mind and leaves the assistant open with its draft; a menu closes first; only the chevron folds it", async () => {
  const { app, window } = await start();
  const assistant = window.getByTestId("assistant");
  const editor = window.getByTestId("mind-editor");
  const input = window.getByTestId("composer-input");

  // A half-typed Question, then Esc: the Mind has the focus, the assistant stays, the draft is kept.
  await window.keyboard.press("ControlOrMeta+j");
  await expect(input).toBeFocused();
  await window.keyboard.type("What makes the");
  await window.keyboard.press("Escape");
  await expect(editor).toBeFocused();
  await expect(assistant).not.toHaveAttribute("data-folded");
  await expect(input).toHaveValue("What makes the");

  // With the model menu open, the first Esc closes the menu only.
  await window.keyboard.press("ControlOrMeta+j");
  await expect(input).toBeFocused();
  await window.getByTestId("composer-model").click();
  const menu = window.getByTestId("composer-model-menu");
  await expect(menu).toBeVisible();
  await window.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await expect(editor).not.toBeFocused();
  await expect(assistant).not.toHaveAttribute("data-folded");
  await window.keyboard.press("Escape");
  await expect(editor).toBeFocused();

  // The chevron folds and unfolds it; folded, Esc does nothing to it.
  await window.getByTestId("assistant-fold").click();
  await expect(assistant).toHaveAttribute("data-folded", "true");
  await window.keyboard.press("Escape");
  await expect(assistant).toHaveAttribute("data-folded", "true");
  await window.getByTestId("assistant-fold").click();
  await expect(assistant).not.toHaveAttribute("data-folded");
  await app.close();
});

for (const language of ["en", "zh-CN"] as const) {
  test(`Copy · Add to note are in weight 400 and the meta grey like the steps line (${language})`, async () => {
    const { app, window } = await start(language);
    const answer = await askInAssistant(window, "Say something plain");
    const steps = answer.getByTestId("answer-steps");
    const style = (locator: typeof steps) =>
      locator.evaluate((el) => {
        const { fontWeight, color } = getComputedStyle(el);
        return { fontWeight, color };
      });
    const meta = await style(steps);
    for (const id of ["answer-copy", "answer-add"]) {
      const action = await style(assistantAnswers(window).last().getByTestId(id));
      expect(action.fontWeight, `${id} weight`).toBe("400");
      expect(action.color, `${id} colour`).toBe(meta.color);
    }
    await app.close();
  });
}
