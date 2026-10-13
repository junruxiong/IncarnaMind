import { expect, test } from "@playwright/test";
import { createFakeOpenAI, signInAt } from "../tests/helpers/fakeOpenAI";
import {
  acceptPlainTextKeys,
  assistantAnswers,
  createDataFolder,
  interceptOpenExternal,
  launchApp,
  newMind,
  removeDataFolder,
  setWindowSize,
  urlsOpened,
} from "./app";

/*
 * The first own Question with no model set (#103): it waits in the conversation while the
 * composer becomes the setup card, whose one recommended button signs in with the ChatGPT plan;
 * the Question is then asked without being typed again.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("by mouse alone: the first own Question, Use my ChatGPT plan, and its Answer, with no dialog between but OpenAI's own sign-in", async () => {
  const openai = await createFakeOpenAI();
  const { app, window } = await launchApp(dataDir, {
    fakeChat: true,
    chatGpt: openai.endpoints.issuer,
  });
  try {
    await setWindowSize(app, window, 1100, 900);
    await interceptOpenExternal(app);
    await acceptPlainTextKeys(window);
    await newMind(window);
    await window.getByTestId("mind-editor").click();
    await window.keyboard.press("ControlOrMeta+j");
    await window.keyboard.type("What is in my Documents?");
    await window.keyboard.press("Enter");

    const card = window.getByTestId("model-setup");
    await expect(card).toBeVisible();
    await expect(window.getByTestId("assistant-log")).toContainText("What is in my Documents?");
    await card.getByTestId("setup-primary").click();
    await expect.poll(() => urlsOpened(app)).toHaveLength(1);
    const [authorize] = await urlsOpened(app);
    expect((await signInAt(openai, authorize ?? "")).status).toBe(200);
    // The chat consent for OpenAI is #718's to remove; until then it is answered here.
    await window.getByTestId("consent-dialog").getByTestId("consent-allow").click();
    await window.getByTestId("chatgpt-welcome-ok").click();

    await expect(card).toBeHidden();
    await expect(assistantAnswers(window).first()).toHaveAttribute("data-status", "done", {
      timeout: 15_000,
    });
  } finally {
    await app.close();
  }
});
