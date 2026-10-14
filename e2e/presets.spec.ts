import { expect, test } from "@playwright/test";
import {
  acceptPlainTextKeys,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openSettings,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * Presets (#863): a hosted service that speaks OpenAI's API is one chip under More…, with its own
 * name and key page, and until its model passes the citing evaluation it says so.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("Groq is a chip under More…, with its key page and its model marked not yet tested for checked Citations, in English and Chinese", async () => {
  // Nothing answers where Ollama would be, so this computer's own Ollama isn't asked.
  const { app, window } = await launchApp(dataDir, { ollama: "http://127.0.0.1:9" });
  try {
    await dismissChatSetup(window);
    await setWindowSize(app, window, 1440, 900);
    await acceptPlainTextKeys(window);
    const settings = await openSettings(window, "models");
    await settings.getByTestId("chat-choice-api-key").check();
    const form = settings.getByTestId("provider-form");
    await form.getByTestId("provider-more-toggle").click();
    // The provider chips are labels around visually hidden radios.
    await form.getByRole("radio", { name: "Groq", exact: true }).locator("xpath=..").click();
    await expect(form.getByRole("radio", { name: "Groq", exact: true })).toBeChecked();
    await expect(form.getByTestId("provider-model")).toHaveValue("openai/gpt-oss-120b");
    const reason = form.getByTestId("provider-model-reason");
    await expect(reason).toHaveText("Not yet tested for checked Citations");
    await expect(form.getByTestId("provider-key-link")).toHaveAttribute(
      "href",
      "https://console.groq.com/keys",
    );
    await expect(form.getByTestId("provider-sends")).toContainText("go to Groq");
    if (process.env.INCARNAMIND_SCREENSHOTS) {
      await settings.screenshot({
        path: `${process.env.INCARNAMIND_SCREENSHOTS}/preset-groq-en.png`,
      });
    }

    await setLanguage(window, "zh-CN");
    await expect(reason).toHaveText("尚未测试引用核对");
    await expect(form.getByRole("radio", { name: "Groq", exact: true })).toBeChecked();
  } finally {
    await app.close();
  }
});
