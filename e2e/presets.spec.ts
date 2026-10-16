import { expect, test } from "@playwright/test";
import {
  acceptPlainTextKeys,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  openSettings,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * Presets: a hosted service that speaks OpenAI's API is one chip under More…, with its own name
 * and key page, and until its model passes the citing evaluation it says so.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("each preset is a chip under More…; Groq's has its key page and its model marked not yet tested for checked Citations, in English and Chinese", async () => {
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
    for (const name of ["Groq", "Together AI", "Fireworks AI", "Cerebras", "Cohere", "Venice"]) {
      await expect(form.getByRole("radio", { name, exact: true })).toHaveCount(1);
    }
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

test("the setup card offers only the first-class providers: no preset and no best-effort provider", async () => {
  const { app, window } = await launchApp(dataDir, { ollama: "http://127.0.0.1:9" });
  try {
    await setWindowSize(app, window, 1100, 900);
    await newMind(window);
    await window.getByTestId("mind-editor").click();
    await window.keyboard.press("ControlOrMeta+j");
    await window.keyboard.type("What is in my Documents?");
    await window.keyboard.press("Enter");
    const card = window.getByTestId("model-setup");
    await card.getByTestId("setup-key").click();
    const form = card.getByTestId("provider-form");
    await form.getByTestId("provider-more-toggle").click();
    await expect(form.getByRole("radio", { name: "OpenRouter", exact: true })).toHaveCount(1);
    for (const name of ["Groq", "Mistral", "xAI", "OpenAI-compatible server"]) {
      await expect(form.getByRole("radio", { name, exact: true })).toHaveCount(0);
    }
    await expect(form.getByRole("radio", { name: /^(Kimi|GLM)/ })).toHaveCount(0);
  } finally {
    await app.close();
  }
});

test("Amazon Bedrock asks for its region from a list, and Cloudflare for its account ID, before Use this key; in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir, { ollama: "http://127.0.0.1:9" });
  try {
    await dismissChatSetup(window);
    await setWindowSize(app, window, 1440, 900);
    await acceptPlainTextKeys(window);
    const settings = await openSettings(window, "models");
    await settings.getByTestId("chat-choice-api-key").check();
    const form = settings.getByTestId("provider-form");
    await form.getByTestId("provider-more-toggle").click();
    const pick = (name: string) =>
      form.getByRole("radio", { name, exact: true }).locator("xpath=..").click();

    await pick("Amazon Bedrock");
    const region = form.getByRole("combobox", { name: /^Region/ });
    await expect(region).toHaveValue("us-east-1");
    await expect(form.getByTestId("provider-field-hint")).toContainText("long-term API key");
    await region.selectOption("eu-west-1");
    await expect(region.locator("option:checked")).toHaveText("Europe (Ireland) · eu-west-1");
    await form.getByLabel(/^API key/).fill("bedrock-key");
    await expect(form.getByTestId("provider-use-key")).toBeEnabled();
    await expect(form.getByTestId("provider-sends")).toContainText("go to Amazon Bedrock");

    await pick("Cloudflare Workers AI");
    const account = form.getByTestId("provider-field");
    await expect(account).toHaveAccessibleName(/^Account ID/);
    await expect(form.getByTestId("provider-field-hint")).toContainText("Use REST API");
    await form.getByLabel(/^API key/).fill("cloudflare-token");
    await expect(form.getByTestId("provider-use-key-why")).toHaveText("Fill in Account ID first.");
    await account.fill("0123456789abcdef0123456789abcdef");
    await expect(form.getByTestId("provider-use-key-why")).toHaveCount(0);
    if (process.env.INCARNAMIND_SCREENSHOTS) {
      await settings.screenshot({
        path: `${process.env.INCARNAMIND_SCREENSHOTS}/preset-cloudflare-en.png`,
      });
    }

    await setLanguage(window, "zh-CN");
    await expect(account).toHaveAccessibleName(/^账户 ID/);
    await account.fill("");
    await expect(form.getByTestId("provider-use-key-why")).toHaveText("请先填写账户 ID。");
    await pick("Amazon Bedrock");
    await expect(form.getByRole("combobox", { name: /^区域/ })).toBeVisible();
    await expect(form.getByTestId("provider-field-hint")).toContainText("长期 API 密钥");
  } finally {
    await app.close();
  }
});
