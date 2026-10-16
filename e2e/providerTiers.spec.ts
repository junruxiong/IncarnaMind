import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  acceptPlainTextKeys,
  assistantAnswers,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  setWindowSize,
  showSettingsPage,
} from "./app";

// ADR-0019: Settings → Models lists the first-class providers first and the rest under "More
// providers"; a provider the User already set up stays among theirs, whatever its tier.

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

const LANGUAGES = [
  {
    language: "en",
    more: "More…",
    first: ["Anthropic", "OpenAI", "Google", "DeepSeek", "Qwen"],
    folded: ["OpenRouter", "SiliconFlow"],
    setUp: "Mistral",
    rest: ["xAI", "Kimi", "GLM", "OpenAI-compatible server"],
    link: "Where do I find it?",
  },
  {
    language: "zh-CN",
    more: "更多…",
    first: ["Anthropic", "OpenAI", "Google", "DeepSeek", "通义千问"],
    folded: ["OpenRouter", "硅基流动"],
    setUp: "Mistral",
    rest: ["xAI", "Kimi", "智谱 GLM", "OpenAI 兼容服务器"],
    link: "在哪里找到密钥？",
  },
] as const;

/** Sets up Mistral (best effort) with a key and makes it the chat model, as the User did before. */
async function setUpMistral(window: Page): Promise<void> {
  await acceptPlainTextKeys(window);
  await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const provider = await bridge.saveChatProvider({
      kind: "mistral",
      modelId: "mistral-small-latest",
      apiKey: "key-not-real",
    });
    await bridge.updateSettings({
      user: { chatModel: { providerId: provider.id, modelId: "mistral-small-latest" } },
    });
  });
}

for (const { language, more, first, folded, setUp, rest, link } of LANGUAGES) {
  test(`Models lists first-class providers first, the rest under "${more}", and keeps a set-up one (${language})`, async () => {
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await setUpMistral(window);
    await window.evaluate(async (value) => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await bridge.updateSettings({ user: { language: value } });
    }, language);

    for (const [width, height] of [
      [1440, 900],
      [1100, 800],
    ] as const) {
      await setWindowSize(app, window, width, height);
      await window.getByRole("button", { name: /^(Settings|设置)$/ }).click();
      const settings = await showSettingsPage(window, "models");
      // The provider set up before still shows as the one in use.
      await expect(settings.getByTestId("chat-provider-current")).toHaveText("Mistral");
      await settings.getByTestId("chat-model-settings").getByRole("button").first().click();
      await settings.getByTestId("chat-choice-api-key").check();

      const form = settings.getByTestId("provider-form");
      // One row of the likeliest providers, one chosen; the rest are behind More…
      const radios = form.getByRole("radio");
      await expect(radios).toHaveCount(first.length);
      const labels = await radios.evaluateAll((all) =>
        all.map((each) => each.closest("label")?.textContent?.trim()),
      );
      expect(labels).toEqual(first.map((each) => expect.stringContaining(each)));
      await expect(form.getByRole("radio", { checked: true })).toHaveCount(1);

      const toggle = form.getByTestId("provider-more-toggle");
      await expect(toggle).toHaveText(more);
      await toggle.click();
      // The folded ones, the one already set up and the rest, join the row.
      for (const name of [...folded, setUp, ...rest]) {
        await expect(form.getByText(name, { exact: false }).first()).toBeVisible();
      }

      // Every provider has its way to a key.
      await expect(form.getByTestId("provider-key-help-toggle")).toContainText(link);

      await settings.getByRole("button", { name: /^(Cancel|取消)$/ }).click();
      await settings.getByRole("button", { name: /^(Close Settings|关闭设置)$/ }).click();
      await expect(settings).toBeHidden();
    }

    // The set-up best-effort provider answers in the assistant, with the scripted model.
    await newMind(window);
    const editor = window.getByTestId("mind-editor");
    await editor.click();
    await window.keyboard.press("ControlOrMeta+j");
    await window.keyboard.type("Does a best-effort provider still answer?");
    await window.keyboard.press("Enter");
    // Choosing Mistral allowed sending to it (ADR-0022): nothing asks before the Answer.
    await expect(window.getByTestId("consent-dialog")).toBeHidden();
    await expect(assistantAnswers(window).first()).toHaveAttribute("data-status", "done", {
      timeout: 15_000,
    });
    await app.close();
  });
}
