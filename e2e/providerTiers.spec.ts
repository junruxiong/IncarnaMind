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
    more: "More providers",
    first: ["Anthropic", "OpenAI", "Google", "OpenRouter", "DeepSeek", "Qwen", "SiliconFlow"],
    setUp: "Mistral",
    rest: ["xAI", "Kimi", "GLM", "OpenAI-compatible server"],
    guide: "How do I get one?",
  },
  {
    language: "zh-CN",
    more: "更多提供商",
    first: ["Anthropic", "OpenAI", "Google", "OpenRouter", "DeepSeek", "通义千问", "硅基流动"],
    setUp: "Mistral",
    rest: ["xAI", "Kimi", "智谱 GLM", "OpenAI 兼容服务器"],
    guide: "如何获取密钥？",
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

for (const { language, more, first, setUp, rest, guide } of LANGUAGES) {
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
      // First-class providers first, then the one already set up; the rest are folded away.
      await expect(form.getByRole("radio")).toHaveCount(first.length + 1);
      await expect(form.getByTestId("provider-more")).toHaveCount(0);
      const labels = await form
        .getByRole("radio")
        .evaluateAll((radios) => radios.map((each) => each.closest("label")?.textContent?.trim()));
      expect(labels).toEqual([...first.map((each) => expect.stringContaining(each)), setUp]);

      const toggle = form.getByTestId("provider-more-toggle");
      await expect(toggle).toHaveText(more);
      await expect(toggle).toHaveAttribute("aria-expanded", "false");
      await toggle.click();
      await expect(toggle).toHaveAttribute("aria-expanded", "true");
      const folded = form.getByTestId("provider-more").getByRole("radio");
      await expect(folded).toHaveCount(rest.length);
      for (const name of rest) {
        await expect(form.getByTestId("provider-more").getByText(name)).toBeVisible();
      }
      // The folded group sits under the first-class ones.
      const groupTop = (await form.getByTestId("provider-more").boundingBox())?.y ?? 0;
      const lastFirst = await form.getByRole("radio").nth(first.length).boundingBox();
      expect(groupTop).toBeGreaterThan(lastFirst?.y ?? Number.POSITIVE_INFINITY);

      // A first-class provider has a key guide; a best-effort one has only its key link.
      await form.getByRole("radio").first().check();
      await expect(form.getByTestId("provider-key-guide")).toContainText(guide);
      await form.getByTestId("provider-key-guide").locator("summary").click();
      await expect(form.getByTestId("provider-key-guide").getByRole("listitem")).toHaveCount(3);
      await folded.first().check();
      await expect(form.getByTestId("provider-key-guide")).toHaveCount(0);
      await expect(form.getByRole("link", { name: /Get a key|获取密钥/ })).toBeVisible();

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
    // Mistral is a cloud service: the first Question asks before sending anything.
    await window.getByTestId("consent-allow").click();
    await expect(assistantAnswers(window).first()).toHaveAttribute("data-status", "done", {
      timeout: 15_000,
    });
    await app.close();
  });
}
