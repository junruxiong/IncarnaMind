import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { CATALOG_PROVIDERS } from "../src/core/providers/catalog";
import { createFakeOpenAI } from "../tests/helpers/fakeOpenAI";
import {
  acceptPlainTextKeys,
  addProviderForm,
  assistantAnswers,
  closeSettings,
  continueSetup,
  createDataFolder,
  interceptOpenExternal,
  launchApp,
  newMind,
  openSettings,
  removeDataFolder,
  setWindowSize,
  urlsOpened,
} from "./app";

/*
 * The key form (#719): one row of providers, the key, a Model pick list and one "Use this key"
 * that tests the key as it saves, in view at 900px high or in a card that scrolls inside itself;
 * "Where do I find it?", closed, with what the catalog knows of getting a key (#920); and the
 * ChatGPT sign-in's way back.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

const language = (window: Page, value: "en" | "zh-CN") =>
  window.evaluate(
    (chosen) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: chosen },
      }),
    value,
  );

/** The first Question with no model: the setup card waits, with the key form opened. */
async function openKeyForm(window: Page) {
  await newMind(window);
  await window.getByTestId("mind-editor").click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("What is in my Documents?");
  await window.keyboard.press("Enter");
  const card = window.getByTestId("model-setup");
  await card.waitFor();
  return card;
}

test("at 900px high the key form is in view, or scrolls inside its card, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    // Linux in CI has no keyring: its notice is accepted first, as a User does once.
    await acceptPlainTextKeys(window);
    for (const value of ["en", "zh-CN"] as const) {
      await language(window, value);
      const card = await openKeyForm(window);
      await card.getByTestId("setup-key").click();
      const form = card.getByTestId("provider-form");
      await expect(form.getByRole("radio", { checked: true })).toHaveCount(1);
      await expect(form.getByTestId("provider-model")).toBeVisible();
      await expect(form.getByTestId("provider-model").locator("option").first()).toContainText(
        value === "en" ? "Recommended" : "推荐",
      );
      await expect(form.getByTestId("provider-key-help-toggle")).toBeVisible();
      await expect(form.getByTestId("provider-key-link")).toBeHidden();
      // The card stays in the window; what doesn't fit scrolls inside it.
      const box = await card.boundingBox();
      expect((box?.y ?? 0) + (box?.height ?? 0)).toBeLessThanOrEqual(900);
      await form.getByTestId("provider-use-key").scrollIntoViewIfNeeded();
      await expect(form.getByTestId("provider-use-key")).toBeInViewport({ ratio: 1 });
      await form.getByTestId("provider-model").scrollIntoViewIfNeeded();
      await expect(form.getByTestId("provider-model")).toBeInViewport({ ratio: 1 });
      if (process.env.INCARNAMIND_SCREENSHOTS) {
        await window.screenshot({
          path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-form-900-${value}.png`,
        });
      }
      await card.getByTestId("setup-back").click();
    }
  } finally {
    await app.close();
  }
});

test("Where do I find it? opens with the keyboard, and holds the notes, the steps, the key line and the key page, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    // The first provider is Anthropic in English, which has all four; DeepSeek in Chinese, with no note and no key line.
    for (const [value, count, keyPage] of [
      ["en", 1, "Open Anthropic’s key page ↗"],
      ["zh-CN", 0, "打开 DeepSeek 的密钥页面 ↗"],
    ] as const) {
      await language(window, value);
      const card = await openKeyForm(window);
      await card.getByTestId("setup-key").click();
      const form = card.getByTestId("provider-form");
      const help = form.getByTestId("provider-key-help");
      await expect(help).not.toHaveAttribute("open", "");
      await form.locator('input[type="password"]').focus();
      await window.keyboard.press("Tab");
      await expect(form.getByTestId("provider-key-help-toggle")).toBeFocused();
      await window.keyboard.press("Enter");
      await expect(help).toHaveAttribute("open", "");
      await expect(help.getByTestId("provider-note")).toHaveCount(count);
      await expect(help.locator("ol > li")).toHaveCount(3);
      await expect(help.getByTestId("provider-key-line")).toHaveCount(count);
      await expect(help.getByTestId("provider-key-link")).toHaveText(keyPage);
      if (process.env.INCARNAMIND_SCREENSHOTS) {
        await window.screenshot({
          path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-help-open-${value}.png`,
        });
      }
      // The notes come first, then the steps, then the key line, then the key page.
      const order = await help.evaluate((element) =>
        [
          ...element.querySelectorAll(
            "[data-testid=provider-note], ol, [data-testid=provider-key-line], [data-testid=provider-key-link]",
          ),
        ].map((each) => each.getAttribute("data-testid") ?? each.tagName),
      );
      expect(order).toEqual(
        count
          ? ["provider-note", "OL", "provider-key-line", "provider-key-link"]
          : ["OL", "provider-key-link"],
      );
      // At first run the notes and the key line are in the disclosure only, and the model has no docs link.
      await expect(form.getByTestId("provider-note")).toHaveCount(count);
      await expect(form.getByTestId("provider-key-line")).toHaveCount(count);
      await expect(form.getByTestId("provider-docs-link")).toHaveCount(0);
      await window.keyboard.press("Enter");
      await expect(help).not.toHaveAttribute("open", "");
      await card.getByTestId("setup-back").click();
    }
  } finally {
    await app.close();
  }
});

test("in Settings the notes and the key line are under the key field, the steps and the key page behind Where do I find it?, and the model has a link to its models and prices; in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    await language(window, "en");
    const settings = await openSettings(window, "models");
    const form = await addProviderForm(settings, "anthropic");
    const help = form.getByTestId("provider-key-help");
    // The form opens on Anthropic, which has all of it, whichever language is chosen after.
    for (const [value, link] of [
      ["en", "Models and prices ↗"],
      ["zh-CN", "模型和价格 ↗"],
    ] as const) {
      await language(window, value);
      if (process.env.INCARNAMIND_SCREENSHOTS) {
        await settings.screenshot({
          path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-settings-${value}.png`,
        });
      }
      const note = form.getByTestId("provider-note");
      const line = form.getByTestId("provider-key-line");
      await expect(note).toBeVisible();
      await expect(line).toBeVisible();
      // One column with the hint lines' own spacing, no more.
      const [noteBox, lineBox] = await Promise.all([note.boundingBox(), line.boundingBox()]);
      expect((lineBox?.y ?? 0) - ((noteBox?.y ?? 0) + (noteBox?.height ?? 0))).toBeLessThanOrEqual(
        4,
      );
      await expect(help.locator("ol")).toBeHidden();
      await expect(form.getByTestId("provider-docs-link")).toHaveText(link);
      await expect(form.getByTestId("provider-docs-link")).toHaveAttribute("href", /^https:\/\//);
    }
    // The link is on the model's heading line, above its list: Tab goes from one to the other.
    const docs = form.getByTestId("provider-docs-link");
    const picker = form.getByTestId("provider-model");
    const [docsBox, pickerBox] = await Promise.all([docs.boundingBox(), picker.boundingBox()]);
    expect(docsBox?.y ?? 0).toBeLessThan(pickerBox?.y ?? 0);
    await docs.focus();
    await window.keyboard.press("Tab");
    await expect(picker).toBeFocused();
    await help.getByTestId("provider-key-help-toggle").click();
    await expect(help.locator("ol > li")).toHaveCount(3);
    await expect(help.getByTestId("provider-note")).toHaveCount(0);
    await expect(help.getByTestId("provider-key-line")).toHaveCount(0);
    await expect(help.getByTestId("provider-key-link")).toHaveText("打开 Anthropic 的密钥页面 ↗");
  } finally {
    await app.close();
  }
});

test("a key line says what to do in IncarnaMind and links to the company's own page for the rest, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    const settings = await openSettings(window, "models");
    for (const [name, en, zh] of [
      [
        "anthropic",
        "Paste an API key from the Anthropic Console. · Details ↗",
        "粘贴 Anthropic Console 中的 API 密钥。 · 详情 ↗",
      ],
      [
        "openai",
        "With a ChatGPT plan, you can use Continue with ChatGPT instead of a key. · Details ↗",
        "有 ChatGPT 订阅时，可以改用“使用 ChatGPT 继续”，无需密钥。 · 详情 ↗",
      ],
      [
        "google",
        "Paste a Gemini API key from Google AI Studio · Pricing ↗",
        "粘贴 Google AI Studio 的 Gemini API 密钥 · 价格 ↗",
      ],
      [
        "openrouter",
        "Pay per use with OpenRouter credits · Pricing ↗",
        "用 OpenRouter 额度按用量付费 · 价格 ↗",
      ],
      [
        "kimi",
        "If Kimi refuses requests, check your Kimi account · Pricing ↗",
        "如果 Kimi 拒绝请求，请检查你的 Kimi 账户 · 价格 ↗",
      ],
    ] as const) {
      await language(window, "en");
      const form = await addProviderForm(settings, name);
      const line = form.getByTestId("provider-key-line");
      await expect(line).toHaveText(en);
      await expect(line.getByRole("link")).toHaveAttribute("href", /^https:\/\//);
      if (process.env.INCARNAMIND_SCREENSHOTS) {
        await settings.screenshot({
          path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-line-${name}-en.png`,
        });
      }
      await language(window, "zh-CN");
      await expect(line).toHaveText(zh);
      if (process.env.INCARNAMIND_SCREENSHOTS) {
        await settings.screenshot({
          path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-line-${name}-zh-CN.png`,
        });
      }
      await settings.getByTestId("models-back").click();
    }
  } finally {
    await app.close();
  }
});

test("the provider's own line on what it does with the data sits under the sentence of where it goes, with a link to the page on it, in English and Chinese, at first run and in Settings", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    for (const [value, link, lead] of [
      ["en", "Data use ↗", ""],
      ["zh-CN", "数据使用 ↗", "数据使用："],
    ] as const) {
      await language(window, value);
      const card = await openKeyForm(window);
      await card.getByTestId("setup-key").click();
      const line = card.getByTestId("provider-form").getByTestId("provider-data-use");
      await expect(line).toContainText(lead);
      await expect(line.getByRole("link")).toHaveText(link);
      await expect(line.getByRole("link")).toHaveAttribute("href", /^https:\/\//);
      await card.getByTestId("setup-back").click();
    }

    await language(window, "en");
    const settings = await openSettings(window, "models");
    const line = (await addProviderForm(settings, "anthropic")).getByTestId("provider-data-use");
    await expect(line.getByRole("link")).toHaveText("Data use ↗");
    await language(window, "zh-CN");
    await expect(line).toContainText("数据使用：");
    await expect(line.getByRole("link")).toHaveText("数据使用 ↗");
  } finally {
    await app.close();
  }
});

test("every provider's line on what it does with the data is one line beside its link at 900px wide, in English and Chinese, at first run and in Settings", async () => {
  const { app, window } = await launchApp(dataDir);
  const hosted = CATALOG_PROVIDERS.filter(
    (each) => each.group !== "local" && each.id !== "openai-compatible",
  );
  const oneLine = (line: Locator) =>
    line.evaluate((element) => {
      const lineHeight = Number.parseFloat(getComputedStyle(element).lineHeight);
      return element.getBoundingClientRect().height < lineHeight * 1.5;
    });
  try {
    await setWindowSize(app, window, 900, 900);
    await acceptPlainTextKeys(window);
    let checked = 0;
    for (const [value, words] of [
      ["en", "en"],
      ["zh-CN", "zh"],
    ] as const) {
      await language(window, value);
      const card = await openKeyForm(window);
      await card.getByTestId("setup-key").click();
      const form = card.getByTestId("provider-form");
      await form.getByTestId("provider-more-toggle").click();
      for (const provider of hosted.filter((each) => each.support === "first-class")) {
        await form
          .locator("label", { has: window.locator(`input[value="${provider.id}"]`) })
          .click();
        const line = form.getByTestId("provider-data-use");
        await expect(line).toContainText(provider.dataUse.summary[words]);
        expect(await oneLine(line), `${provider.id} at first run, ${value}`).toBe(true);
        checked += 1;
      }
      await card.getByTestId("setup-back").click();

      await language(window, "en");
      const settings = await openSettings(window, "models");
      await language(window, value);
      for (const provider of hosted) {
        const line = (await addProviderForm(settings, provider.id)).getByTestId(
          "provider-data-use",
        );
        await expect(line).toContainText(provider.dataUse.summary[words]);
        expect(await oneLine(line), `${provider.id} in Settings, ${value}`).toBe(true);
        await expect(line.getByRole("link")).toHaveAttribute(
          "href",
          (words === "zh" && provider.dataUse.zhUrl) || (provider.dataUse.url ?? ""),
        );
        checked += 1;
        await settings.getByTestId("models-back").click();
      }
      await language(window, "en");
      await closeSettings(window);
    }
    expect(checked).toBeGreaterThan(40);
  } finally {
    await app.close();
  }
});

test("a way that can't be used yet says why, with the next step: Ollama not running, and Use this key before a key is pasted, in English and Chinese", async () => {
  // Nothing answers where Ollama would be.
  const { app, window } = await launchApp(dataDir, { ollama: "http://127.0.0.1:9" });
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    const card = await openKeyForm(window);
    await card.getByTestId("setup-other").click();
    const local = card.getByTestId("way-local");
    await expect(local).toContainText("Needs the free Ollama app");
    await expect(local.locator("xpath=..").getByRole("link", { name: /Get Ollama/ })).toBeVisible();
    await language(window, "zh-CN");
    await expect(local).toContainText("需要免费的 Ollama 应用");
    await language(window, "en");
    await card.getByTestId("setup-key").click();
    const form = card.getByTestId("provider-form");
    await expect(form.getByTestId("provider-use-key")).toBeDisabled();
    await expect(form.getByTestId("provider-use-key-why")).toHaveText("Paste the key first.");
    await form.getByLabel(/^API key/).fill("sk-ant-something");
    await expect(form.getByTestId("provider-use-key-why")).toHaveCount(0);
    await expect(form.getByTestId("provider-use-key")).toBeEnabled();
    await form.getByLabel(/^API key/).fill("");

    await language(window, "zh-CN");
    await expect(form.getByTestId("provider-use-key-why")).toHaveText("请先粘贴密钥。");
    if (process.env.INCARNAMIND_SCREENSHOTS) {
      await window.screenshot({
        path: `${process.env.INCARNAMIND_SCREENSHOTS}/why-unavailable-zh-CN.png`,
      });
    }
  } finally {
    await app.close();
  }
});

/** Made-up keys: the prefixes some services' keys all start with, and a bare sk- that names nobody. */
const KEYS = {
  anthropic: `sk-ant-api03-${"aZ3".repeat(14)}`,
  openai: `sk-proj-${"aZ3".repeat(20)}`,
  openrouter: `sk-or-v1-${"0f9a".repeat(16)}`,
  xai: `xai-${"aZ3".repeat(26)}`,
  bare: `sk-${"aZ3".repeat(16)}`,
};

const checked = (form: Locator) => form.getByRole("radio", { checked: true });

for (const { value, first, other, anyway, notAnthropic } of [
  {
    value: "en",
    first: "Anthropic",
    other: "OpenAI",
    anyway: "Use it for Anthropic anyway",
    notAnthropic:
      "This isn’t Anthropic’s key: its keys start with sk-ant-. Choose the service it belongs to.",
  },
  {
    value: "zh-CN",
    first: "DeepSeek",
    other: "OpenAI",
    anyway: "仍用于 DeepSeek",
    notAnthropic: "这不是 Anthropic 的密钥：它的密钥以 sk-ant- 开头。请选择它所属的服务。",
  },
] as const) {
  test(`a key goes only to its own service: its prefix picks the service, a key the chosen one can't have is never sent, and nothing goes before the User agrees (${value})`, async () => {
    const { app, window } = await launchApp(dataDir);
    try {
      await setWindowSize(app, window, 1100, 900);
      await acceptPlainTextKeys(window);
      await language(window, value);
      const card = await openKeyForm(window);
      await card.getByTestId("setup-key").click();
      const form = card.getByTestId("provider-form");
      const key = form.locator('input[type="password"]');
      const line = form.getByTestId("provider-key-wrong");
      const useKey = form.getByTestId("provider-use-key");
      await expect(checked(form)).toHaveAccessibleName(first);

      await key.fill(KEYS.openai);
      await expect(checked(form)).toHaveAccessibleName(other);
      await expect(line).toContainText(
        value === "en" ? "This is OpenAI’s key" : "这是 OpenAI 的密钥",
      );
      await expect(line.getByTestId("provider-key-anyway")).toHaveText(anyway);
      await expect(key).toHaveValue(KEYS.openai);
      await expect(useKey).toBeEnabled();
      if (process.env.INCARNAMIND_SCREENSHOTS) {
        await window.screenshot({
          path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-wrong-service-${value}.png`,
        });
      }
      await line.getByTestId("provider-key-anyway").click();
      await expect(checked(form)).toHaveAccessibleName(first);
      await expect(key).toHaveValue(KEYS.openai);
      await expect(line).toHaveCount(0);
      await expect(useKey).toBeEnabled();

      await key.fill(KEYS.openrouter);
      await expect(checked(form)).toHaveAccessibleName("OpenRouter");

      // xAI is a best-effort provider, so first run doesn't offer it (ADR-0019): the key waits for the User.
      await key.fill(KEYS.xai);
      await expect(line).toContainText(value === "en" ? "This is xAI’s key" : "这是 xAI 的密钥");
      await expect(checked(form)).toHaveAccessibleName("OpenRouter");
      await expect(useKey).toBeDisabled();
      await key.press("Enter");
      await expect(form.getByTestId("connection-test")).toHaveCount(0);
      await expect(useKey).toBeDisabled();
      await line.getByTestId("provider-key-anyway").click();
      await expect(line).toHaveCount(0);
      await expect(useKey).toBeEnabled();
      await key.fill(`${KEYS.xai}x`);
      await expect(line).toHaveCount(1);

      await form.getByRole("radio", { name: "Anthropic" }).check({ force: true });
      await key.fill(KEYS.bare);
      await expect(line).toHaveText(notAnthropic);
      await expect(line.getByTestId("provider-key-anyway")).toHaveCount(0);
      await expect(useKey).toBeDisabled();
      await key.press("Enter");
      await expect(form.getByTestId("connection-test")).toHaveCount(0);
      await expect(useKey).toHaveText(value === "en" ? "Use this key" : "使用这个密钥");

      await form.getByRole("radio", { name: "DeepSeek" }).check({ force: true });
      await expect(key).toHaveValue(KEYS.bare);
      await expect(line).toHaveCount(0);
      await expect(useKey).toBeEnabled();

      await key.fill("not-a-key-at-all");
      await expect(line).toHaveCount(0);
      await expect(useKey).toBeEnabled();
    } finally {
      await app.close();
    }
  });
}

test("Kimi's form opens on the region of the interface's language: International in English, China in Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    const settings = await openSettings(window, "models");
    for (const [value, region] of [
      ["en", "intl"],
      ["zh-CN", "cn"],
    ] as const) {
      await language(window, value);
      const form = await addProviderForm(settings, "kimi");
      await expect(form.getByRole("combobox", { name: /^(Region|地域)/ })).toHaveValue(region);
      await settings.getByTestId("models-back").click();
    }
  } finally {
    await app.close();
  }
});

test("in Settings, a key that is not the page's service goes nowhere on Add a provider and on a provider's page, and the page stays the provider's", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    const settings = await openSettings(window, "models");
    const form = await addProviderForm(settings, "anthropic");
    const key = form.locator('input[type="password"]');
    const line = form.getByTestId("provider-key-wrong");
    const useKey = form.getByTestId("provider-use-key");
    const title = settings.getByTestId("provider-page-title");

    await key.fill(KEYS.openai);
    await expect(line).toContainText("This is OpenAI’s key");
    await expect(title).toHaveText("Anthropic");
    await expect(useKey).toBeDisabled();
    await key.press("Enter");
    await expect(form.getByTestId("connection-test")).toHaveCount(0);
    await key.fill(KEYS.bare);
    await expect(line).toHaveText("This isn’t Anthropic’s key: its keys start with sk-ant-.");
    await expect(useKey).toBeDisabled();
    await key.fill(KEYS.openai);
    await line.getByTestId("provider-key-anyway").click();
    await expect(line).toHaveCount(0);
    await expect(useKey).toBeEnabled();
    await expect(title).toHaveText("Anthropic");
    await key.fill(KEYS.anthropic);
    await expect(line).toHaveCount(0);
    await expect(useKey).toBeEnabled();
    await settings.getByTestId("models-back").click();

    const saved = await window.evaluate(async (apiKey) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      return core.saveChatProvider({ kind: "anthropic", apiKey, modelId: "claude-haiku-5-5" });
    }, KEYS.anthropic);
    await settings.getByTestId("provider-row").first().click();
    const page = settings.getByTestId("provider-page");
    await expect(page).toHaveAttribute("data-provider-id", saved.id);
    const pageKey = page.getByTestId("provider-page-key");
    const save = page.getByTestId("provider-page-save-key");
    await pageKey.fill(KEYS.bare);
    await expect(page.getByTestId("provider-key-wrong")).toHaveText(
      "This isn’t Anthropic’s key: its keys start with sk-ant-.",
    );
    await expect(save).toBeDisabled();
    await pageKey.press("Enter");
    await expect(page.getByTestId("provider-page-key-error")).toHaveCount(0);
    await pageKey.fill(KEYS.openai);
    await expect(page.getByTestId("provider-key-wrong")).toContainText("This is OpenAI’s key");
    await expect(save).toBeDisabled();
    await page.getByTestId("provider-key-anyway").click();
    await expect(save).toBeEnabled();
    await pageKey.fill(KEYS.anthropic);
    await expect(page.getByTestId("provider-key-wrong")).toHaveCount(0);
    await expect(save).toBeEnabled();
  } finally {
    await app.close();
  }
});

/**
 * An OpenAI-compatible server that refuses one key and answers with the other, and answers a
 * third with Anthropic's 400 for a model that needs 30-day data retention.
 */
async function server(): Promise<{ server: Server; url: string }> {
  const http = createServer((request, response) => {
    request.resume();
    request.on("end", () => {
      if (request.headers.authorization === "Bearer locked-key") {
        response.writeHead(400, { "content-type": "application/json" });
        response.end(
          JSON.stringify({
            type: "error",
            error: {
              type: "invalid_request_error",
              message:
                "In order to access this model, your organization or workspace must have data retention enabled.",
            },
          }),
        );
        return;
      }
      const good = request.headers.authorization === "Bearer good-key";
      response.writeHead(good ? 200 : 401, { "content-type": "application/json" });
      response.end(
        JSON.stringify(
          good
            ? {
                id: "c1",
                object: "chat.completion",
                created: 1,
                model: "any",
                choices: [
                  {
                    index: 0,
                    message: { role: "assistant", content: "OK" },
                    finish_reason: "stop",
                  },
                ],
                usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
              }
            : { error: { message: "bad key" } },
        ),
      );
    });
  });
  await new Promise<void>((resolve) => http.listen(0, "127.0.0.1", resolve));
  return { server: http, url: `http://127.0.0.1:${(http.address() as AddressInfo).port}/v1` };
}

test("Use this key shows a refused key in place, and with a good one saves and the waiting Question is asked", async () => {
  const fake = await server();
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    const card = await openKeyForm(window);
    // A server the User gives is offered in Settings, not at first run (ADR-0019); saved there,
    // it answers the Question waiting in the setup card all the same.
    const settings = await openSettings(window, "models");
    const form = await addProviderForm(settings, "openai-compatible");
    // A compatible server keeps a free-text Model.
    await expect(form.getByTestId("provider-model-text")).toBeVisible();
    await form.getByLabel("Server URL").fill(fake.url);
    await form.getByTestId("provider-model-text").fill("any");
    await form.getByLabel(/^API key/).fill("wrong-key");
    await form.getByTestId("provider-use-key").click();
    await expect(form.getByTestId("connection-test")).toContainText("The key was refused", {
      timeout: 20_000,
    });
    await expect(card).toBeVisible();

    await form.getByLabel(/^API key/).fill("good-key");
    await form.getByTestId("provider-use-key").click();
    await expect(form).toBeHidden();
    await closeSettings(window);
    await continueSetup(window);
    // The waiting Question was asked: the conversation holds its Answer (the server only answers
    // the connection test, so how it ends is not the point).
    await expect(assistantAnswers(window).first()).toBeVisible({ timeout: 15_000 });
  } finally {
    await app.close();
    fake.server.close();
  }
});

test("a model that needs 30-day data retention says so with the model's name, and its action puts the cursor where the model is chosen; in English and Chinese", async () => {
  const fake = await server();
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    await language(window, "en");
    const settings = await openSettings(window, "models");
    const form = await addProviderForm(settings, "openai-compatible");
    await form.locator('input[type="url"]').fill(fake.url);
    await form.getByTestId("provider-model-text").fill("claude-fable-5-1");
    await form.locator('input[type="password"]').fill("locked-key");
    await form.getByTestId("provider-use-key").click();
    const result = form.getByTestId("connection-test");
    await expect(result).toContainText(
      "The key works, but your organization disallows 30-day retention for claude-fable-5-1. Choose another model. · Choose another model",
      { timeout: 20_000 },
    );
    await expect(result).not.toContainText("Console");
    await language(window, "zh-CN");
    await expect(result).toContainText(
      "密钥可用，但你的组织未开启 claude-fable-5-1 所需的 30 天数据保留。请换一个模型。· 换一个模型",
    );
    await result.getByTestId("connection-test-choose-model").click();
    await expect(form.getByTestId("provider-model-text")).toBeFocused();
  } finally {
    await app.close();
    fake.server.close();
  }
});

test("the ChatGPT wait offers the sign-in page again, and after five minutes says so in place; Cancel goes back", async () => {
  const openai = await createFakeOpenAI();
  const { app, window } = await launchApp(dataDir, {
    fakeChat: true,
    chatGpt: openai.endpoints.issuer,
  });
  try {
    await setWindowSize(app, window, 1100, 900);
    await interceptOpenExternal(app);
    await acceptPlainTextKeys(window);
    await window.clock.install();
    const card = await openKeyForm(window);
    await card.getByTestId("setup-chatgpt").click();
    await expect.poll(() => urlsOpened(app)).toHaveLength(1);
    await expect(card).toContainText("Finish signing in in your browser");
    await expect(card).toContainText("Nothing extra to pay");
    await card.getByTestId("chatgpt-reopen").click();
    await expect.poll(() => urlsOpened(app)).toHaveLength(2);
    await expect(card.getByTestId("chatgpt-late")).toHaveCount(0);

    await window.clock.fastForward("05:01");
    await expect(card.getByTestId("chatgpt-late")).toHaveText(
      "Didn’t finish? Open the sign-in page again.",
    );
    await expect(window.getByTestId("chatgpt-error")).toHaveCount(0);

    await card.getByTestId("chatgpt-cancel").click();
    await expect(card.getByTestId("setup-chatgpt")).toBeVisible();
  } finally {
    await app.close();
  }
});
