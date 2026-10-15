import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { createFakeOpenAI, signInAt } from "../tests/helpers/fakeOpenAI";
import {
  acceptPlainTextKeys,
  assistantAnswers,
  createDataFolder,
  interceptOpenExternal,
  launchApp,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  urlsOpened,
} from "./app";

/*
 * The first own Question with no model set (#103): it waits in the conversation while the
 * composer becomes the setup card, which leads with the ChatGPT plan and a key, the same size
 * (#890); the Question is then asked without being typed again.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("by mouse alone: the first own Question, Use my ChatGPT plan, and its Answer, with no consent dialog", async () => {
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
    await card.getByTestId("setup-chatgpt").click();
    await expect.poll(() => urlsOpened(app)).toHaveLength(1);
    const [authorize] = await urlsOpened(app);
    expect((await signInAt(openai, authorize ?? "")).status).toBe(200);
    await window.getByTestId("chatgpt-welcome-ok").click();

    await expect(card).toBeHidden();
    await expect(assistantAnswers(window).first()).toHaveAttribute("data-status", "done", {
      timeout: 15_000,
    });
  } finally {
    await app.close();
  }
});

/** Asks the first own Question in a new Mind; the setup card takes the composer's place. */
async function askFirstQuestion(window: Page) {
  await newMind(window);
  await window.getByTestId("mind-editor").click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("What is in my Documents?");
  await window.keyboard.press("Enter");
  const card = window.getByTestId("model-setup");
  await expect(card).toBeVisible();
  return card;
}

const WHO = {
  en: {
    chatgpt: "If you pay for ChatGPT Plus or Pro",
    key: "From OpenAI, Anthropic, Google, DeepSeek and more, yours or your company's",
  },
  "zh-CN": {
    chatgpt: "如果你订阅了 ChatGPT Plus 或 Pro（中国大陆无法使用）",
    key: "来自 OpenAI、Anthropic、Google、DeepSeek 等，你自己的或公司的",
  },
} as const;

for (const [width, height] of [
  [1100, 760],
  [1440, 900],
] as const) {
  test(`at ${width} px the card offers the ChatGPT plan and Paste an API key at the same size, each with who it's for; in Chinese the key comes first`, async () => {
    const { app, window } = await launchApp(dataDir);
    try {
      await setWindowSize(app, window, width, height);
      await acceptPlainTextKeys(window);
      const card = await askFirstQuestion(window);
      for (const language of ["en", "zh-CN"] as const) {
        await setLanguage(window, language);
        const plan = card.getByTestId("setup-chatgpt");
        const key = card.getByTestId("setup-key");
        await expect(card.getByTestId("setup-chatgpt-for")).toHaveText(WHO[language].chatgpt);
        await expect(card.getByTestId("setup-key-for")).toHaveText(WHO[language].key);
        await expect(plan).toBeInViewport({ ratio: 1 });
        await expect(key).toBeInViewport({ ratio: 1 });
        const planBox = await plan.boundingBox();
        const keyBox = await key.boundingBox();
        const cardBox = await card.boundingBox();
        if (!planBox || !keyBox || !cardBox) throw new Error("The two ways aren't laid out.");
        expect(Math.abs(planBox.width - keyBox.width)).toBeLessThan(1);
        expect(Math.abs(planBox.height - keyBox.height)).toBeLessThan(1);
        for (const box of [planBox, keyBox]) {
          expect(box.x).toBeGreaterThanOrEqual(cardBox.x);
          expect(box.x + box.width).toBeLessThanOrEqual(cardBox.x + cardBox.width);
        }
        const [first, second] = language === "en" ? [planBox, keyBox] : [keyBox, planBox];
        const sameRow = Math.abs(first.y - second.y) < 1;
        expect(sameRow ? second.x > first.x : second.y > first.y).toBe(true);
      }
    } finally {
      await app.close();
    }
  });
}

test("a newcomer pastes a key from the card in two clicks, picks Claude Sonnet 5.5 from the list without typing an id, and gets an Answer", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    const card = await askFirstQuestion(window);

    await card.getByTestId("setup-key").click();
    const form = card.getByTestId("provider-form");
    await expect(card.getByTestId("provider-form-title")).toHaveText("Paste your Anthropic key");
    await expect(form.getByLabel(/^API key/)).toBeFocused();
    await window.keyboard.insertText("sk-ant-a-key-from-the-company");

    const model = form.getByTestId("provider-model");
    await expect(model.locator("option").first()).toHaveText("Claude Haiku 5.5");
    await expect(form.getByTestId("provider-model-reason")).toHaveText(
      "Recommended: Claude Haiku 5.5",
    );
    const options = await model.locator("option").allTextContents();
    expect(options).toEqual(
      expect.arrayContaining([
        "Claude Fable 5.1 · Most capable",
        "Claude Sonnet 5.5 · Good with images and scans",
        "Claude Opus 5.5",
      ]),
    );
    expect(options.at(-1)).toBe("Another model…");
    await model.selectOption({ label: "Claude Sonnet 5.5 · Good with images and scans" });
    await form.getByTestId("provider-use-key").click();

    await expect(card).toBeHidden();
    await expect(assistantAnswers(window).first()).toHaveAttribute("data-status", "done", {
      timeout: 15_000,
    });
    const chatModel = await window.evaluate(
      async () =>
        (await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.getSettings())
          .user.chatModel,
    );
    expect(chatModel?.modelId).toBe("claude-sonnet-5-5");
  } finally {
    await app.close();
  }
});

test("the key step's title follows the company picked at once, and Another model… shows that company's example id, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    const card = await askFirstQuestion(window);
    const title = card.getByTestId("provider-form-title");
    const form = card.getByTestId("provider-form");
    const chip = (name: string) =>
      form.locator("label").filter({ hasText: new RegExp(`^${name}$`) });

    await card.getByTestId("setup-key").click();
    await expect(title).toHaveText("Paste your Anthropic key");
    await chip("OpenAI").click();
    await expect(title).toHaveText("Paste your OpenAI key");
    await form.getByTestId("provider-model").selectOption({ label: "Another model…" });
    await expect(form.getByTestId("provider-model-text")).toHaveAttribute(
      "placeholder",
      "e.g. gpt-6-luna",
    );

    await card.getByTestId("setup-back").click();
    await setLanguage(window, "zh-CN");
    await card.getByTestId("setup-key").click();
    // In Chinese the key step opens on DeepSeek, and says so until another company is picked.
    await expect(title).toHaveText("粘贴你的 DeepSeek 密钥");
    await chip("Anthropic").click();
    await expect(title).toHaveText("粘贴你的 Anthropic 密钥");
    await form.getByTestId("provider-model").selectOption({ label: "其他模型…" });
    await expect(form.getByTestId("provider-model-text")).toHaveAttribute(
      "placeholder",
      "例如 claude-haiku-5-5",
    );
  } finally {
    await app.close();
  }
});
