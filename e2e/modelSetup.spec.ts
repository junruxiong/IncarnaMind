import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { createFakeOpenAI, signInAt } from "../tests/helpers/fakeOpenAI";
import {
  acceptPlainTextKeys,
  assistantAnswers,
  createDataFolder,
  interceptOpenExternal,
  launchApp,
  newMind,
  placeAssistant,
  pointAndClick,
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

const WORDS = {
  en: {
    title: "Choose your AI",
    chatgpt: "Plus or Pro · no key needed",
    sends: "Sends your questions and document text to OpenAI. · Details",
    key: "OpenAI, Anthropic, Google, DeepSeek and more",
  },
  "zh-CN": {
    title: "选择你的 AI",
    chatgpt: "Plus 或 Pro（中国大陆不可用）· 无需密钥",
    sends: "会把你的问题和文档内容发送给 OpenAI。 · 详情",
    key: "OpenAI、Anthropic、Google、DeepSeek 等",
  },
} as const;

/** One line of text: no taller than its line height, with room for rounding. */
async function expectOneLine(line: Locator): Promise<void> {
  const [box, lineHeight] = await Promise.all([
    line.boundingBox(),
    line.evaluate((element) => Number.parseFloat(getComputedStyle(element).lineHeight)),
  ]);
  if (!box) throw new Error("The line isn't laid out.");
  expect(box.height).toBeLessThan(lineHeight * 1.5);
}

for (const [width, height] of [
  [1100, 760],
  [1440, 900],
] as const) {
  test(`at ${width} px the card offers the ChatGPT plan and Paste an API key at the same size, each with a short line under it; in Chinese the key comes first`, async () => {
    const { app, window } = await launchApp(dataDir);
    try {
      await setWindowSize(app, window, width, height);
      await acceptPlainTextKeys(window);
      const card = await askFirstQuestion(window);
      for (const language of ["en", "zh-CN"] as const) {
        await setLanguage(window, language);
        const plan = card.getByTestId("setup-chatgpt");
        const key = card.getByTestId("setup-key");
        const words = WORDS[language];
        await expect(card.getByRole("heading", { level: 2 })).toHaveText(words.title);
        await expect(card.getByTestId("setup-chatgpt-line")).toHaveText(words.chatgpt);
        await expect(card.getByTestId("setup-chatgpt-sends").locator("p")).toHaveText(words.sends);
        await expect(card.getByTestId("setup-key-for")).toHaveText(words.key);
        if (width === 1440) {
          await expectOneLine(card.getByTestId("setup-chatgpt-line"));
          await expectOneLine(card.getByTestId("setup-key-for"));
        }
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

for (const language of ["en", "zh-CN"] as const) {
  test(`after narrowing the window, Claude's notice, Details and key form remain reachable (${language})`, async () => {
    const { app, window } = await launchApp(dataDir);
    try {
      await setWindowSize(app, window, 1440, 760);
      await acceptPlainTextKeys(window);
      await setLanguage(window, language);
      const card = await askFirstQuestion(window);
      await pointAndClick(window, card.getByTestId("setup-other"));
      const row = card.getByTestId("way-claude");
      await pointAndClick(window, row.getByTestId("way-claude-open"));

      for (const width of [900, 1440]) {
        await setWindowSize(app, window, width, 760);
        const sends = row.getByTestId("claude-plan-sends");
        const details = sends.getByTestId("claude-plan-sends-details");
        await details.scrollIntoViewIfNeeded();
        await expect(details).toBeInViewport({ ratio: 1 });
        await pointAndClick(window, details);
        const items = sends.getByRole("listitem");
        await expect(items).toHaveCount(5);
        for (const item of [items.first(), items.last()]) {
          await item.scrollIntoViewIfNeeded();
          // Text can end a fraction of a pixel past the scroller's edge on Linux, still readable.
          await expect(item).toBeInViewport({ ratio: 0.99 });
        }
        const signIn = row.getByTestId("claude-plan-sign-in");
        await signIn.scrollIntoViewIfNeeded();
        await expect(signIn).toBeInViewport({ ratio: 1 });
        await details.press("Space");
        await expect(details).toHaveAttribute("aria-expanded", "false");
      }

      await setWindowSize(app, window, 900, 760);
      await row.getByTestId("claude-plan-cancel").press("Enter");
      await pointAndClick(window, card.getByTestId("setup-key"));
      const form = card.getByTestId("provider-form");
      for (const target of [form.getByLabel(/^API /), form.getByTestId("provider-use-key")]) {
        await target.scrollIntoViewIfNeeded();
        await expect(target).toBeInViewport({ ratio: 1 });
      }
      await card.getByTestId("setup-back").press("Enter");
      await expect(card.getByTestId("setup-key")).toBeFocused();
      await expect(card.getByTestId("setup-key")).toBeInViewport({ ratio: 1 });
    } finally {
      await app.close();
    }
  });
}

/** The smallest window the app allows (`src/main/index.ts`). */
const SMALLEST_WINDOW = { width: 900, height: 560 };

async function expectInsideCardAndWindow(window: Page, card: Locator, target: Locator) {
  const [box, cardBox, page] = await Promise.all([
    target.boundingBox(),
    card.boundingBox(),
    window.evaluate(() => ({ width: globalThis.innerWidth, height: globalThis.innerHeight })),
  ]);
  if (!box || !cardBox) throw new Error("The card and its control aren't laid out.");
  expect(cardBox.y).toBeGreaterThanOrEqual(0);
  expect(cardBox.y + cardBox.height).toBeLessThanOrEqual(page.height);
  expect(box.y).toBeGreaterThanOrEqual(cardBox.y);
  expect(box.y + box.height).toBeLessThanOrEqual(cardBox.y + cardBox.height + 0.5);
  expect(box.x).toBeGreaterThanOrEqual(cardBox.x);
  expect(box.x + box.width).toBeLessThanOrEqual(cardBox.x + cardBox.width + 0.5);
}

for (const dock of ["bottom", "float"] as const) {
  for (const language of ["en", "zh-CN"] as const) {
    test(`at 900 px with the assistant ${dock === "bottom" ? "docked at the bottom" : "floating"}, the card scrolls inside itself and the last control under Other ways is reached by wheel and by Tab (${language})`, async () => {
      const { app, window } = await launchApp(dataDir);
      try {
        await setWindowSize(app, window, SMALLEST_WINDOW.width, SMALLEST_WINDOW.height);
        await acceptPlainTextKeys(window);
        await setLanguage(window, language);
        const card = await askFirstQuestion(window);
        await placeAssistant(window, { dock });
        await pointAndClick(window, card.getByTestId("setup-other"));
        const last = card.locator("details").locator("button, a").last();

        for (const first of [
          card.getByRole("heading", { level: 2 }),
          card.getByTestId("setup-chatgpt"),
          card.getByTestId("setup-key"),
        ]) {
          await expectInsideCardAndWindow(window, card, first);
        }

        const cardBox = await card.boundingBox();
        if (!cardBox) throw new Error("The card isn't laid out.");
        await window.mouse.move(cardBox.x + cardBox.width / 2, cardBox.y + cardBox.height / 2, {
          steps: 8,
        });
        await expect
          .poll(async () => {
            await window.mouse.wheel(0, 120);
            const box = await last.boundingBox();
            return box !== null && box.y + box.height <= cardBox.y + cardBox.height + 0.5;
          })
          .toBe(true);
        await expectInsideCardAndWindow(window, card, last);
        await expect(card.getByTestId("setup-later")).toBeInViewport({ ratio: 1 });

        await card.evaluate((element) => {
          element.scrollTop = 0;
        });
        await card.locator("[data-setup-choice]:not(:disabled)").first().focus();
        for (let tabs = 0; tabs < 12; tabs++) {
          await window.keyboard.press("Tab");
          const focused = window.locator(":focus");
          await expectInsideCardAndWindow(window, card, focused);
          if (await last.evaluate((element) => element === document.activeElement)) break;
        }
        await expect(last).toBeFocused();
      } finally {
        await app.close();
      }
    });
  }
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
    await expect(model.locator("option").first()).toHaveText("Claude Haiku 5.5 · Recommended");
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

test("the key step follows the company picked and Back returns keyboard focus to Paste an API key, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    const card = await askFirstQuestion(window);
    const title = card.getByTestId("provider-form-title");
    const form = card.getByTestId("provider-form");
    const chip = (name: string) =>
      form.locator("label").filter({ hasText: new RegExp(`^${name}$`) });

    await card.getByTestId("setup-key").click();
    await expect(title).toHaveText("Paste your Anthropic key");
    await chip("OpenAI").click();
    await expect(title).toHaveText("Paste your OpenAI key");
    const model = form.getByTestId("provider-model");
    await model.selectOption({ label: "Another model…" });
    await expect(form.getByTestId("provider-model-text")).toHaveAttribute(
      "placeholder",
      "e.g. gpt-6-luna",
    );
    // The list stays, so a model in it can still be picked instead.
    await model.selectOption({ index: 0 });
    await expect(form.getByTestId("provider-model-text")).toHaveCount(0);

    await card.getByTestId("setup-back").focus();
    await window.keyboard.press("Enter");
    await expect(card.getByTestId("setup-key")).toBeFocused();
    await setLanguage(window, "zh-CN");
    await card.getByTestId("setup-key").click();
    // In Chinese the key step opens on DeepSeek, and says so until another company is picked.
    await expect(title).toHaveText("粘贴 DeepSeek 密钥");
    await chip("Anthropic").click();
    await expect(title).toHaveText("粘贴 Anthropic 密钥");
    await model.selectOption({ label: "其他模型…" });
    await expect(form.getByTestId("provider-model-text")).toHaveAttribute(
      "placeholder",
      "例如 claude-haiku-5-5",
    );
    await card.getByTestId("setup-back").focus();
    await window.keyboard.press("Enter");
    await expect(card.getByTestId("setup-key")).toBeFocused();
    await card.getByTestId("setup-other").press("Enter");
    await card.getByTestId("way-local").press("Enter");
    await card.getByTestId("setup-back").press("Enter");
    await expect(card.locator("details")).toHaveAttribute("open", "");
    await expect(card.getByTestId("way-local")).toBeFocused();
    await expect(card.getByTestId("way-local")).toBeInViewport({ ratio: 1 });
  } finally {
    await app.close();
  }
});
