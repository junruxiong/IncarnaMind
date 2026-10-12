import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { createFakeOpenAI, signInAt } from "../tests/helpers/fakeOpenAI";
import {
  acceptPlainTextKeys,
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  interceptOpenExternal,
  launchApp,
  newMind,
  openSettings,
  removeDataFolder,
  setWindowSize,
  urlsOpened,
} from "./app";

/*
 * Cloud cost (#83): Settings → General shows what the User's keys spent this
 * month and holds one optional cap, off by default; Settings → Models shows
 * each key's month and the ChatGPT plan's own line; an Answer the cap stops
 * says so in place, and "Change cap" opens the cap. Set INCARNAMIND_SCREENSHOTS
 * to a folder to also save screenshots, narrow and wide, in English and Chinese.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const NARROW = { width: 900, height: 720 };
const WIDE = { width: 1440, height: 860 };

async function shot(window: Page, name: string) {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const bridge = (window: Page) => ({
  language: (language: "en" | "zh-CN") =>
    window.evaluate(
      (value) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
          user: { language: value },
        }),
      language,
    ),
  /** Claude Haiku 5.5 on an Anthropic key, billed per token, with Questions to Anthropic allowed. */
  keyProvider: () =>
    window.evaluate(async () => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const provider = await core.saveChatProvider({
        kind: "anthropic",
        apiKey: "sk-ant-test",
        modelId: "claude-haiku-5-5",
      });
      if (provider.service) await core.allowDataFlow("chat", provider.service.id);
    }),
  cap: () =>
    window.evaluate(
      async () =>
        (await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.getSettings())
          .user.monthlyCap,
    ),
});

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("an Answer stopped by the monthly cap says so in place, and Change cap opens the cap; set by mouse and keyboard, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await dismissChatSetup(window);
    await setWindowSize(app, window, WIDE.width, WIDE.height);
    await acceptPlainTextKeys(window);
    await bridge(window).keyProvider();

    // Off by default; turned on, it starts at a first cap that can be changed in place.
    const settings = await openSettings(window, "general");
    const cap = settings.getByTestId("spending-cap-settings");
    await expect(cap).toBeVisible();
    await expect(cap.getByTestId("spending-cap-spent")).toHaveText(
      "Nothing spent on your keys this month.",
    );
    const toggle = cap.getByTestId("spending-cap-switch");
    await expect(toggle).not.toBeChecked();
    expect(await bridge(window).cap()).toBeNull();
    await toggle.hover();
    await toggle.click();
    const amount = cap.getByTestId("spending-cap-amount");
    await expect(amount).toHaveValue("20");
    await expect(cap.getByTestId("spending-cap-spent")).toHaveText(
      "Spent this month: $0 of $20.00 on your keys.",
    );
    await shot(window, "cap-control-wide-en");

    // Esc puts a changed amount back without closing Settings; Enter saves.
    await amount.fill("5");
    await window.keyboard.press("Escape");
    await expect(amount).toHaveValue("20");
    await expect(settings).toBeVisible();
    await amount.fill("0");
    await window.keyboard.press("Enter");
    await expect.poll(() => bridge(window).cap()).toEqual({ amount: 0, currency: "USD" });
    await closeSettings(window);

    // Asked, the Answer is stopped before anything is sent, with one line and one action.
    await newMind(window);
    await window.getByTestId("mind-editor").click();
    await window.keyboard.press("ControlOrMeta+j");
    await window.keyboard.type("What is in my Documents?");
    await window.keyboard.press("Enter");
    const answer = window.getByTestId("answer");
    await expect(answer).toHaveAttribute("data-status", "failed");
    const error = answer.getByTestId("answer-error");
    await expect(error).toHaveAttribute("data-error-kind", "cost-cap");
    const line = error.getByRole("alert");
    await expect(line).toContainText("Monthly cap reached");
    await expect(line.getByRole("button")).toHaveText("Change cap");
    await shot(window, "cap-problem-wide-en");
    await setWindowSize(app, window, NARROW.width, NARROW.height);
    // The line keeps to one line, under the Answer, at a narrow window too.
    expect((await line.boundingBox())?.height).toBeLessThan(24);
    await shot(window, "cap-problem-narrow-en");

    // Change cap opens the cap, where it can be raised.
    await error.getByTestId("answer-error-action").click();
    await expect(settings).toHaveAttribute("data-page", "general");
    await expect(cap).toBeInViewport();
    await shot(window, "cap-control-narrow-en");
    await closeSettings(window);

    // The same in Chinese.
    await bridge(window).language("zh-CN");
    await expect(line).toContainText("已达到每月花费上限");
    await expect(line.getByRole("button")).toHaveText("更改上限");
    await shot(window, "cap-problem-narrow-zh");
    await setWindowSize(app, window, WIDE.width, WIDE.height);
    await shot(window, "cap-problem-wide-zh");
    await error.getByTestId("answer-error-action").click();
    await expect(cap.getByTestId("spending-cap-spent")).toHaveText(
      "本月 API 密钥已花费：US$0，上限 US$0。",
    );
    await shot(window, "cap-control-wide-zh");
    await setWindowSize(app, window, NARROW.width, NARROW.height);
    await shot(window, "cap-control-narrow-zh");
  } finally {
    await app.close();
  }
});

test("Settings → Models shows what a key spent this month, and the ChatGPT plan keeps its own line", async () => {
  const openai = await createFakeOpenAI();
  const { app, window } = await launchApp(dataDir, {
    fakeChat: true,
    chatGpt: openai.endpoints.issuer,
  });
  try {
    await dismissChatSetup(window);
    await setWindowSize(app, window, WIDE.width, WIDE.height);
    await acceptPlainTextKeys(window);
    await bridge(window).keyProvider();

    // One Answer on the key: its month shows under the provider.
    await newMind(window);
    await window.getByTestId("mind-editor").click();
    await window.keyboard.press("ControlOrMeta+j");
    await window.keyboard.type("What is in my Documents?");
    await window.keyboard.press("Enter");
    await expect(window.getByTestId("answer")).toHaveAttribute("data-status", "done");
    const settings = await openSettings(window, "models");
    const cost = settings.getByTestId("chat-provider-cost");
    await expect(cost).toHaveText("This month: $0");
    await shot(window, "models-key-wide-en");
    await setWindowSize(app, window, NARROW.width, NARROW.height);
    await shot(window, "models-key-narrow-en");
    await bridge(window).language("zh-CN");
    await expect(cost).toHaveText("本月：US$0");
    await shot(window, "models-key-narrow-zh");
    await bridge(window).language("en");

    // Signed in with ChatGPT and using the plan: its line names the plan, never a price.
    await interceptOpenExternal(app);
    const plan = settings.getByTestId("chatgpt-settings");
    await plan.getByTestId("chatgpt-continue").click();
    await expect.poll(() => urlsOpened(app)).toHaveLength(1);
    const [authorize] = await urlsOpened(app);
    expect((await signInAt(openai, authorize ?? "")).status).toBe(200);
    await window.getByTestId("chatgpt-welcome-ok").click();
    await plan.getByTestId("chatgpt-use").click();
    await expect(plan.getByTestId("chatgpt-in-use")).toHaveText(
      "Using ChatGPT plan · Manage usage ↗",
    );
    await expect(settings.getByTestId("chat-provider-current")).toHaveText("ChatGPT plan");
    await expect(settings.getByTestId("chat-provider-cost")).toHaveCount(0);
    await plan.scrollIntoViewIfNeeded();
    await shot(window, "models-plan-narrow-en");
    await setWindowSize(app, window, WIDE.width, WIDE.height);
    await shot(window, "models-plan-wide-en");
    await bridge(window).language("zh-CN");
    await expect(plan.getByTestId("chatgpt-in-use")).toContainText("ChatGPT");
    await shot(window, "models-plan-wide-zh");
  } finally {
    await app.close();
    await openai.close();
  }
});
