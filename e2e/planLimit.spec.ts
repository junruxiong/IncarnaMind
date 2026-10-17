import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import { ANSWER, createFakeOpenAI, type FakeOpenAIServer } from "../tests/helpers/fakeOpenAI";
import {
  assistantAnswers,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  pointAndClick,
  pointAt,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  signInWithChatGpt,
} from "./app";

/*
 * On a ChatGPT plan, a model whose usage limit is used up (#551): the Problem
 * line under the Answer names it and when it resets, and offers the plan's
 * next model with limit left; one click asks the same Question again in place
 * with it, and the model chip says which model answers, until when. The
 * limits last the session: reopened, the app says "for now". Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots, narrow and
 * wide, in English and Chinese.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const NARROW = { width: 900, height: 720 };
const WIDE = { width: 1600, height: 720 };

const SOL = "gpt-6.1-sol";
const LUNA = "gpt-6-luna";
const ASTRA = "gpt-6-astra";

let dataDir: string;
let home: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  home = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(home);
});

/** The plan refuses `model` for its usage limit, which resets in `hours`. */
function usedUp(openai: FakeOpenAIServer, model: string, hours: number) {
  openai.replyFor[model] = {
    kind: "error",
    status: 429,
    body: {
      error: {
        code: "subscription_sharing_usage_limit_exceeded",
        message: "Limit reached.",
        resets_at: Math.round(Date.now() / 1000 + hours * 3600),
      },
    },
  };
}

async function expectChipWhole(window: Page) {
  const assistant = window.getByTestId("assistant");
  const fits = (element: HTMLElement | SVGElement) => element.scrollWidth <= element.clientWidth;
  expect(await assistant.getByTestId("composer-model").locator("span").evaluate(fits)).toBe(true);
  const scope = assistant.getByTestId("composer-scope-all");
  expect(await scope.evaluate(fits)).toBe(true);
  const scopeBox = await scope.boundingBox();
  const askBox = await assistant.getByTestId("composer-ask").boundingBox();
  if (!scopeBox || !askBox) throw new Error("The composer's controls aren't visible.");
  expect(scopeBox.x + scopeBox.width).toBeLessThanOrEqual(askBox.x);
}

async function shot(window: Page, name: string) {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

/** Shots of the assistant now, narrow and wide, ending wide. */
async function shots(app: Parameters<typeof setWindowSize>[0], window: Page, name: string) {
  if (!SCREENSHOTS) return;
  await setWindowSize(app, window, NARROW.width, NARROW.height);
  await shot(window, `${name}-narrow`);
  await setWindowSize(app, window, WIDE.width, WIDE.height);
  await shot(window, `${name}-wide`);
}

async function askAndFail(window: Page, question: string): Promise<Locator> {
  const answers = assistantAnswers(window);
  const before = await answers.count();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type(question);
  await window.keyboard.press("Enter");
  await expect(answers).toHaveCount(before + 1);
  await expect(answers.last()).toHaveAttribute("data-status", "failed");
  return answers.last();
}

async function resetTime(line: Locator, pattern: RegExp): Promise<string> {
  await expect(line).toHaveText(pattern);
  return pattern.exec((await line.textContent()) ?? "")?.[1] ?? "";
}

test("a used-up plan model offers the next one in one click, which answers in place and holds the chip until the reset, in English and Chinese; reopened, it says for now", async () => {
  test.setTimeout(120_000);
  const openai = await createFakeOpenAI();
  openai.modelList = [
    { slug: SOL, display_name: "GPT-6.1 Sol", visibility: "list" },
    { slug: LUNA, display_name: "GPT-6 Luna", visibility: "list" },
    { slug: ASTRA, display_name: "GPT-6 Astra", visibility: "list" },
  ];
  let { app, window } = await launchApp(dataDir, { chatGpt: openai.endpoints.issuer, home });
  try {
    await setWindowSize(app, window, WIDE.width, WIDE.height);
    await dismissChatSetup(window);
    await signInWithChatGpt(app, window, openai);
    const chip = window.getByTestId("assistant").getByTestId("composer-model");
    await expect(chip).toHaveText("GPT-6.1 Sol");

    usedUp(openai, SOL, 2);
    const answer = await askAndFail(window, "What did the committee decide?");
    const line = answer.getByTestId("answer-error");
    const time = await resetTime(
      line,
      /^GPT-6\.1 Sol’s limit on your plan is used up until (.+)\. · Answer with GPT-6 Luna$/,
    );
    await shots(app, window, "limit-en");
    await setLanguage(window, "zh-CN");
    // A reset on another day names the day, which each language writes its own way.
    const zhTime = await resetTime(
      line,
      /^你的套餐里 GPT-6\.1 Sol 的额度已用完，(.+) 恢复。· 改用 GPT-6 Luna 回答$/,
    );
    await shots(app, window, "limit-zh");
    await setLanguage(window, "en");
    await expect(chip).toHaveText("GPT-6.1 Sol");

    await pointAndClick(window, line.getByTestId("answer-error-action"));
    await expect(assistantAnswers(window)).toHaveCount(1);
    await expect(answer).toHaveAttribute("data-status", "done");
    await expect(answer).toContainText(ANSWER);
    expect(openai.responses().at(-1)?.body.model).toBe(LUNA);
    await expect(chip).toHaveText(`GPT-6 Luna · until ${time}`);
    await expect(chip).toHaveAttribute(
      "title",
      `GPT-6.1 Sol’s limit is used up until ${time}. Questions go to GPT-6 Luna.`,
    );
    await expectChipWhole(window);
    await shots(app, window, "answered-en");
    await setLanguage(window, "zh-CN");
    await expect(chip).toHaveText(`GPT-6 Luna · ${zhTime} 前`);
    await expectChipWhole(window);
    await shots(app, window, "answered-zh");

    usedUp(openai, LUNA, 1);
    const second = await askAndFail(window, "什么时候上线？");
    const secondLine = second.getByTestId("answer-error");
    const lunaTime = await resetTime(
      secondLine,
      /^你的套餐里 GPT-6 Luna 的额度已用完，(.+) 恢复。· 改用 GPT-6 Astra 回答$/,
    );
    const action = await pointAt(window, secondLine.getByTestId("answer-error-action"));
    await window.mouse.dblclick(action.x, action.y);
    await expect(second).toHaveAttribute("data-status", "done");
    await expect(second).toContainText(ANSWER);
    expect(openai.responses().filter((request) => request.body.model === ASTRA)).toHaveLength(1);
    // Astra stands in for Sol, the model the User chose, until Sol's reset.
    await expect(chip).toHaveText(`GPT-6 Astra · ${zhTime} 前`);

    // None left: Luna's limit resets first.
    usedUp(openai, ASTRA, 3);
    const third = await askAndFail(window, "谁负责？");
    await expect(third.getByTestId("answer-error")).toHaveText(
      `你的套餐额度已全部用完，${lunaTime} 恢复。`,
    );
    await expect(third.getByTestId("answer-error-action")).toHaveCount(0);
    await shots(app, window, "none-left-zh");

    // Quit and reopen: the limits are forgotten, so the stopped Answer says "for now" and offers
    // the plan's first model again, which answers.
    const conversation = (await window.getByTestId("assistant-title").textContent()) ?? "";
    await app.close();
    ({ app, window } = await launchApp(dataDir, { chatGpt: openai.endpoints.issuer, home }));
    await setWindowSize(app, window, WIDE.width, WIDE.height);
    await window.keyboard.press("ControlOrMeta+k");
    await window.keyboard.type(conversation);
    await pointAndClick(
      window,
      window.getByTestId("palette-item").filter({ hasText: conversation }).first(),
    );
    const reopened = assistantAnswers(window).last();
    const reopenedLine = reopened.getByTestId("answer-error");
    await expect(reopenedLine).toHaveText(
      "你的套餐里 GPT-6 Astra 的额度暂时已用完。· 改用 GPT-6.1 Sol 回答",
    );
    openai.replyFor = {};
    await pointAndClick(window, reopenedLine.getByTestId("answer-error-action"));
    await expect(reopened).toHaveAttribute("data-status", "done");
    await expect(reopened).toContainText(ANSWER);
    expect(openai.responses().at(-1)?.body.model).toBe(SOL);
    await expect(window.getByTestId("assistant").getByTestId("composer-model")).toHaveText(
      "GPT-6.1 Sol",
    );
  } finally {
    await app.close();
    await openai.close();
  }
});
