import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import type { ClaudeAccountStatus, CoreBridge } from "../src/core/api";
import { ANSWER, createFakeAnthropic, type FakeAnthropic } from "../tests/helpers/fakeAnthropic";
import {
  acceptPlainTextKeys,
  assistantAnswers,
  closeSettings,
  createDataFolder,
  interceptOpenExternal,
  launchApp,
  newMind,
  openSettings,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  urlsOpened,
} from "./app";

/*
 * "Use my Claude plan" (#872): a row under Other ways in first run and a row in Settings ›
 * Models. A local stand-in for Anthropic signs in and answers; the browser is stood in for, and
 * the User pastes the code Anthropic's page shows, as they do.
 */

const USAGE = "Answers count toward your Claude plan's usage limits.";

let dataDir: string;
let fake: FakeAnthropic;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  fake = await createFakeAnthropic();
});
test.afterEach(async () => {
  await fake.close();
  await removeDataFolder(dataDir);
});

const account = (window: Page): Promise<ClaudeAccountStatus> =>
  window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.getClaudeAccount(),
  );

async function askFirstQuestion(window: Page): Promise<Locator> {
  await newMind(window);
  await window.getByTestId("mind-editor").click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("What is in my Documents?");
  await window.keyboard.press("Enter");
  const card = window.getByTestId("model-setup");
  await expect(card).toBeVisible();
  return card;
}

/** The code Anthropic's page shows once the User approves the sign-in the app opened. */
async function codeAfterApproving(app: ElectronApplication, count: number): Promise<string> {
  await expect.poll(() => urlsOpened(app)).toHaveLength(count);
  const authorize = (await urlsOpened(app))[count - 1] ?? "";
  const { code, state } = fake.approve(authorize);
  return `${code}#${state}`;
}

/** Each box sits inside the card, so nothing is cut off at its edge. */
async function expectInside(card: Locator, parts: Locator[]) {
  const outer = await card.boundingBox();
  if (!outer) throw new Error("The card isn't laid out.");
  for (const part of parts) {
    const box = await part.boundingBox();
    if (!box) throw new Error("A part of the row isn't laid out.");
    expect(box.x).toBeGreaterThanOrEqual(outer.x);
    expect(box.x + box.width).toBeLessThanOrEqual(outer.x + outer.width + 0.5);
  }
}

test("first run: the row is first under Other ways, says the usage limits first before anything is sent, and the pasted code signs in and answers", async () => {
  const { app, window } = await launchApp(dataDir, { claude: fake.origin });
  try {
    await setWindowSize(app, window, 1100, 660);
    await interceptOpenExternal(app);
    await acceptPlainTextKeys(window);
    const card = await askFirstQuestion(window);

    await card.getByTestId("setup-other").click();
    const ways = card.locator("[data-testid^='way-']:not([data-testid$='-open'])");
    await expect(ways.first()).toHaveAttribute("data-testid", "way-claude");
    const row = card.getByTestId("way-claude");
    await expect(row.getByText("Use my Claude plan", { exact: true })).toBeVisible();
    await expect(
      row.getByText("Pro, Max or Team · Sign in with your Claude account", { exact: true }),
    ).toBeVisible();
    await expect(row.getByTestId("claude-plan-notice")).toHaveCount(0);

    await row.getByTestId("way-claude-open").click();
    const notice = row.getByTestId("claude-plan-notice");
    const lines = notice.getByRole("listitem");
    await expect(lines).toHaveCount(3);
    await expect(lines.first()).toHaveText(USAGE);
    await expect(lines.first()).toHaveCSS("font-weight", "600");
    await expect(lines.nth(1)).toHaveCSS("font-weight", "400");
    await expect(notice).not.toContainText("Claude Code");
    // The safe choice has the focus, and nothing has been sent.
    await expect(notice.getByTestId("claude-plan-cancel")).toBeFocused();
    expect(await urlsOpened(app)).toHaveLength(0);
    await expectInside(card, [lines.first(), lines.nth(1), lines.nth(2)]);

    await notice.getByTestId("claude-plan-cancel").click();
    await expect(notice).toBeHidden();
    await expect(row.getByTestId("way-claude-open")).toHaveAttribute("aria-expanded", "false");

    await row.getByTestId("way-claude-open").click();
    await row.getByTestId("claude-plan-sign-in").click();
    const code = await codeAfterApproving(app, 1);
    const waiting = row.getByTestId("claude-plan-waiting");
    await expect(waiting).toContainText(
      "Sign in to Claude in your browser, then paste the code it shows.",
    );
    await expect(waiting.getByTestId("claude-plan-usage")).toHaveText(USAGE);
    // The code is the way on, so its field is open, focused and named for what goes in it.
    const field = waiting.getByLabel("Code from Claude's page");
    await expect(field).toBeFocused();
    await expect(waiting.getByTestId("claude-paste-continue")).toHaveText("Finish signing in");

    await field.fill("not a code");
    await field.press("Enter");
    await expect(waiting.getByTestId("claude-paste-problem")).toHaveText(
      "That isn't the code from Claude's page.",
    );
    await field.fill(code);
    await waiting.getByTestId("claude-paste-continue").click();

    await expect(card).toBeHidden();
    const answer = assistantAnswers(window).first();
    await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
    await expect(answer).toContainText(ANSWER);
    expect(fake.messages.at(-1)?.headers.authorization).toMatch(/^Bearer /);
    expect((await account(window)).account).toMatchObject({ state: "signed-in" });
  } finally {
    await app.close();
  }
});

test("first run: a code Anthropic refuses says so in the row, with a key as the way back", async () => {
  const { app, window } = await launchApp(dataDir, { claude: fake.origin });
  try {
    await interceptOpenExternal(app);
    await acceptPlainTextKeys(window);
    const card = await askFirstQuestion(window);
    await card.getByTestId("setup-other").click();
    const row = card.getByTestId("way-claude");
    await row.getByTestId("way-claude-open").click();
    await row.getByTestId("claude-plan-sign-in").click();

    const code = await codeAfterApproving(app, 1);
    const [, state] = code.split("#");
    await row.getByTestId("claude-paste-address").fill(`not-the-code#${state}`);
    await row.getByTestId("claude-paste-continue").click();

    const error = row.getByTestId("claude-plan-error");
    await expect(error).toContainText("The Claude sign-in didn't work · Use an API key instead");
    await expect(row.getByTestId("claude-plan-sign-in")).toBeVisible();
    await row.getByTestId("claude-plan-use-key").click();
    await expect(card.getByTestId("setup-back")).toBeVisible();
    expect((await account(window)).account).toEqual({ state: "signed-out" });
  } finally {
    await app.close();
  }
});

test("Settings: Cancel keeps nothing; signed in, the row names the account and Sign out forgets it; expired, Sign in again", async () => {
  const { app, window } = await launchApp(dataDir, { claude: fake.origin });
  try {
    await setWindowSize(app, window, 1100, 660);
    await interceptOpenExternal(app);
    await acceptPlainTextKeys(window);
    const settings = await openSettings(window, "models");
    const row = settings.getByTestId("plans-claude-account");
    await expect(row.getByTestId("claude-plan-account")).toHaveText(
      `Sign in with your Claude account. ${USAGE}`,
    );

    await row.getByTestId("plans-claude-sign-in").click();
    await expect(row.getByTestId("claude-plan-usage")).toHaveCSS("font-weight", "600");
    await row.getByTestId("claude-plan-sign-in").click();
    await expect.poll(() => urlsOpened(app)).toHaveLength(1);
    await row.getByTestId("claude-plan-cancel").click();
    await expect(row.getByTestId("claude-plan-waiting")).toBeHidden();
    await expect(row.getByTestId("plans-claude-sign-in")).toBeVisible();
    await expect(row.getByTestId("claude-plan-error")).toHaveCount(0);
    expect((await account(window)).account).toEqual({ state: "signed-out" });
    expect(fake.tokenRequests).toHaveLength(0);

    // Its tokens are due for a refresh at once, which is refused: the sign-in has expired.
    fake.expiresIn = 60;
    await row.getByTestId("plans-claude-sign-in").click();
    await row.getByTestId("claude-plan-sign-in").click();
    const code = await codeAfterApproving(app, 2);
    await row.getByTestId("claude-paste-address").fill(code);
    await row.getByTestId("claude-paste-continue").click();
    await expect(row.getByTestId("claude-plan-account")).toHaveText(
      "Signed in as reader@example.test",
    );
    await expect(row.getByTestId("claude-plan-usage")).toHaveText(USAGE);
    await expect(row.getByTestId("claude-plan-sign-out")).toBeVisible();

    fake.refuseRefresh = "invalid_grant";
    await closeSettings(window);
    await newMind(window);
    await window.getByTestId("mind-editor").click();
    await window.keyboard.press("ControlOrMeta+j");
    await window.keyboard.type("What is in my Documents?");
    await window.keyboard.press("Enter");
    await expect.poll(async () => (await account(window)).account.state).toBe("expired");

    const again = await openSettings(window, "models");
    const expired = again.getByTestId("plans-claude-account");
    await expect(expired.getByTestId("claude-plan-account")).toHaveText(
      "Your Claude sign-in has expired",
    );
    fake.refuseRefresh = null;
    fake.expiresIn = 3600;
    await expired.getByTestId("plans-claude-sign-in").click();
    // Signing in again skips the words already read, straight to the browser.
    await expect(expired.getByTestId("claude-plan-notice")).toHaveCount(0);
    const second = await codeAfterApproving(app, 3);
    await expired.getByTestId("claude-paste-address").fill(second);
    await expired.getByTestId("claude-paste-continue").click();
    await expect(expired.getByTestId("claude-plan-sign-out")).toBeVisible();

    await expired.getByTestId("claude-plan-sign-out").click();
    await expect(expired.getByTestId("plans-claude-sign-in")).toBeVisible();
    expect((await account(window)).account).toEqual({ state: "signed-out" });
  } finally {
    await app.close();
  }
});

test("⌘K: Sign in with Claude opens the words in Settings › Models, before anything is sent", async () => {
  const { app, window } = await launchApp(dataDir, { claude: fake.origin });
  try {
    await interceptOpenExternal(app);
    await window.keyboard.press("ControlOrMeta+k");
    await expect(window.getByTestId("palette-search")).toBeFocused();
    await window.keyboard.type("> Sign in with Claude");
    const item = window.getByTestId("palette").getByTestId("palette-item").first();
    await expect(item).toHaveText(/Sign in with Claude/);
    await window.keyboard.press("Enter");

    const settings = window.getByTestId("settings");
    await expect(settings).toHaveAttribute("data-page", "models");
    const row = settings.getByTestId("plans-claude-account");
    await expect(row.getByTestId("claude-plan-usage")).toHaveText(USAGE);
    await expect(row.getByTestId("claude-plan-cancel")).toBeFocused();
    expect(await urlsOpened(app)).toHaveLength(0);
  } finally {
    await app.close();
  }
});

test("in Chinese: the row is first under Other ways too and a model on this computer last, with the usage limits first", async () => {
  const { app, window } = await launchApp(dataDir, { claude: fake.origin });
  try {
    await setWindowSize(app, window, 1100, 660);
    await interceptOpenExternal(app);
    await setLanguage(window, "zh-CN");
    await acceptPlainTextKeys(window);
    const card = await askFirstQuestion(window);
    await card.getByTestId("setup-other").click();
    const ways = card.locator("[data-testid^='way-']:not([data-testid$='-open'])");
    await expect(ways).toHaveCount(2);
    await expect(ways.nth(0)).toHaveAttribute("data-testid", "way-claude");
    await expect(ways.nth(1)).toHaveAttribute("data-testid", "way-local");

    const row = card.getByTestId("way-claude");
    await expect(row).toContainText("使用我的 Claude 订阅");
    await row.getByTestId("way-claude-open").click();
    const lines = row.getByTestId("claude-plan-notice").getByRole("listitem");
    await expect(lines.first()).toHaveText("回答会计入你 Claude 订阅的用量限额。");
    await expect(lines.first()).toHaveCSS("font-weight", "600");
    await expect(row.getByTestId("claude-plan-sign-in")).toHaveText("用 Claude 账号登录");
    await expectInside(card, [lines.first(), lines.nth(1), lines.nth(2)]);
    await row.getByTestId("claude-plan-sign-in").click();
    await expect(row.getByLabel("Claude 页面上的代码")).toBeFocused();
    await expect(row.getByTestId("claude-paste-continue")).toHaveText("完成登录");
  } finally {
    await app.close();
  }
});
