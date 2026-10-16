import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { createFakeGitHub, type FakeGitHub, LOGIN, USER_CODE } from "../tests/helpers/fakeGitHub";
import {
  acceptPlainTextKeys,
  assistantAnswers,
  catchCopies,
  copied,
  createDataFolder,
  interceptOpenExternal,
  launchApp,
  newMind,
  openSettings,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  urlsOpened,
} from "./app";

/*
 * Using a GitHub Copilot plan (#876): first run's Other ways and Settings › Models list it, off
 * until the User signs in; its notice, the usage line first, is said before anything is sent;
 * GitHub's code waits in large type with Copy code; an organization's block says so with its ways
 * back. A local stand-in for GitHub signs in and answers. Set INCARNAMIND_SCREENSHOTS to a folder
 * to also save screenshots there.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(target: Page | Locator, name: string): Promise<void> {
  if (SCREENSHOTS) await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

let dataDir: string;
let github: FakeGitHub;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  github = await createFakeGitHub();
});
test.afterEach(async () => {
  await github.close();
  await removeDataFolder(dataDir);
});

const copilotAccount = (window: Page) =>
  window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return (await bridge.getCopilot()).account;
  });

async function askFirstQuestion(window: Page): Promise<Locator> {
  await newMind(window);
  await window.getByTestId("mind-editor").click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("What does my plan cover?");
  await window.keyboard.press("Enter");
  const card = window.getByTestId("model-setup");
  await expect(card).toBeVisible();
  await pointAndClick(window, card.getByTestId("setup-other"));
  return card;
}

test("first run: Use my Copilot plan says what it costs and sends before anything is sent, shows GitHub's code, and Cancel keeps nothing; signed in, the Question is answered", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, copilot: github.origin });
  await interceptOpenExternal(app);
  await setWindowSize(app, window, 1100, 900);
  await acceptPlainTextKeys(window);
  await catchCopies(window);
  const card = await askFirstQuestion(window);

  const row = card.getByTestId("way-copilot");
  await expect(row).toContainText("Use my Copilot plan");
  await expect(row).toContainText("Pro, Pro+, Business or Enterprise · Sign in with GitHub");
  await expect(row).toContainText("Experimental");
  await expect(row).toHaveAttribute("aria-expanded", "false");
  await pointAndClick(window, row);

  const notice = card.getByTestId("copilot-notice");
  await expect(notice.getByRole("listitem")).toHaveText([
    "Answers count toward your Copilot plan's usage limits.",
    "This way of signing in is experimental: GitHub may refuse or block it.",
    "Your Questions, the Mind you're in, Passages from your Documents, and Document text for Tags go to GitHub under your account.",
    "It stays off until you sign in here, and you can sign out any time in Settings › Models.",
  ]);
  // The experimental way starts on the safe choice.
  await expect(notice.getByTestId("copilot-cancel")).toBeFocused();
  expect(github.requests).toEqual([]);
  await screenshot(card, "setup-copilot-notice");

  await pointAndClick(window, notice.getByTestId("copilot-sign-in"));
  const waiting = card.getByTestId("copilot-waiting");
  await expect(waiting.getByTestId("copilot-code")).toHaveText(USER_CODE);
  await expect(waiting).toContainText("Enter this code on GitHub's page");
  await expect(waiting).toContainText("Waiting for you to approve it in the browser…");
  await expect(waiting).toContainText("Answers count toward your Copilot plan's usage limits.");
  await expect.poll(() => urlsOpened(app)).toEqual([`${github.origin}/login/device`]);
  await pointAndClick(window, waiting.getByTestId("copilot-copy-code"));
  expect(await copied(window)).toBe(USER_CODE);
  await pointAndClick(window, waiting.getByTestId("copilot-reopen"));
  await expect.poll(() => urlsOpened(app)).toHaveLength(2);
  await screenshot(card, "setup-copilot-code");

  await pointAndClick(window, waiting.getByTestId("copilot-cancel"));
  await expect(card.getByTestId("copilot-notice")).toBeVisible();
  expect(await copilotAccount(window)).toEqual({ state: "signed-out" });

  await pointAndClick(window, card.getByTestId("copilot-sign-in"));
  await expect(card.getByTestId("copilot-code")).toHaveText(USER_CODE);
  github.approve();
  await expect(card).toBeHidden();
  await expect(assistantAnswers(window).first()).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  expect(await copilotAccount(window)).toMatchObject({ state: "signed-in", login: LOGIN });
  await app.close();
});

test("first run: an organization's block says so, with an API key or another way, in English and Chinese", async () => {
  github.exchangePolicy = true;
  const { app, window } = await launchApp(dataDir, { copilot: github.origin });
  await interceptOpenExternal(app);
  await setWindowSize(app, window, 1100, 900);
  await acceptPlainTextKeys(window);
  const card = await askFirstQuestion(window);
  await pointAndClick(window, card.getByTestId("way-copilot"));
  await pointAndClick(window, card.getByTestId("copilot-sign-in"));
  await expect(card.getByTestId("copilot-code")).toBeVisible();
  github.approve();

  const failure = card.getByTestId("copilot-failure");
  await expect(failure).toContainText("Your organization doesn't allow this app");
  await expect(failure).toContainText(
    "Your organization's Copilot policy blocks IncarnaMind. Ask an admin to allow it, or choose another way.",
  );
  await expect(failure.getByRole("button")).toHaveText([
    "Use an API key instead",
    "Choose another way",
  ]);
  expect(await copilotAccount(window)).toEqual({ state: "signed-out" });
  await screenshot(card, "setup-copilot-blocked");

  await setLanguage(window, "zh-CN");
  await expect(failure).toContainText("你所在的组织不允许使用此应用");
  await expect(failure.getByRole("button")).toHaveText(["改用 API 密钥", "换一种方式"]);
  await screenshot(card, "setup-copilot-blocked-zh");
  await setLanguage(window, "en");

  await pointAndClick(window, failure.getByTestId("copilot-way-back-paste-api-key"));
  await expect(card.getByTestId("provider-form")).toBeVisible();
  await app.close();
});

test("Settings › Models: the Copilot plan opens its notice before signing in, then names the plan and account, with Sign out", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, copilot: github.origin });
  await interceptOpenExternal(app);
  await setWindowSize(app, window, 1100, 800);
  await acceptPlainTextKeys(window);
  const settings = await openSettings(window, "models");
  const row = settings.getByTestId("plans-copilot");
  await expect(row).toContainText("GitHub Copilot Pro, Pro+, Business or Enterprise");
  await expect(row).toContainText(
    "Sign in with GitHub. Answers count toward your plan's usage limits.",
  );

  await pointAndClick(window, row.getByTestId("plans-copilot-sign-in"));
  await expect(row.getByTestId("copilot-notice-usage")).toHaveText(
    "Answers count toward your Copilot plan's usage limits.",
  );
  expect(github.requests).toEqual([]);
  await pointAndClick(window, row.getByTestId("copilot-sign-in"));
  await expect(row.getByTestId("copilot-code")).toHaveText(USER_CODE);
  github.approve();

  const account = row.getByTestId("plans-copilot-account");
  await expect(account).toHaveText(`Copilot Pro+ · ${LOGIN}`);
  await expect(row).toContainText("Answers count toward your plan's usage limits.");
  const signOut = row.getByTestId("plans-copilot-sign-out");
  const [accountBox, signOutBox] = await Promise.all([
    account.boundingBox(),
    signOut.boundingBox(),
  ]);
  if (!accountBox || !signOutBox) throw new Error("The row isn't laid out.");
  expect(signOutBox.x).toBeGreaterThan(accountBox.x + accountBox.width);
  await screenshot(row, "settings-copilot-signed-in");
  await setWindowSize(app, window, 1440, 900);
  await expect(signOut).toBeInViewport();
  await screenshot(row, "settings-copilot-signed-in-wide");

  await setLanguage(window, "zh-CN");
  await expect(row).toContainText("GitHub Copilot Pro、Pro+、Business 或 Enterprise");
  await expect(row).toContainText("回答会计入订阅的用量限额。");
  await screenshot(row, "settings-copilot-signed-in-zh");
  await setLanguage(window, "en");

  await pointAndClick(window, signOut);
  await expect(row.getByTestId("plans-copilot-sign-in")).toBeVisible();
  await expect(row).not.toContainText(LOGIN);
  expect(await copilotAccount(window)).toEqual({ state: "signed-out" });
  await app.close();
});
