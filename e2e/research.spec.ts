import { readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  acceptPlainTextKeys,
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  newMind,
  ollamaListing,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * Research (#130): "Research…" in the assistant's composer takes a topic, and
 * a plan card opens in the tray above it, with the steps the composer's model
 * proposed, each edited in place, the sources, where it writes and an
 * estimate with the composer's model. Start makes it a Task whose progress
 * shows in the thread under its Question, as it searches the web and the
 * Library. A local server stands in for the web, and the scripted chat model
 * plans and searches. Set INCARNAMIND_SCREENSHOTS to a folder to also save
 * screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const fixture = (name: string) => readFileSync(join(__dirname, `../tests/fixtures/web/${name}`));
const MODEL = "claude-haiku-5-5";
const MODEL_NAME = "Claude Haiku 5.5";
const TOPIC = "how tea reached Europe";

let dataDir: string;
let home: string;
let sources: string;
let server: Server;
let port: number;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  home = await createDataFolder();
  sources = await createDataFolder();
  server = createServer((request, response) => {
    const page =
      request.url === "/tea"
        ? fixture("article.html")
        : request.url === "/members"
          ? fixture("login.html")
          : null;
    response.writeHead(page ? 200 : 404, { "Content-Type": "text/html; charset=utf-8" });
    response.end(page ?? "Not found");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  port = (server.address() as AddressInfo).port;
});
test.afterEach(async () => {
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
  await removeDataFolder(dataDir);
  await removeDataFolder(home);
  await removeDataFolder(sources);
});

/** An Anthropic key, Questions and automatic tagging allowed to go there, and web search on. */
async function useAnthropic(window: Page): Promise<void> {
  await acceptPlainTextKeys(window);
  await window.evaluate(async (modelId) => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const provider = await core.saveChatProvider({
      kind: "anthropic",
      apiKey: "sk-ant-test",
      modelId,
    });
    if (!provider.service) return;
    await core.allowDataFlow("chat", provider.service.id);
    await core.allowDataFlow("tagging", provider.service.id);
    await core.updateSettings({ user: { webSearch: true } });
  }, MODEL);
}

/** The tray sits joined above the composer, its width, and never over it. */
async function expectAboveComposer(window: Page, tray: Locator): Promise<void> {
  const [trayBox, composerBox] = [
    await tray.boundingBox(),
    await window.getByTestId("composer").boundingBox(),
  ];
  if (!trayBox || !composerBox) throw new Error("The tray or the composer isn't on screen.");
  expect(trayBox.y + trayBox.height).toBeLessThanOrEqual(composerBox.y + 1);
  expect(Math.abs(trayBox.width - composerBox.width)).toBeLessThan(2);
}

const valuesOf = (fields: Locator) =>
  fields.evaluateAll((each) => each.map((field) => (field as HTMLTextAreaElement).value));

/** "/" in the composer, "Research…", then the topic: the plan card opens. */
async function research(window: Page, topic: string): Promise<Locator> {
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("/res");
  await expect(window.getByTestId("slash-item-research")).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("composer-research")).toBeVisible();
  await window.keyboard.type(topic);
  await window.keyboard.press("Enter");
  const card = window.getByTestId("research-plan");
  await expect(card.getByTestId("research-plan-step")).toHaveCount(3);
  return card;
}

test("Research… plans in the tray: the steps edited in place, the sources, where it writes and an estimate with the composer's model; Start shows it searching under its Question", async () => {
  test.setTimeout(120_000);
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home, web: port });
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1600, 900);
  await useAnthropic(window);
  const note = join(sources, "Tea trade.txt");
  await writeFile(note, "Tea trade.\nDutch traders shipped tea to Amsterdam in 1610.\n");
  await addDocuments(window, [note]);
  await newMind(window);
  await window.keyboard.type("Credit memo");
  await screenshot(window, "research-before-en-1600");

  const card = await research(window, TOPIC);
  // The topic is the Question in the thread, with the plan below.
  const thread = window.getByTestId("assistant-thread");
  const question = thread.getByTestId("assistant-question").last();
  await expect(question).toContainText(TOPIC);
  await expect(question.getByTestId("assistant-question-research")).toHaveText("Research");
  await expect(thread.getByTestId("research-plan-line")).toHaveText(
    "Plan below. Edit it, then Start.",
  );
  await expectAboveComposer(window, window.getByTestId("task-tray"));
  await expect(card.getByTestId("research-plan-title")).toHaveText(
    `What do the sources say about ${TOPIC}?`,
  );
  const steps = card.getByTestId("research-plan-step");
  await expect.poll(() => valuesOf(steps)).toEqual([
    `Search the web and the Library for ${TOPIC}`,
    `Read what the sources say about ${TOPIC}`,
    "Compare what they found",
  ]);
  await expect(card.getByTestId("research-plan-write-step")).toHaveText(
    "Write a cited draft into “Credit memo”",
  );
  await expect(card.getByTestId("research-plan-web")).toHaveAttribute("aria-pressed", "true");
  await expect(card.getByTestId("research-plan-library")).toHaveAttribute("aria-pressed", "true");
  await expect(card.getByTestId("research-plan-estimate")).toHaveText(
    new RegExp(`^About \\$\\d+\\.\\d+ · 4–8 min · ${MODEL_NAME}$`),
  );
  await expect(window.getByTestId("composer-input")).toHaveAttribute(
    "placeholder",
    "Change the plan…",
  );
  await expect(card.getByTestId("research-plan-write-into")).toBeInViewport();
  await screenshot(window, "research-plan-en-1600");

  // A step is edited in place; Enter adds one after it, Backspace in an empty one takes it out.
  await pointAndClick(window, steps.nth(1));
  await window.keyboard.press("ControlOrMeta+a");
  await window.keyboard.type("Read the first traders' accounts");
  await window.keyboard.press("Enter");
  await expect(steps).toHaveCount(4);
  await expect(steps.nth(2)).toBeFocused();
  await window.keyboard.press("Backspace");
  await expect(steps).toHaveCount(3);
  await expect(steps.nth(1)).toBeFocused();
  await expect(card.getByTestId("research-plan-estimate")).toContainText("4–8 min");

  // The sources by keyboard; one at least.
  await card.getByTestId("research-plan-library").focus();
  await window.keyboard.press("Space");
  await expect(card.getByTestId("research-plan-library")).toHaveAttribute("aria-pressed", "false");
  await card.getByTestId("research-plan-web").focus();
  await window.keyboard.press("Space");
  await expect(card.getByTestId("research-plan-start")).toBeDisabled();
  await expect(card.getByTestId("research-plan-estimate")).toHaveText(
    "Choose the web, the Library or both",
  );
  await window.keyboard.press("Space");
  await card.getByTestId("research-plan-library").focus();
  await window.keyboard.press("Space");
  await expect(card.getByTestId("research-plan-start")).toBeEnabled();

  // Write into: another Mind or a new one, and back.
  await pointAndClick(window, card.getByTestId("research-plan-write-into"));
  await pointAndClick(window, window.getByTestId("research-plan-new-mind"));
  await expect(card.getByTestId("research-plan-write-step")).toHaveText(
    "Write a cited draft into a new Mind",
  );
  await pointAndClick(window, card.getByTestId("research-plan-write-into"));
  await pointAndClick(window, window.getByTestId("research-plan-mind").first());
  await expect(card.getByTestId("research-plan-write-into")).toHaveText("Credit memo");

  // "Change the plan…" in the composer has the model redo the steps.
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("Check the dates too");
  await window.keyboard.press("Enter");
  await expect(steps).toHaveCount(4);
  await expect(steps.nth(3)).toHaveValue("Check the dates too");
  await expect(card.getByTestId("research-plan-estimate")).toContainText("5–10 min");

  // Start (⌘↵ from the composer): the card goes, and its progress shows under its Question.
  await window.keyboard.press("ControlOrMeta+Enter");
  await expect(window.getByTestId("research-plan")).toHaveCount(0);
  await expect(window.getByTestId("task-tray")).toHaveCount(0);
  const progress = thread.getByTestId("task-progress");
  await expect(progress).toContainText("Research");
  await expect(thread.getByTestId("assistant-question").last()).toContainText(TOPIC);
  // Its first search asks to send to Anthropic, as an Answer's does.
  const consent = window.getByTestId("consent-dialog");
  await expect(consent).toContainText("Anthropic");
  await pointAndClick(window, consent.getByTestId("consent-allow"));
  await expect(progress.getByTestId("task-progress-steps")).toContainText("Searched the web for", {
    timeout: 30_000,
  });
  await expect(progress.getByTestId("task-progress-steps")).toContainText(
    "Read “Tea - The Open Encyclopedia”",
  );
  await screenshot(window, "research-running-en-1600");
  await expect(progress).toHaveAttribute("data-status", "done", { timeout: 60_000 });
  await expect(progress.getByTestId("task-progress-meta")).toHaveText(/^Done in \d+ min$/);
  const documents = await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listDocuments(),
  );
  expect(documents.filter((each) => each.kind === "web").map((each) => each.name)).toEqual([
    "Tea - The Open Encyclopedia",
  ]);
  await app.close();
});

test("the plan card follows the composer's model chip, says when the web can't be searched, fits 900px in Chinese, and Esc puts the topic back; Stop ends a running Research", async () => {
  test.setTimeout(120_000);
  const ollama = await ollamaListing(["fake-model", "other-model"]);
  try {
    const { app, window } = await launchApp(dataDir, { fakeChat: true, home });
    await dismissChatSetup(window);
    await setWindowSize(app, window, 900, 760);
    await window.evaluate(async (baseUrl) => {
      await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.saveChatProvider({
        kind: "ollama",
        baseUrl,
        modelId: "fake-model",
      });
    }, ollama.url);
    await newMind(window);
    await window.keyboard.type("Credit memo");
    await setLanguage(window, "zh-CN");

    const card = await research(window, `${TOPIC}, slowly`);
    // A model on this computer can't search the web: the card says so, with Web off.
    await expect(card.getByTestId("research-plan-estimate")).toHaveText("4–8 分钟 · fake-model");
    await expect(card.getByTestId("research-plan-web")).toBeDisabled();
    await expect(card.getByTestId("research-plan-web")).toHaveAttribute("aria-pressed", "false");
    await expect(card.getByTestId("research-plan-web-unavailable")).toHaveText(
      "fake-model 不能联网搜索",
    );
    // The composer's model chip is the card's model: changing it changes the card.
    await pointAndClick(window, window.getByTestId("composer-model"));
    const option = window
      .getByTestId("composer-model-menu")
      .getByTestId("composer-model-option")
      .filter({ hasText: "other-model" });
    await pointAndClick(window, option);
    await expect(card.getByTestId("research-plan-estimate")).toHaveText("4–8 分钟 · other-model");
    await expect(card.getByTestId("research-plan-web-unavailable")).toHaveText(
      "other-model 不能联网搜索",
    );

    // At 900px in Chinese, the card holds inside the composer's width, Start in view.
    const [cardBox, composerBox] = [
      await card.boundingBox(),
      await window.getByTestId("composer").boundingBox(),
    ];
    if (!cardBox || !composerBox) throw new Error("The card or the composer isn't on screen.");
    expect(cardBox.x + cardBox.width).toBeLessThanOrEqual(composerBox.x + composerBox.width + 1);
    await expect(card.getByTestId("research-plan-start")).toBeInViewport();
    await expect(card.getByTestId("research-plan-write-into")).toBeInViewport();
    await motionSettled(window);
    await screenshot(window, "research-plan-zh-900");

    // Esc cancels: the topic comes back to the composer, ready to be asked again.
    await card.getByTestId("research-plan-start").focus();
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("research-plan")).toHaveCount(0);
    await expect(window.getByTestId("composer-input")).toHaveValue(`${TOPIC}, slowly`);
    await expect(window.getByTestId("composer-research")).toBeVisible();

    // Asked again and started, it runs slowly; Stop ends it in the middle of a step.
    await window.keyboard.press("Enter");
    await expect(card.getByTestId("research-plan-step")).toHaveCount(3);
    await card.getByTestId("research-plan-start").focus();
    await window.keyboard.press("ControlOrMeta+Enter");
    const progress = window.getByTestId("assistant-thread").getByTestId("task-progress");
    await expect(progress).toHaveAttribute("data-status", "running");
    await expect(progress.getByTestId("task-progress-meta")).toHaveText(
      /^已完成 0\/3 步 · \d+ 分钟$/,
    );
    await expect(progress.getByTestId("task-progress-actions").getByRole("button")).toHaveText([
      "暂停",
      "停止",
    ]);
    await screenshot(window, "research-running-zh-900");
    await progress.getByTestId("task-progress-stop").focus();
    await window.keyboard.press("Enter");
    await expect(progress).toHaveAttribute("data-status", "stopped");
    await expect(progress.getByTestId("task-progress-meta")).toHaveText("已停止");
    await app.close();
  } finally {
    ollama.server.close();
  }
});
