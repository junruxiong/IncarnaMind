/**
 * Choosing a service is the consent for what it sends (ADR-0022, #718): the
 * place the User chooses the chat model's provider says what goes to it, and
 * no dialog follows, neither before the first Answer nor for Tags. Tagging,
 * background work, never asks: it waits and says so where its status shows.
 * Set INCARNAMIND_SCREENSHOTS to a folder to keep screenshots of each step.
 */
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  acceptPlainTextKeys,
  addDocuments,
  askInAssistant,
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  openSettings,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

async function screenshot(window: Page, name: string): Promise<void> {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const SENDS_EN =
  "Your Questions, the Mind you're in, Passages from your Documents, and Document text for Tags go to OpenAI when you use it.";
const SENDS_ZH =
  "你的问题、当前 Mind、文档中的相关段落，以及用于打标签的文档文字，会在使用时发送给 OpenAI。";

let dataDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

/** Keeps every consent request the core raises from now on, whether or not a dialog shows it. */
async function recordConsentRequests(window: Page): Promise<() => Promise<string[]>> {
  await window.evaluate(() => {
    const scope = globalThis as unknown as { incarnamind: CoreBridge; consentAsked: string[] };
    scope.consentAsked = [];
    scope.incarnamind.on("consent.requested", (request) => {
      scope.consentAsked.push(request.flow.id);
    });
  });
  return () =>
    window.evaluate(() => (globalThis as unknown as { consentAsked: string[] }).consentAsked);
}

/** A text file in the sources folder, which the scripted tagger tags "Invoice". */
async function invoice(name: string): Promise<string> {
  const path = join(sources, `${name}.txt`);
  await writeFile(path, `${name}. Amount due: 120 EUR.\n`);
  return path;
}

test("first run: choosing a provider with a key is the consent, so no dialog comes before the Answer, and Tags start", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1280, 800);
  const consentAsked = await recordConsentRequests(window);

  // The line above the button says what goes to the provider.
  const setup = window.getByTestId("chat-setup");
  await setup.getByTestId("chat-choice-api-key").check();
  const form = setup.getByTestId("provider-form");
  await form.getByRole("radio", { name: "OpenAI", exact: true }).check();
  await expect(form.getByTestId("provider-sends")).toHaveText(SENDS_EN);
  await acceptPlainTextKeys(window, form);
  await form.getByLabel(/^API key/).fill("sk-not-a-real-key");
  await form.getByTestId("provider-sends").scrollIntoViewIfNeeded();
  await screenshot(window, "choosing-line-setup-en");
  await form.getByRole("button", { name: "Use this provider" }).click();
  await expect(setup).toBeHidden();

  // A Document is tagged without asking.
  await addDocuments(window, [await invoice("Invoice 41")]);
  const first = window.getByTestId("document-list-item").filter({ hasText: "Invoice 41" });
  await expect(first).toHaveAttribute("data-tagging", "tagged", { timeout: 15_000 });

  // A Question is answered without asking.
  await newMind(window);
  await askInAssistant(window, "What is due?");
  await expect(window.getByTestId("consent-dialog")).toBeHidden();
  expect(await consentAsked()).toEqual([]);

  // Settings › Privacy: each flow says it was allowed when OpenAI was chosen.
  const settings = await openSettings(window, "privacy");
  const decisionOf = (flow: string) =>
    settings
      .locator(`[data-testid="data-flow"][data-flow-id="${flow}"]`)
      .getByTestId("data-flow-service");
  for (const flow of ["chat", "edits", "checks", "suggestions", "tagging", "classification"]) {
    await expect(decisionOf(flow).getByTestId("data-flow-decision")).toHaveText(
      /^Allowed when you chose OpenAI, /,
    );
  }
  await screenshot(window, "privacy-chosen-en");

  // Revoking Tags stops tagging alone: the next Document waits, saying why, with Allow in place.
  await decisionOf("tagging").getByTestId("data-flow-revoke").click();
  await expect(decisionOf("tagging")).toHaveAttribute("data-consent", "not-asked");
  await expect(decisionOf("chat")).toHaveAttribute("data-consent", "accepted");
  await closeSettings(window);
  await window.getByTestId("add-documents-input").setInputFiles([await invoice("Invoice 42")]);
  const second = window.getByTestId("document-list-item").filter({ hasText: "Invoice 42" });
  await expect(second).toHaveAttribute("data-status", "ready");
  await expect(second).toHaveAttribute("data-tagging", "waiting-for-provider");
  const status = window.getByTestId("tagging-waiting");
  await expect(status).toContainText("Tags need your OK");
  await expect(status).toHaveAttribute(
    "title",
    "Tags wait until you allow Document text to go to OpenAI.",
  );
  await screenshot(window, "tagging-permission-en");
  await askInAssistant(window, "And now?");
  expect(await consentAsked()).toEqual([]);

  await status.getByTestId("tagging-waiting-allow").click();
  await expect(second).toHaveAttribute("data-tagging", "tagged", { timeout: 15_000 });
  await expect(status).toHaveCount(0);
  expect(await consentAsked()).toEqual([]);
  await app.close();
});

test("before a model is chosen, a new Document's tagging waits with its line, and nothing asks", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  const consentAsked = await recordConsentRequests(window);

  await addDocuments(window, [await invoice("Invoice 43")]);

  const item = window.getByTestId("document-list-item").filter({ hasText: "Invoice 43" });
  await expect(item).toHaveAttribute("data-tagging", "waiting-for-provider");
  const status = window.getByTestId("tagging-waiting");
  await expect(status).toContainText("Tags need a model");
  await expect(status).toHaveAttribute("title", "Tags wait until you choose a model.");
  await expect(window.getByTestId("consent-dialog")).toBeHidden();
  expect(await consentAsked()).toEqual([]);

  await setLanguage(window, "zh-CN");
  await expect(status).toHaveAttribute("title", "选好模型后再打标签。");
  await app.close();
});

test("in Chinese, Settings › Models says what goes to the provider above its button", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const settings = await openSettings(window, "models");
  await setLanguage(window, "zh-CN");

  await settings.getByTestId("chat-choice-api-key").check();
  const form = settings.getByTestId("provider-form");
  await form.getByRole("radio", { name: "OpenAI", exact: true }).check();

  const line = form.getByTestId("provider-sends");
  await expect(line).toHaveText(SENDS_ZH);
  await line.scrollIntoViewIfNeeded();
  // Right above the button it is said for.
  const lineBox = await line.boundingBox();
  const buttonBox = await form.getByRole("button", { name: "使用此提供商" }).boundingBox();
  expect(lineBox && buttonBox && lineBox.y + lineBox.height <= buttonBox.y).toBe(true);
  await screenshot(window, "choosing-line-settings-zh");
  await app.close();
});
