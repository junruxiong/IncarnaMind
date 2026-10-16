import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge, SaveChatProviderInput } from "../src/core/api";
import {
  addDocuments,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/**
 * A model that can't use Tools, asked to write into the Mind (#947): the
 * Answer comes in the panel, with one line under it that says why nothing
 * was written, and Add to Mind, which puts the Answer in the Mind; Undo takes
 * it back. A Question shows no line.
 */

const NO_OLLAMA = "http://127.0.0.1:9";
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

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

const saveProvider = (window: Page, input: SaveChatProviderInput) =>
  window.evaluate(async (provider) => {
    await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.saveChatProvider(
      provider,
    );
  }, input);

test("asked to write, a model that can't says so under its Answer, and Add to Mind puts it in the Mind until Undo; a Question shows no line", async () => {
  const tides = join(sources, "Tides.md");
  await writeFile(tides, "# Tides\n\nSpring tides happen at new moon and at full moon.\n");
  const { app, window } = await launchApp(dataDir, { fakeChat: true, ollama: NO_OLLAMA });
  await dismissChatSetup(window);
  await setLanguage(window, "en");
  // The narrowest window, so the assistant is too.
  await setWindowSize(app, window, 900, 860);
  await saveProvider(window, { kind: "ollama", modelId: "tiny-no-tools" });
  await addDocuments(window, [tides]);
  await newMind(window);
  await window.keyboard.type("Tide notes");
  const editor = window.getByTestId("mind-editor");
  await editor.click();

  const question = await askInAssistant(window, "When do spring tides happen?");
  await expect(question.getByTestId("answer-cannot-write")).toHaveCount(0);

  const answer = await askInAssistant(window, "write a summary page");
  const line = answer.getByTestId("answer-cannot-write");
  await expect(line).toHaveText("This model can't write into your Mind itself. · Add to Mind");
  // It fits the assistant at its narrowest.
  const lineBox = await line.boundingBox();
  const answerBox = await answer.boundingBox();
  expect(
    lineBox && answerBox && lineBox.x + lineBox.width <= answerBox.x + answerBox.width + 1,
  ).toBe(true);
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "cannot-write-en.png") });

  const before = await editor.innerText();
  await pointAndClick(window, line.getByTestId("answer-cannot-write-add"));
  await expect(answer.getByTestId("answer-added")).toContainText("Added to “Tide notes”");
  await expect(editor).toContainText("This is a scripted Answer to: write a summary page");
  await pointAndClick(window, answer.getByTestId("answer-added-undo"));
  await expect(answer.getByTestId("answer-added")).toHaveCount(0);
  await expect.poll(() => editor.innerText()).toBe(before);

  await setLanguage(window, "zh-CN");
  await expect(line).toHaveText("这个模型不能自己写进你的 Mind。· 添加到 Mind");
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "cannot-write-zh.png") });
  await app.close();
});
