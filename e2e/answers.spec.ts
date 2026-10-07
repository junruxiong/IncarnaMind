import { expect, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  removeDataFolder,
  slideToHandle,
  useLocalChatModel,
} from "./app";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("a Question asked in a new Mind gets an Answer that streams in, and it is still there after reopening the app", async () => {
  const first = await launchApp(dataDir, { fakeChat: true });
  const { window } = first;
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await expect(window.getByTestId("chat-readiness")).toBeHidden();
  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();

  // Cmd/Ctrl+J turns the empty line into a Question; Enter asks it.
  await window.keyboard.press("ControlOrMeta+j");
  const question = editor.getByTestId("question");
  await expect(question).toBeVisible();
  await window.keyboard.type("What is IncarnaMind?");
  await window.keyboard.press("Enter");

  // The Answer streams in below the Question: part of it shows while the rest is still coming.
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "streaming");
  await expect(answer.getByTestId("answer-stop")).toBeVisible();
  await expect(answer).toContainText("scripted");
  expect(await answer.textContent()).not.toContain("That is all.");

  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("answer-stop")).toHaveCount(0);
  await expect(answer).toContainText("That is all.");
  await expect(answer).toContainText("What is IncarnaMind?");
  await expect(answer.getByTestId("answer-model")).toContainText("fake-model");
  // Rendered as rich text: bold, a list, a formula and code.
  await expect(answer.locator("strong")).toHaveText("scripted");
  await expect(answer.locator("li")).toHaveCount(2);
  await expect(answer.locator('[data-type="inline-math"] .katex')).toBeVisible();
  await expect(answer.locator("pre code")).toContainText('console.log("IncarnaMind");');
  await first.app.close();

  const second = await launchApp(dataDir, { fakeChat: true });
  await second.window.getByTestId("mind-list-item").click();
  const reopened = second.window.getByTestId("mind-editor");
  await expect(reopened.getByTestId("question")).toContainText("What is IncarnaMind?");
  const restored = reopened.getByTestId("answer");
  await expect(restored).toHaveAttribute("data-status", "done");
  await expect(restored).toContainText("That is all.");
  await expect(restored.locator('[data-type="inline-math"] .katex')).toBeVisible();
  await second.app.close();
});

test("a Note switched out of Question context looks muted, and asking without a chat model says why", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type("A side note the Answer shouldn't see.");

  // The Block menu switches the Note out of Question context.
  const sideNote = editor.locator("p").first();
  await (await slideToHandle(window, sideNote)).click();
  const toggle = window.getByTestId("block-menu-context");
  await expect(toggle).toHaveAttribute("aria-checked", "true");
  await toggle.click();
  await expect(sideNote).toHaveClass(/context-off/);

  // A Question from the slash menu; asking it without a chat model explains what to set up.
  await sideNote.click();
  await window.keyboard.press("End");
  await window.keyboard.press("Enter");
  await window.keyboard.type("/question");
  await expect(window.getByTestId("slash-item-question")).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Is anyone there?");
  await window.keyboard.press("Enter");
  const notReady = editor.getByTestId("question-not-ready");
  await expect(notReady).toBeVisible();
  await expect(notReady).toContainText("needs a chat model");
  await expect(editor.getByTestId("answer")).toHaveCount(0);
  await app.close();
});
