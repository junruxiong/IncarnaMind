import { expect, test } from "@playwright/test";
import {
  assistantAnswers,
  createDataFolder,
  dismissChatSetup,
  dragBlock,
  launchApp,
  mindWrittenBefore,
  newMind,
  pointAndClick,
  pointAt,
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

test("a Question asked in the assistant gets an Answer that streams in there, and the conversation is still there after reopening the app", async () => {
  const first = await launchApp(dataDir, { fakeChat: true });
  const { window } = first;
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await expect(window.getByTestId("chat-readiness")).toBeHidden();
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();

  // Cmd/Ctrl+J focuses the composer; Enter asks, and the Question shows in the assistant.
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("What is IncarnaMind?");
  await window.keyboard.press("Enter");
  const assistant = window.getByTestId("assistant");
  await expect(assistant.getByTestId("assistant-question")).toContainText("What is IncarnaMind?");

  // The Answer streams in below the Question: part of it shows while the rest is still coming.
  const answer = assistantAnswers(window).first();
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
  // The Mind holds no Question and no Answer: asking never writes into it.
  await expect(editor.getByTestId("question")).toHaveCount(0);
  await expect(editor.getByTestId("answer")).toHaveCount(0);

  // Kept in the Mind, with a link back to its conversation.
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  await pointAndClick(window, answer.getByTestId("answer-add"));
  await expect(editor.locator("strong")).toHaveText("scripted");
  await first.app.close();

  // Reopened: the link back opens the conversation, its Answer as it was.
  const second = await launchApp(dataDir, { fakeChat: true });
  await second.window.getByTestId("mind-list-item").click();
  await expect(second.window.getByTestId("assistant-empty")).toBeVisible();
  await pointAndClick(second.window, second.window.getByTestId("conversation-link"));
  const restored = assistantAnswers(second.window).first();
  await expect(restored).toHaveAttribute("data-status", "done");
  await expect(restored).toContainText("That is all.");
  await expect(restored.locator('[data-type="inline-math"] .katex')).toBeVisible();
  await expect(second.window.getByTestId("assistant-question")).toContainText(
    "What is IncarnaMind?",
  );
  await second.app.close();
});

test("a Note switched out of Question context looks muted, and asking without a chat model says why", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await newMind(window);
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

  // Asking from the composer without a chat model explains, in its row, what to set up; nothing
  // is kept in the assistant, and the words stay in the composer.
  await sideNote.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("Is anyone there?");
  await window.keyboard.press("Enter");
  const notReady = window.getByTestId("composer").getByTestId("composer-not-ready");
  await expect(notReady).toBeVisible();
  await expect(notReady).toContainText("needs a chat model");
  await expect(window.getByTestId("composer-input")).toHaveValue("Is anyone there?");
  await expect(window.getByTestId("assistant-question")).toHaveCount(0);
  await expect(window.getByTestId("assistant-empty")).toBeVisible();
  await expect(editor.getByTestId("question")).toHaveCount(0);
  await app.close();
});

test("in a Mind written before the assistant, a Question and its Answer are dragged and deleted together", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await mindWrittenBefore(window, "Written before", [
    { note: "Intro" },
    { question: "What is IncarnaMind?", answer: "A notebook." },
    { note: "Next thought" },
    { heading: "Later" },
  ]);
  await window.getByTestId("mind-list-item").filter({ hasText: "Written before" }).click();
  const editor = window.getByTestId("mind-editor");
  const question = editor.getByTestId("question");
  await expect(question.locator(".question-text")).toHaveText("What is IncarnaMind?");
  const intro = editor.locator(":scope > p").first();
  const blocks = editor.locator(":scope > *");
  const order = () =>
    blocks.evaluateAll((all) =>
      all.map((block) =>
        block.matches(".node-question")
          ? "question"
          : block.matches(".node-answer")
            ? "answer"
            : (block.textContent ?? "").trim(),
      ),
    );
  await expect.poll(order).toEqual(["Intro", "question", "answer", "Next thought", "Later"]);

  // Dragging the Question to the top takes its Answer along.
  await dragBlock(window, question, intro);
  await expect.poll(order).toEqual(["question", "answer", "Intro", "Next thought", "Later"]);

  // Deleting the Question from its menu deletes its Answer too.
  await (await slideToHandle(window, question)).click();
  await window.getByTestId("block-menu-delete").click();
  await expect.poll(order).toEqual(["Intro", "Next thought", "Later"]);
  await app.close();
});
