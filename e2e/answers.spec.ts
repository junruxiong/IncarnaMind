import { expect, type Locator, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  dragBlock,
  launchApp,
  mindWrittenBefore,
  pointAndClick,
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

/** Each top-level Block of the note: "question", "answer", or a Note's text. */
const order = (editor: Locator) =>
  editor
    .locator(":scope > *")
    .evaluateAll((all) =>
      all.map((block) =>
        block.matches(".node-question")
          ? "question"
          : block.matches(".node-answer")
            ? "answer"
            : (block.textContent ?? "").trim(),
      ),
    );

test("a Question asked in a new Mind gets an Answer that streams into its thread, and it is still there after reopening the app", async () => {
  const first = await launchApp(dataDir, { fakeChat: true });
  const { window } = first;
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await expect(window.getByTestId("chat-readiness")).toBeHidden();
  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();

  // Cmd/Ctrl+J focuses the composer; Enter asks, and the Question goes into the thread (#201).
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("What is IncarnaMind?");
  await window.keyboard.press("Enter");
  const thread = window.getByTestId("thread");
  await expect(thread.getByTestId("thread-question")).toBeVisible();

  // The Answer streams in below the Question: part of it shows while the rest is still coming.
  const answer = thread.getByTestId("answer");
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
  // The note has neither.
  await expect(editor.getByTestId("question")).toHaveCount(0);
  await expect(editor.getByTestId("answer")).toHaveCount(0);
  await first.app.close();

  // Reopened: the thread is folded to its header, naming the chat; unfolded, it is all there.
  const second = await launchApp(dataDir, { fakeChat: true });
  await second.window.getByTestId("mind-list-item").click();
  const reopened = second.window.getByTestId("thread");
  await expect(reopened.getByTestId("thread-subtitle")).toHaveText("What is IncarnaMind?");
  await pointAndClick(second.window, reopened.getByTestId("thread-fold"));
  await expect(reopened.getByTestId("thread-question")).toContainText("What is IncarnaMind?");
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

  // Asking from the composer without a chat model explains, in its row, what to set up; the
  // Question isn't left in the note or the thread, and its words stay in the composer.
  await sideNote.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("Is anyone there?");
  await window.keyboard.press("Enter");
  const notReady = window.getByTestId("composer").getByTestId("composer-not-ready");
  await expect(notReady).toBeVisible();
  await expect(notReady).toContainText("needs a chat model");
  await expect(window.getByTestId("composer-input")).toHaveValue("Is anyone there?");
  await expect(editor.getByTestId("question")).toHaveCount(0);
  await expect(editor.getByTestId("answer")).toHaveCount(0);
  await expect(window.getByTestId("thread")).toHaveCount(0);
  await app.close();
});

test("a Question and its Answer written in the note before (#201) are dragged and deleted together, and asking now leaves them as they are", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await mindWrittenBefore(window, "From before", [
    { note: "Intro" },
    { question: "What is IncarnaMind?", answer: "A notebook." },
    { heading: "Later" },
  ]);
  await pointAndClick(
    window,
    window.getByTestId("mind-list-item").filter({ hasText: "From before" }),
  );
  const editor = window.getByTestId("mind-editor");
  await expect.poll(() => order(editor)).toEqual(["Intro", "question", "answer", "Later"]);

  // A new Question goes to the thread; the Question and Answer in the note stay where they are.
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("And now?");
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("thread").getByTestId("answer")).toHaveAttribute(
    "data-status",
    "done",
    { timeout: 15_000 },
  );
  await expect.poll(() => order(editor)).toEqual(["Intro", "question", "answer", "Later"]);
  await window.keyboard.press("Escape");

  // Dragging the note's Question to the top takes its Answer along.
  const question = editor.getByTestId("question");
  const intro = editor.locator(":scope > p").first();
  await dragBlock(window, question, intro);
  await expect.poll(() => order(editor)).toEqual(["question", "answer", "Intro", "Later"]);

  // Deleting the Question from its menu deletes its Answer too.
  await (await slideToHandle(window, question)).click();
  await window.getByTestId("block-menu-delete").click();
  await expect.poll(() => order(editor)).toEqual(["Intro", "Later"]);
  await app.close();
});
