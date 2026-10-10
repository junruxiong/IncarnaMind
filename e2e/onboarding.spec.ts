import { access, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  linkedFolderRow,
  removeDataFolder,
  showSourceLocations,
  useLocalChatModel,
} from "./app";

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

/**
 * Moves the mouse onto an element as a person does, in steps, checks that it
 * is what is under the pointer there (nothing covers it), and clicks.
 * `atX`: that far into it from its left edge, instead of its middle;
 * `through`: it lets the pointer through to what is under it, which is clicked.
 */
async function pointAndClick(
  window: Page,
  target: Locator,
  { atX, through = false }: { atX?: number; through?: boolean } = {},
): Promise<void> {
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  const x = box.x + (atX ?? box.width / 2);
  const y = box.y + box.height / 2;
  await window.mouse.move(x, y, { steps: 8 });
  const hit = await target.evaluate(
    (element, [px, py]) => {
      const under = document.elementFromPoint(px as number, py as number);
      return under !== null && (under === element || element.contains(under));
    },
    [x, y],
  );
  expect(hit).toBe(!through);
  await window.mouse.click(x, y);
}

/** The "Get started" card's count, e.g. "· 1 of 3". */
const progress = (window: Page) => window.getByTestId("get-started-count");

test("a first run opens on the example Mind: its Answer is written in advance, its Citations are checked against its Documents, and the chat setup waits", async () => {
  const { window } = await launchApp(dataDir, { examples: true });

  const pane = window.getByTestId("mind-pane");
  await expect(window.getByTestId("mind-title")).toHaveValue("Where tea comes from");
  await expect(pane.getByTestId("example-banner")).toContainText("This is an example Mind.");
  // "Example" on the Mind and on its Documents' Linked folder; both Documents are listed.
  await expect(window.getByTestId("mind-list-item").getByTestId("example-chip")).toHaveText(
    "Example",
  );
  await expect(window.getByTestId("document-list-item")).toHaveText([
    /Tea · Wikipedia|茶 · 维基百科/,
    /Tea · Wikipedia|茶 · 维基百科/,
  ]);
  // No model needed: no setup over it, and no "set up a chat model" notice in it.
  await expect(window.getByTestId("chat-setup")).toBeHidden();
  await expect(pane.getByTestId("chat-readiness")).toHaveCount(0);

  const answer = pane.getByTestId("answer");
  await expect(answer.getByTestId("answer-example")).toHaveText(
    "Example Answer, written in advance: no model needed to try it",
  );
  // Both Citations are checked against the Documents' text: the quotes are found.
  const checks = pane.getByTestId("margin-check");
  await expect(checks).toHaveCount(2);
  await expect(checks.nth(0)).toHaveAttribute("data-check", "found", { timeout: 20_000 });
  await expect(checks.nth(1)).toHaveAttribute("data-check", "found", { timeout: 20_000 });

  // The Linked folder's chip, as on disk, once its status (indexing) leaves the room for it.
  await showSourceLocations(window);
  await expect(linkedFolderRow(window, "Examples").getByTestId("example-chip")).toBeVisible({
    timeout: 30_000,
  });

  await expect(progress(window)).toHaveText("· 0 of 3");
  // Clicking a green check opens its card and the Document at the quote, and ticks the first step.
  await pointAndClick(window, checks.nth(0));
  await expect(window.getByTestId("citation-card")).toBeVisible();
  await expect(window.getByTestId("viewer")).toBeVisible();
  await expect(progress(window)).toHaveText("· 1 of 3");
  await expect(
    window.locator('[data-testid="get-started-step"][data-step="citation"] [data-done]'),
  ).toContainText("Check a Citation in the example");

  // Moving on to a Mind of one's own: the chat setup is offered then.
  await window.keyboard.press("Escape");
  await pointAndClick(window, pane.getByTestId("example-add-own"));
  await expect(window.getByTestId("chat-setup")).toBeVisible();
});

test("removing the examples deletes the example Mind, its Documents and the copies of their files, and they aren't made again", async () => {
  const first = await launchApp(dataDir, { examples: true });
  const window = first.window;
  await expect(window.getByTestId("example-banner")).toBeVisible();

  await pointAndClick(window, window.getByTestId("example-remove"));
  await expect(window.getByTestId("mind-list-item")).toHaveCount(0);
  await expect(window.getByTestId("document-list-item")).toHaveCount(0);
  await expect(linkedFolderRow(window, "Examples")).toHaveCount(0);
  await expect(window.getByTestId("mind-none-open")).toBeVisible();
  // Moving on from the example: setting up a chat model is offered now.
  await dismissChatSetup(window);
  await expect
    .poll(() =>
      access(join(dataDir, "Examples")).then(
        () => "there",
        () => "gone",
      ),
    )
    .toBe("gone");
  await first.app.close();

  // A first run happens once: the next launch doesn't make them again.
  const again = await launchApp(dataDir, { examples: true });
  await expect(again.window.getByTestId("mind-none-open")).toBeVisible();
  await expect(again.window.getByTestId("mind-list-item")).toHaveCount(0);
  // "Check a Citation in the example" makes them again, and opens them.
  await pointAndClick(
    again.window,
    again.window.locator('[data-testid="get-started-step"][data-step="citation"] button'),
  );
  await expect(again.window.getByTestId("example-banner")).toBeVisible();
  await again.app.close();
});

test("with only the example Documents, the footer doesn't ask for a model to tag them; a Document of one's own without a model does", async () => {
  const own = join(sources, "Harbour notes.md");
  await writeFile(own, "# Harbour notes\n\nThe harbour master publishes the tide tables.\n");
  const { app, window } = await launchApp(dataDir, { examples: true });
  const items = window.getByTestId("document-list-item");
  const notice = window.getByTestId("sidebar-footer").getByTestId("tagging-waiting");

  // The examples are ready and their tagging waits for a model, but they need none: no notice.
  await expect(items).toHaveCount(2);
  for (const item of await items.all()) {
    await expect(item).toHaveAttribute("data-tagging", "waiting-for-provider", {
      timeout: 30_000,
    });
  }
  await expect(notice).toHaveCount(0);

  // A Document of one's own, with no model: the notice, as always.
  await window.getByTestId("add-documents-input").setInputFiles([own]);
  const added = items.filter({ hasText: "Harbour notes" });
  await expect(added).toHaveAttribute("data-tagging", "waiting-for-provider");
  await expect(notice).toBeVisible();
  await expect(notice).toContainText("Tags need a model");
  await app.close();
});

test("an empty Mind of one's own shows the three steps; Get started ticks itself as Documents are added and a Question is asked, and goes once all are done", async () => {
  await writeFile(
    join(sources, "Harbour notes.md"),
    "# Harbour notes\n\nThe harbour master publishes the tide tables every January.\n",
  );
  const { window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  await expect(progress(window)).toHaveText("· 0 of 3");

  // The banner's way on: a Mind of one's own, which shows the three steps.
  await pointAndClick(window, window.getByTestId("example-add-own"));
  await dismissChatSetup(window);
  const guide = window.getByTestId("start-guide");
  await expect(guide).toBeVisible();
  await expect(guide.getByTestId("start-step")).toHaveCount(3);
  await expect(guide).toContainText("Only your Documents get checked Citations.");

  // Adding a file of one's own ticks "Index your Documents or connect apps".
  await window
    .getByTestId("add-documents-input")
    .setInputFiles([join(sources, "Harbour notes.md")]);
  await expect(progress(window)).toHaveText("· 1 of 3");
  await expect(
    window.locator('[data-testid="get-started-step"][data-step="documents"] [data-done]'),
  ).toBeVisible();

  // The empty Mind's hint focuses the composer, for a Question to go where the hint is.
  await useLocalChatModel(window);
  const hint = window.getByTestId("end-hint");
  await expect(hint).toHaveAttribute("data-place", "empty");
  // A first click puts the cursor on the line; then its link is armed.
  await pointAndClick(window, hint, { atX: 12, through: true });
  await expect(hint).toHaveAttribute("data-armed", "true");
  await pointAndClick(window, hint.getByTestId("end-hint-ask"));
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("When are the tide tables published?");
  await window.keyboard.press("Enter");
  const editor = window.getByTestId("mind-editor");
  await expect(editor.getByTestId("question")).toHaveCount(1);
  await expect(guide).toBeHidden();
  await expect(editor.getByTestId("answer")).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  await expect(progress(window)).toHaveText("· 2 of 3");

  // The last step, from the card: the example Mind opens, and checking a Citation there ends it.
  await pointAndClick(
    window,
    window.locator('[data-testid="get-started-step"][data-step="citation"] button'),
  );
  const checks = window.getByTestId("mind-pane").getByTestId("margin-check");
  await expect(checks.nth(0)).toHaveAttribute("data-check", "found", { timeout: 20_000 });
  await pointAndClick(window, checks.nth(0));
  await expect(window.getByTestId("get-started")).toHaveCount(0);
});

test("Get started can be hidden, and stays hidden", async () => {
  const first = await launchApp(dataDir, { examples: true });
  await pointAndClick(first.window, first.window.getByTestId("get-started-hide"));
  await expect(first.window.getByTestId("get-started")).toHaveCount(0);
  await first.app.close();

  const again = await launchApp(dataDir, { examples: true });
  await expect(again.window.getByTestId("example-banner")).toBeVisible();
  await expect(again.window.getByTestId("get-started")).toHaveCount(0);
  await again.app.close();
});

test("the hint at the end of a Mind focuses the composer when its link is clicked once the cursor is on its line, and is otherwise the line to write on", async () => {
  const { window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  const editor = window.getByTestId("mind-editor");
  const hint = window.getByTestId("end-hint");
  await expect(hint).toHaveAttribute("data-place", "afterAnswer");
  const shortcut = process.platform === "darwin" ? "⌘J" : "Ctrl+J";
  await expect(hint).toHaveText(
    `Try it: write a line here, or press ${shortcut} to ask your own Question`,
  );

  // Before the cursor is on the line, a click on the link puts the cursor there, to write.
  await expect(hint).not.toHaveAttribute("data-armed");
  await pointAndClick(window, hint.getByTestId("end-hint-ask"), { through: true });
  await expect(hint).toHaveAttribute("data-armed", "true");
  await expect(editor.getByTestId("question")).toHaveCount(1);
  await window.keyboard.type("Green and black tea come from the same plant.");
  const last = editor.locator(":scope > *").last();
  await expect(last).toHaveText("Green and black tea come from the same plant.");
  await expect(editor.getByTestId("question")).toHaveCount(1);
  // A Note at the end: no hint, as after an Answer only.
  await expect(hint).toHaveCount(0);

  // An empty line after a Note has none.
  await window.keyboard.press("Enter");
  await expect(hint).toHaveCount(0);
  await window.keyboard.press("Backspace");
  // Back on the Note: put the cursor on a new empty line right after the Answer.
  await pointAndClick(window, last, { atX: 2 });
  // The editor takes the click's place from the browser's selectionchange: wait, as a person would.
  await expect
    .poll(() =>
      editor.evaluate(
        (dom) => (dom as unknown as { editor: Editor }).editor.state.selection.$from.parentOffset,
      ),
    )
    .toBe(0);
  await window.keyboard.press("Enter");
  await window.keyboard.press("ArrowUp");
  // An empty line after an Answer, not at the end, shows the hint while it is written in.
  await expect(hint).toHaveAttribute("data-place", "afterAnswer");

  // Its link focuses the composer; the Question asked there goes on that line.
  await pointAndClick(window, hint.getByTestId("end-hint-ask"));
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await useLocalChatModel(window);
  await window.keyboard.type("Which tea is drunk most in Britain?");
  await window.keyboard.press("Enter");
  await expect(editor.getByTestId("question")).toHaveCount(2);
  await expect(editor.getByTestId("question").nth(1)).toContainText(
    "Which tea is drunk most in Britain?",
  );
  // Right after the example's Answer, before the Note, with a line under its Answer to write on.
  await expect(editor.getByTestId("answer").nth(1)).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  const order = await editor
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
  expect(order.slice(-6)).toEqual([
    "question",
    "answer",
    "question",
    "answer",
    "",
    "Green and black tea come from the same plant.",
  ]);
});
