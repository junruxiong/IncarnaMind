import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";

import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  pointAndClick,
  removeDataFolder,
  useLocalChatModel,
} from "./app";

/** Three short pages: the line about spring tides is on page 2. */
const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
  { lines: ["Tide tables", "Harbours publish the times of high water every year."] },
]);

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

/** Asks a Question in a new Mind and waits for its Answer to finish. */
async function ask(window: Page, text: string) {
  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type(text);
  await window.keyboard.press("Enter");
  // Answered in the thread above the composer (#201).
  const answer = window.getByTestId("thread").getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  return answer;
}

test("an Answer cites a PDF: its badge says the quote was found, and clicking it opens the page with the quote highlighted", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf")]);

  const answer = await ask(window, "When do spring tides happen?");

  // It shows that it searched the Documents, and what for.
  await expect(answer.getByTestId("answer-tools")).toContainText("Searched your Documents");
  await answer.getByTestId("answer-tools-toggle").click();
  await expect(answer.getByTestId("answer-tool-call")).toContainText(
    "When do spring tides happen?",
  );

  // The marker became a Citation whose quote was found on page 2: numbered 1 in the text,
  // its label naming the Document, the page and the check.
  const citation = answer.getByTestId("citation");
  await expect(citation).toHaveCount(1);
  await expect(citation).toHaveAttribute("data-check", "found");
  await expect(citation.getByTestId("citation-chip")).toHaveText("1");
  await expect(citation.getByTestId("citation-chip")).toHaveAttribute(
    "aria-label",
    "Citation 1: Tides, p. 2. Quote found on p. 2",
  );
  await expect(answer).toContainText("Your Documents answer this");
  await expect(answer).not.toContainText("[^1]");

  // Added to the note, it keeps its Citation, checked again there: found (#201).
  const editor = window.getByTestId("mind-editor");
  await pointAndClick(window, answer.getByTestId("answer-add"));
  const note = editor.locator(":scope > p").first();
  await expect(note).toContainText("Your Documents answer this");
  await expect(note.getByTestId("citation")).toHaveAttribute("data-check", "found");
  await expect(editor.getByTestId("answer")).toHaveCount(0);

  // Clicking it shows its badge and opens the viewer at that page, the quote highlighted.
  await citation.getByTestId("citation-chip").click();
  await expect(window.getByTestId("citation-badge")).toHaveText("Quote found on p. 2");
  const viewer = window.getByTestId("viewer");
  await expect(window.getByTestId("viewer-title")).toHaveText("Tides");
  await expect(window.getByTestId("pdf-page-number")).toHaveValue("2");
  const highlights = viewer.locator('[data-page-number="2"] [data-quote-highlight]');
  await expect(highlights).toHaveText(["Spring tides happen at new moon and at full moon."]);
  await expect(highlights.first()).toBeInViewport();
  await app.close();
});

test("a quote that isn't on its page says so, and the page can still be opened, or the Citation removed", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf")]);

  const answer = await ask(window, "When do spring tides happen? Misquote it.");

  const citation = answer.getByTestId("citation");
  await expect(citation).toHaveAttribute("data-check", "not-found");
  await citation.getByTestId("citation-chip").click();
  const card = window.getByTestId("citation-card");
  await expect(card.getByTestId("citation-badge")).toHaveText("Quote not found on p. 2");
  await expect(card.getByTestId("citation-reason")).toContainText("isn't in the text");
  // Not found: the viewer doesn't open by itself.
  await expect(window.getByTestId("viewer")).toHaveCount(0);

  await card.getByTestId("citation-open-anyway").click();
  await expect(window.getByTestId("pdf-page-number")).toHaveValue("2");
  await expect(window.getByTestId("viewer").locator("[data-quote-highlight]")).toHaveCount(0);

  // Regenerating from the card writes the Answer again: it wasn't edited, so it doesn't ask.
  await citation.getByTestId("citation-chip").click();
  await window.getByTestId("citation-card").getByTestId("citation-regenerate").click();
  await expect(answer).toHaveAttribute("data-status", "streaming");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("answer-confirm")).toHaveCount(0);

  // In the thread an Answer is read, not edited: its card offers no Remove. Added to the note,
  // the Citation is still not found there, and the Note's copy of it can be removed.
  await citation.getByTestId("citation-chip").click();
  await expect(window.getByTestId("citation-card").getByTestId("citation-remove")).toHaveCount(0);
  await window.keyboard.press("Escape");
  await expect(window.getByTestId("citation-card")).toHaveCount(0);
  await pointAndClick(window, answer.getByTestId("answer-add"));
  const editor = window.getByTestId("mind-editor");
  const kept = editor.getByTestId("citation");
  await expect(kept).toHaveAttribute("data-check", "not-found");
  await kept.getByTestId("citation-chip").click();
  await window.getByTestId("citation-card").getByTestId("citation-remove").click();
  await expect(editor.getByTestId("citation")).toHaveCount(0);
  await expect(editor).toContainText("Your Documents answer this");
  await expect(answer.getByTestId("citation")).toHaveCount(1);
  await app.close();
});
