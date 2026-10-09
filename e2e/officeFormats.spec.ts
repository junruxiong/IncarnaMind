/**
 * Word, PowerPoint, Excel and CSV Documents (ADR-0011), end to end: linked,
 * processed, cited by slide, rows and section, opened at each Citation with
 * the quote highlighted and its mark beside it, and exported with each
 * Citation's Location in its footnote.
 *
 * The files are tests/fixtures/formats/ (see tests/core/formats.test.ts). Set
 * INCARNAMIND_SCREENSHOTS to a folder to keep a screenshot of each preview.
 */
import { copyFile, mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { footnotesOf, part, unzip } from "../tests/helpers/docx";
import {
  createDataFolder,
  dismissChatSetup,
  documentIdOf,
  interceptSaveDialog,
  launchApp,
  linkFolderFromSidebar,
  openDocumentAt,
  removeDataFolder,
  useLocalChatModel,
} from "./app";

const FIXTURES = join(__dirname, "..", "tests", "fixtures", "formats");
const FILES = [
  "Coastal Flood Risk Review.docx",
  "Quarterly Research Update.pptx",
  "Regional Revenue.xlsx",
  "Orders.csv",
];

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

/** Keeps a screenshot of the window, once the viewer has settled, when INCARNAMIND_SCREENSHOTS names a folder. */
async function screenshot(window: Page, name: string) {
  const folder = process.env.INCARNAMIND_SCREENSHOTS;
  if (!folder) return;
  await window.waitForFunction(() =>
    document.getAnimations().every((animation) => animation.playState !== "running"),
  );
  await mkdir(folder, { recursive: true });
  await window.screenshot({ path: join(folder, `${name}.png`) });
}

/** A wide window with a wide viewer: room for each page's margin, where the mark and its label go. */
async function widen(app: ElectronApplication, window: Page) {
  await app.evaluate(({ BrowserWindow }) => {
    BrowserWindow.getAllWindows()[0]?.setSize(1720, 1000);
  });
  await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.updateSettings({ device: { viewerWidth: 860 } });
  });
}

/** An element's box; it must be shown. */
async function boxOf(locator: Locator) {
  const box = await locator.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  return box;
}

/** Expects the Citation's mark beside the highlight: right of it, centred on its first line. */
async function expectMarkBeside(mark: Locator, highlight: Locator) {
  const markBox = await boxOf(mark);
  const line = await highlight.evaluate((element) => {
    const rect = element.getClientRects()[0] ?? element.getBoundingClientRect();
    return { right: rect.right, middle: rect.top + rect.height / 2 };
  });
  expect(markBox.height).toBe(18);
  expect(markBox.x).toBeGreaterThan(line.right);
  expect(Math.abs(markBox.y + markBox.height / 2 - line.middle)).toBeLessThanOrEqual(1.5);
}

/** Asks a Question in a new Mind and waits for its Answer to finish. */
async function ask(window: Page, text: string) {
  await window.getByTestId("new-mind").click();
  await window.getByTestId("mind-title").fill("Formats");
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type(text);
  await window.keyboard.press("Enter");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 20_000 });
  return answer;
}

test("Word, PowerPoint, Excel and CSV files in a linked folder are cited by section, slide and rows, opened at each Citation, and exported with their Locations", async () => {
  const library = join(await realpath(sources), "Library");
  await mkdir(library, { recursive: true });
  for (const file of FILES) await copyFile(join(FIXTURES, file), join(library, file));

  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await widen(app, window);
  await linkFolderFromSidebar(app, window, library);

  // Each file becomes a Document, and is processed.
  const items = window.getByTestId("document-list-item");
  await expect(items).toHaveCount(FILES.length);
  for (let index = 0; index < FILES.length; index++) {
    await expect(items.nth(index)).toHaveAttribute("data-status", "ready", { timeout: 30_000 });
  }

  const answer = await ask(
    window,
    "Compare each: the western fastest growth, the barrier crest halves damage, and West totals in rows.",
  );
  const chips = answer.getByTestId("citation-chip");
  await expect(chips).toHaveCount(4);
  const labels = await chips.evaluateAll((elements) =>
    elements.map((element) => element.getAttribute("aria-label") ?? ""),
  );
  const labelOf = (document: string) => labels.find((label) => label.includes(document)) ?? "";
  expect(labelOf("Quarterly Research Update")).toMatch(
    /^Citation \d: Quarterly Research Update, slide 3\. Quote found on slide 3$/,
  );
  expect(labelOf("Regional Revenue")).toMatch(
    /^Citation \d: Regional Revenue, Revenue, rows 7–8\. Quote found in Revenue, rows 7–8$/,
  );
  expect(labelOf("Coastal Flood Risk Review")).toMatch(
    /^Citation \d: Coastal Flood Risk Review, § 2\.1 Sensitivity\. Quote found in § 2\.1 Sensitivity$/,
  );
  expect(labelOf("Orders")).toMatch(/^Citation \d: Orders, rows 4–5\. Quote found in rows 4–5$/);

  const viewer = window.getByTestId("viewer");
  const mark = viewer.getByTestId("viewer-quote-mark");
  const open = async (document: string) => {
    await window.keyboard.press("Escape");
    const index = labels.findIndex((label) => label.includes(document));
    await chips.nth(index).click();
    await expect(window.getByTestId("viewer-title")).toHaveText(document);
  };

  // The slide: drawn as PowerPoint lays it out, at slide 3, the quote washed on it and the mark beside it.
  await open("Quarterly Research Update");
  await expect(viewer.getByTestId("viewer-slides")).toHaveAttribute("data-drawn", "yes");
  const slide = viewer.locator('[data-slide="3"]');
  const slideQuote = slide.locator("[data-quote-highlight]");
  await expect(slideQuote).toHaveText([
    "The western region grew fastest, at 18 per cent year on year.",
  ]);
  await expect(slideQuote.first()).toBeInViewport();
  // At the deck's own 16:9, as wide as the column allows.
  const drawnSlide = await boxOf(slide.locator(".viewer-deck-sheet"));
  expect(drawnSlide.width / drawnSlide.height).toBeCloseTo(16 / 9, 1);
  // Its speaker notes are there, folded, as the quote is on the slide.
  const notes = slide.getByTestId("viewer-slide-notes");
  await expect(notes).toContainText("Point at the red bar: that is the west.");
  await expect(notes).not.toHaveAttribute("open");
  await expect(slide.locator("img")).toHaveCount(1);
  await expect(mark.getByTestId("viewer-quote-mark-label")).toHaveText("slide 3");
  await expect(mark).toHaveAttribute("data-label-shown", "true");
  await expectMarkBeside(mark, slideQuote.first());
  await screenshot(window, "pptx-slides");

  // The rows: the Revenue sheet's grid, rows 7 and 8 washed, the mark beside them.
  await open("Regional Revenue");
  const sheet = viewer.getByTestId("viewer-sheet");
  await expect(sheet.getByRole("tab", { name: "Revenue" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  const cells = sheet.locator("td[data-quote-highlight]");
  await expect(cells).toHaveCount(12);
  expect(await cells.evaluateAll((elements) => elements.map((each) => each.dataset.ref))).toEqual([
    "A7",
    "B7",
    "C7",
    "D7",
    "E7",
    "F7",
    "A8",
    "B8",
    "C8",
    "D8",
    "E8",
    "F8",
  ]);
  await expect(cells.first()).toHaveText("West");
  await expect(cells.nth(1)).toHaveText("£350,200");
  await expect(cells.first()).toBeInViewport();
  await expect(mark.getByTestId("viewer-quote-mark-label")).toHaveText("Revenue, rows 7–8");
  await expect(mark).toHaveAttribute("data-label-shown", "true");
  await expectMarkBeside(mark, cells.nth(5));
  await screenshot(window, "xlsx-grid");
  // Its other sheets: the notes, and an empty one.
  await sheet.getByRole("tab", { name: "Notes" }).click();
  await expect(sheet.locator('td[data-ref="C3"]')).toHaveText(
    "Northern figures exclude the Leeds office, which reported late.",
  );
  await sheet.getByRole("tab", { name: "Empty" }).click();
  await expect(sheet.getByTestId("viewer-sheet-empty")).toBeVisible();

  // The section: the Word file's pages, the quote washed under 2.1 Sensitivity.
  await open("Coastal Flood Risk Review");
  const docx = viewer.getByTestId("viewer-docx");
  await expect(docx).toHaveAttribute("data-rendered", "yes");
  const wordQuote = docx.locator("[data-quote-highlight]");
  await expect(wordQuote.first()).toBeVisible();
  const washed = (await wordQuote.allTextContents()).join("");
  expect(washed).toContain("Raising the barrier crest by 40 centimetres halves");
  // It is under the cited heading: the heading comes before it in the drawn pages.
  const follows = await docx.evaluate((element) => {
    const heading = [...element.querySelectorAll("p")].find(
      (paragraph) => paragraph.textContent === "2.1 Sensitivity",
    );
    const quote = element.querySelector("[data-quote-highlight]");
    return heading && quote
      ? (heading.compareDocumentPosition(quote) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0
      : false;
  });
  expect(follows).toBe(true);
  await expect(wordQuote.first()).toBeInViewport();
  await expect(mark.getByTestId("viewer-quote-mark-label")).toHaveText("§ 2.1 Sensitivity");
  await expectMarkBeside(mark, wordQuote.first());
  await screenshot(window, "docx-pages");
  // Its outline lists its sections, and goes to one.
  await viewer.getByTestId("viewer-outline-button").click();
  const outline = viewer.getByTestId("viewer-outline-item");
  await expect(outline).toHaveText([
    "1 Introduction",
    "1.1 Scope",
    "2 Results",
    "2.1 Sensitivity",
    "3 Discussion",
    "3.1 Limitations",
  ]);
  await screenshot(window, "docx-outline");

  // The CSV: one sheet, rows 4 and 5 washed.
  await open("Orders");
  const orders = viewer.getByTestId("viewer-sheet").locator("td[data-quote-highlight]");
  await expect(orders).toHaveCount(10);
  await expect(orders.first()).toHaveText("1003");
  await expect(mark.getByTestId("viewer-quote-mark-label")).toHaveText("rows 4–5");
  await screenshot(window, "csv-grid");

  // Exported to .docx, each Citation's footnote names its Location.
  const target = join(sources, "Formats.docx");
  await interceptSaveDialog(app, target);
  await window.keyboard.press("Escape");
  await window.getByTestId("export-mind").click();
  const dialog = window.getByTestId("export-dialog");
  await expect(dialog.getByTestId("export-citations")).toHaveAttribute("data-unverified", "0");
  await dialog.getByTestId("export-save").click();
  await expect(dialog.getByTestId("export-done")).toBeVisible();
  await dialog.getByTestId("export-close").click();
  await expect(dialog).toBeHidden();
  const footnotes = Object.values(
    footnotesOf(part(unzip(await readFile(target)), "word/footnotes.xml")),
  ).sort();
  expect(footnotes).toEqual([
    "Coastal Flood Risk Review, § 2.1 Sensitivity",
    "Orders, rows 4–5",
    "Quarterly Research Update, slide 3",
    "Regional Revenue, Revenue, rows 7–8",
  ]);
  await app.close();
});

test("a Markdown file opens at its cited section and a text file at its cited lines, with the quote washed there", async () => {
  const notes = join(sources, "Field notes.md");
  const log = join(sources, "Harbour log.txt");
  await writeFile(
    notes,
    [
      "# Field notes",
      "",
      "Arrived at the harbour before dawn.",
      "",
      "## Methods",
      "",
      "Each gauge was read at high water.",
      "",
      "## Results",
      "",
      "Each gauge was read at high water, and again at low water.",
      "",
    ].join("\n"),
  );
  await writeFile(
    log,
    Array.from(
      { length: 140 },
      (_, index) => `${index + 1}: the tide turned at ${index % 12} o'clock.`,
    ).join("\n"),
  );
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await widen(app, window);
  await window.getByTestId("add-documents-input").setInputFiles([notes, log]);
  const items = window.getByTestId("document-list-item");
  await expect(items).toHaveCount(2);
  for (let index = 0; index < 2; index++) {
    await expect(items.nth(index)).toHaveAttribute("data-status", "ready");
  }
  const viewer = window.getByTestId("viewer");
  const text = viewer.getByTestId("viewer-text");

  // Section 3 is "Results": the quote is washed there, not under "Methods", where it also starts.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Field notes"),
    pageFrom: 3,
    quote: "Each gauge was read at high water",
    citation: { check: "found", number: 2, label: "§ Results" },
  });
  const washed = text.locator("[data-quote-highlight]");
  await expect(washed).toHaveCount(1);
  expect(await washed.evaluate((element) => element.closest("p")?.textContent ?? "")).toContain(
    "and again at low water",
  );
  await expect(viewer.getByTestId("viewer-quote-mark-label")).toHaveText("§ Results");
  await expect(viewer.getByTestId("viewer-quote-mark")).toHaveAttribute("data-label-shown", "true");
  await screenshot(window, "markdown-section");

  // Lines 101–140 are the third block of lines: the quote, on lines 5, 17… 137, is washed
  // where it first is in that block, on line 101, and scrolled into view.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Harbour log"),
    pageFrom: 3,
    quote: "the tide turned at 4 o'clock.",
    citation: { check: "found", number: 1, label: "line 101" },
  });
  const line = text.locator("[data-quote-highlight]");
  await expect(line).toHaveCount(1);
  expect(
    await line.evaluate((element) => element.previousSibling?.textContent?.split("\n").at(-1)),
  ).toBe("101: ");
  await expect(line).toBeInViewport();
  await screenshot(window, "txt-lines");
  await app.close();
});

test("a deck opens at a quote in a slide's speaker notes, which open, and marks a slide whose quote isn't on it", async () => {
  const deck = join(sources, "Quarterly Research Update.pptx");
  await copyFile(join(FIXTURES, "Quarterly Research Update.pptx"), deck);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await widen(app, window);
  await window.getByTestId("add-documents-input").setInputFiles([deck]);
  await expect(window.getByTestId("document-list-item")).toHaveAttribute("data-status", "ready");
  const documentId = await documentIdOf(window, "Quarterly Research Update");
  const viewer = window.getByTestId("viewer");
  const mark = viewer.getByTestId("viewer-quote-mark");

  // In the notes: they open under the slide, the quote washed there, the mark beside it.
  await openDocumentAt(window, {
    documentId,
    pageFrom: 3,
    quote: "Point at the red bar: that is the west.",
    citation: { check: "found", number: 2, label: "slide 3" },
  });
  const notes = viewer.locator('[data-slide="3"]').getByTestId("viewer-slide-notes");
  await expect(notes).toHaveAttribute("open", "");
  const washed = notes.locator("[data-quote-highlight]");
  await expect(washed).toHaveText("Point at the red bar: that is the west.");
  await expect(washed).toBeInViewport();
  await expectMarkBeside(mark, washed);
  await screenshot(window, "pptx-notes");
  // The notes fold again by hand.
  await notes.locator("summary").click();
  await expect(notes).not.toHaveAttribute("open");

  // Not on the slide or anywhere in the deck: the cited slide is shown, its mark at its top.
  await openDocumentAt(window, {
    documentId,
    pageFrom: 4,
    quote: "Revenue doubled in the north.",
    citation: { check: "not-found", number: 3, label: "slide 4" },
  });
  const cited = viewer.locator('[data-slide="4"]');
  await expect(cited).toHaveAttribute("data-quote-found", "none");
  await expect(viewer.locator("[data-quote-highlight]")).toHaveCount(0);
  await expect(mark).toHaveAttribute("data-check", "not-found");
  const markBox = await boxOf(mark);
  const sheetBox = await boxOf(cited.locator(".viewer-deck-sheet"));
  expect(markBox.x).toBeGreaterThan(sheetBox.x + sheetBox.width);
  expect(Math.abs(markBox.y - sheetBox.y - 8)).toBeLessThanOrEqual(1);
  await expect(cited.locator(".viewer-deck-sheet")).toBeInViewport();
  await screenshot(window, "pptx-slide-mark");
  await app.close();
});
