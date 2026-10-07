import { rm, writeFile } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { expect, type Locator, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  darkPixels,
  dismissChatSetup,
  documentIdOf,
  interceptOpenPath,
  launchApp,
  openDocumentAt,
  openDocumentMenu,
  pathsOpened,
  removeDataFolder,
  widthOf,
} from "./app";

/** An element's box; it must be shown. */
async function boxOf(locator: Locator) {
  const box = await locator.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  return box;
}

/**
 * Expects the Citation's mark to sit beside the highlight: right of it, inside
 * the page, and centred on its first line.
 */
async function expectMarkBeside(mark: Locator, highlight: Locator, page: Locator) {
  const markBox = await boxOf(mark);
  const line = await highlight.evaluate((element) => {
    const rect = element.getClientRects()[0] ?? element.getBoundingClientRect();
    return { right: rect.right, middle: rect.top + rect.height / 2 };
  });
  const pageBox = await boxOf(page);
  expect(markBox.height).toBe(18);
  expect(markBox.x).toBeGreaterThan(line.right);
  expect(markBox.x + markBox.width).toBeLessThanOrEqual(pageBox.x + pageBox.width);
  expect(Math.abs(markBox.y + markBox.height / 2 - line.middle)).toBeLessThanOrEqual(1.5);
}

const backgroundOf = (locator: Locator) =>
  locator.evaluate((element) => getComputedStyle(element).backgroundColor);

/** success-wash and warning-wash, as computed. */
const FOUND_WASH = "rgb(220, 243, 228)";
const NOT_FOUND_WASH = "rgb(254, 243, 226)";

/** Three pages; on page 2 a sentence runs across a line break. */
const REPORT = buildPdf([
  { lines: ["Quarterly report", "Prepared for the board."] },
  {
    lines: [
      "Results",
      "Revenue grew by ten percent",
      "in the third quarter, led by exports.",
      "Costs stayed flat.",
    ],
  },
  { lines: ["Outlook", "We expect steady growth next year."] },
]);

/** Six pages with an outline: two entries at the top, two under the second. */
const GUIDE = buildPdf(
  Array.from({ length: 6 }, (_, index) => ({ lines: [`Guide page ${index + 1}`] })),
  {
    outline: [
      { title: "Getting started", page: 1 },
      {
        title: "Reference",
        page: 3,
        via: "named",
        items: [
          { title: "Settings", page: 4 },
          { title: "Troubleshooting", page: 6, via: "action" },
        ],
      },
    ],
  },
);

/** Long enough to scroll, with the quoted sentence near the end. */
const NOTES = [
  "# Field notes",
  "",
  ...Array.from(
    { length: 120 },
    (_, index) => `Observation ${index + 1}: nothing unusual today.\n`,
  ),
  "The *key* finding was that attention",
  "spans shrink after lunch.",
  "",
].join("\n");

let dataDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder(); // the User's own files live outside the data folder
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

async function writeSources() {
  const report = join(sources, "Report.pdf");
  const notes = join(sources, "Field notes.md");
  await writeFile(report, REPORT);
  await writeFile(notes, NOTES);
  return { report, notes };
}

test("clicking a PDF in the sidebar opens it in the viewer, and another Document replaces it", async () => {
  const { report, notes } = await writeSources();
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [report, notes]);
  const viewer = window.getByTestId("viewer");
  const mindArea = window.getByTestId("mind-area");
  const items = window.getByTestId("document-list-item");
  await expect(viewer).toHaveCount(0);
  const fullWidth = await widthOf(mindArea);

  await items.filter({ hasText: "Report" }).getByTestId("open-document").click();
  await expect(viewer).toBeVisible();
  expect(await widthOf(mindArea)).toBeLessThan(fullWidth);
  await expect(window.getByTestId("viewer-title")).toHaveText("Report");

  // Page 1 is drawn on its canvas, with its text selectable in the text layer.
  const page1 = viewer.locator('[data-page-number="1"]');
  await expect(page1).toHaveAttribute("data-drawn", "true");
  expect(await darkPixels(page1.locator("canvas"))).toBeGreaterThan(0);
  await expect(page1.locator(".textLayer")).toContainText("Quarterly report");
  await expect(window.getByTestId("pdf-page-number")).toHaveValue("1");
  await expect(window.getByTestId("viewer-header")).toContainText("of 3");

  // Page navigation: next, and typing a page number.
  await window.getByTestId("pdf-next-page").click();
  await expect(window.getByTestId("pdf-page-number")).toHaveValue("2");
  await expect(viewer.locator('[data-page-number="2"]')).toBeInViewport();
  await window.getByTestId("pdf-page-number").fill("3");
  await window.getByTestId("pdf-page-number").press("Enter");
  await expect(window.getByTestId("pdf-page-number")).toHaveValue("3");
  await expect(viewer.locator('[data-page-number="3"] .textLayer')).toContainText("Outlook");

  // Zoom: in from fit-width, then back to fit-width.
  const fitted = await widthOf(page1);
  await window.getByTestId("pdf-zoom-in").click();
  await expect(window.getByTestId("pdf-fit-width")).toHaveAttribute("aria-pressed", "false");
  await expect.poll(() => widthOf(page1)).toBeGreaterThan(fitted);
  await window.getByTestId("pdf-fit-width").click();
  await expect.poll(() => widthOf(page1)).toBe(fitted);

  // Another Document replaces the PDF: one Document at a time.
  await items.filter({ hasText: "Field notes" }).getByTestId("open-document").click();
  await expect(window.getByTestId("viewer-title")).toHaveText("Field notes");
  await expect(viewer.locator("[data-page-number]")).toHaveCount(0);
  await expect(window.getByTestId("viewer-text").locator("h1")).toHaveText("Field notes");

  // Closing gives the Mind its full width back.
  await window.getByTestId("viewer-close").click();
  await expect(viewer).toHaveCount(0);
  expect(await widthOf(mindArea)).toBe(fullWidth);
  await app.close();
});

test("a PDF's outline shows beside its pages, and an entry goes to its page", async () => {
  const guide = join(sources, "Guide.pdf");
  await writeFile(guide, GUIDE);
  const { report } = await writeSources();
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [guide, report]);
  const viewer = window.getByTestId("viewer");
  const items = window.getByTestId("document-list-item");
  const pageNumber = window.getByTestId("pdf-page-number");

  // A PDF without an outline has no outline button.
  await items.filter({ hasText: "Report" }).getByTestId("open-document").click();
  await expect(viewer.locator('[data-page-number="1"]')).toHaveAttribute("data-drawn", "true");
  await expect(window.getByTestId("pdf-outline-button")).toHaveCount(0);

  await items.filter({ hasText: "Guide" }).getByTestId("open-document").click();
  const button = window.getByTestId("pdf-outline-button");
  await expect(button).toHaveAttribute("aria-pressed", "false");
  await expect(window.getByTestId("pdf-outline")).toHaveCount(0);

  // Shown: the top-level entries, the nested ones once their entry is expanded.
  await button.click();
  const outline = window.getByTestId("pdf-outline");
  await expect(button).toHaveAttribute("aria-pressed", "true");
  const entries = outline.getByTestId("pdf-outline-item");
  await expect(entries).toHaveText(["Getting started", "Reference"]);
  await outline.getByRole("button", { name: "Expand Reference" }).click();
  await expect(entries).toHaveText(["Getting started", "Reference", "Settings", "Troubleshooting"]);

  // Rows are 28px; a nested entry is indented 16px more than its parent.
  const reference = entries.filter({ hasText: "Reference" });
  const settings = entries.filter({ hasText: "Settings" });
  expect((await boxOf(reference)).height).toBe(28);
  const indent = (entry: Locator) =>
    entry.evaluate((element) => Number.parseFloat(getComputedStyle(element).paddingLeft));
  expect((await indent(settings)) - (await indent(reference))).toBe(16);

  // The entry for the page in view is marked.
  const current = outline.locator('[aria-current="location"]');
  await expect(current).toHaveText("Getting started");

  // Clicking an entry goes to its page.
  await entries.filter({ hasText: "Troubleshooting" }).click();
  await expect(pageNumber).toHaveValue("6");
  await expect(viewer.locator('[data-page-number="6"] .textLayer')).toContainText("Guide page 6");
  await expect(current).toHaveText("Troubleshooting");
  // Collapsed, an entry stands for the entries under it.
  await outline.getByRole("button", { name: "Collapse Reference" }).click();
  await expect(current).toHaveText("Reference");
  await outline.getByRole("button", { name: "Expand Reference" }).click();
  await reference.click();
  await expect(pageNumber).toHaveValue("3");
  await expect(viewer.locator('[data-page-number="3"]')).toBeInViewport();
  await expect(current).toHaveText("Reference");

  // Hidden again: the pages take the whole width back.
  const narrower = await widthOf(window.getByTestId("pdf-scroller"));
  await button.click();
  await expect(outline).toHaveCount(0);
  expect(await widthOf(window.getByTestId("pdf-scroller"))).toBeGreaterThan(narrower);
  await app.close();
});

test("a long PDF draws only the pages near the view", async () => {
  const long = join(sources, "Long.pdf");
  await writeFile(
    long,
    buildPdf(Array.from({ length: 40 }, (_, index) => ({ lines: [`Page ${index + 1} of 40`] }))),
  );
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [long]);
  await window.getByTestId("open-document").click();
  const viewer = window.getByTestId("viewer");
  const drawn = viewer.locator('[data-drawn="true"]');

  await expect(viewer.locator("[data-page-number]")).toHaveCount(40);
  await expect(viewer.locator('[data-page-number="1"]')).toHaveAttribute("data-drawn", "true");
  expect(await drawn.count()).toBeLessThan(10);

  // Far down the Document, its pages are drawn and the first page is released.
  await window.getByTestId("pdf-page-number").fill("30");
  await window.getByTestId("pdf-page-number").press("Enter");
  await expect(viewer.locator('[data-page-number="30"]')).toHaveAttribute("data-drawn", "true");
  await expect(viewer.locator('[data-page-number="30"] .textLayer')).toContainText("Page 30 of 40");
  await expect(viewer.locator('[data-page-number="1"]')).toHaveAttribute("data-drawn", "false");
  expect(await drawn.count()).toBeLessThan(10);
  await app.close();
});

test("opened at a page range with a quote, the viewer shows that page with the quote highlighted", async () => {
  const { report, notes } = await writeSources();
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [report, notes]);
  const viewer = window.getByTestId("viewer");

  // The quote runs across a line break on page 2.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Report"),
    pageFrom: 2,
    pageTo: 2,
    quote: "Revenue grew by ten percent in the third quarter",
  });
  await expect(window.getByTestId("pdf-page-number")).toHaveValue("2");
  const highlights = viewer.locator('[data-page-number="2"] [data-quote-highlight]');
  await expect(highlights).toHaveCount(2);
  await expect(highlights).toHaveText(["Revenue grew by ten percent", "in the third quarter"]);
  await expect(highlights.first()).toBeInViewport();
  await expect(viewer.locator('[data-page-number="1"] [data-quote-highlight]')).toHaveCount(0);

  // A quote with an ellipsis: each part is highlighted, and the words left out aren't.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Report"),
    pageFrom: 2,
    pageTo: 2,
    quote: "Revenue grew by ten percent … led by exports. Costs stayed flat.",
  });
  await expect(highlights).toHaveText([
    "Revenue grew by ten percent",
    "led by exports.",
    "Costs stayed flat.",
  ]);

  // A quote that isn't on those pages: the page opens, with nothing highlighted.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Report"),
    pageFrom: 3,
    quote: "Revenue grew by ten percent",
  });
  await expect(window.getByTestId("pdf-page-number")).toHaveValue("3");
  await expect(viewer.locator("[data-quote-highlight]")).toHaveCount(0);

  // Markdown: the viewer scrolls down to the quote and highlights it.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Field notes"),
    quote: "attention spans shrink after lunch.",
  });
  const marks = window.getByTestId("viewer-text").locator("[data-quote-highlight]");
  await expect(marks.first()).toBeInViewport();
  expect((await marks.allTextContents()).join("")).toBe("attention\nspans shrink after lunch.");

  // With an ellipsis, each part is highlighted.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Field notes"),
    quote: "Observation 119: nothing unusual today. ... Observation 120: nothing unusual today.",
  });
  await expect(marks).toHaveText([
    "Observation 119: nothing unusual today.",
    "Observation 120: nothing unusual today.",
  ]);
  await expect(marks.first()).toBeInViewport();
  await app.close();
});

test("a Document deleted while it is open, or already deleted, shows Document removed", async () => {
  const { report } = await writeSources();
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [report]);
  const viewer = window.getByTestId("viewer");
  const item = window.getByTestId("document-list-item");
  const documentId = await documentIdOf(window, "Report");

  await item.getByTestId("open-document").click();
  await expect(viewer.locator('[data-page-number="1"]')).toHaveAttribute("data-drawn", "true");

  await openDocumentMenu(item);
  await item.getByTestId("delete-document").click();
  await window.getByTestId("confirm-delete-document").click();
  await expect(item).toHaveCount(0);
  await expect(window.getByTestId("viewer-removed")).toContainText("Document removed");
  await expect(viewer.locator("[data-page-number]")).toHaveCount(0);

  // A Citation of a deleted Document opens the panel on the message and its stored quote.
  await window.getByTestId("viewer-close").click();
  await openDocumentAt(window, { documentId, pageFrom: 2, quote: "Revenue grew by ten percent" });
  await expect(window.getByTestId("viewer-removed")).toContainText("Document removed");
  await expect(window.getByTestId("viewer-removed")).toContainText("Revenue grew by ten percent");
  await app.close();
});

test("the viewer has one 44px header, level with the Mind's, holding every control in order", async () => {
  const guide = join(sources, "Guide.pdf");
  await writeFile(guide, GUIDE);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [guide]);
  await window.getByTestId("new-mind").click();
  await window.getByTestId("open-document").click();
  const viewer = window.getByTestId("viewer");
  await expect(viewer.locator('[data-page-number="1"]')).toHaveAttribute("data-drawn", "true");

  // One header, 44px tall, its top level with the Mind pane's header.
  const header = viewer.getByTestId("viewer-header");
  await expect(viewer.locator("header")).toHaveCount(1);
  const headerBox = await boxOf(header);
  const mindHeader = window.getByTestId("mind-area").locator(":scope > *").first();
  expect(headerBox.height).toBe(44);
  expect(headerBox.y).toBe((await boxOf(mindHeader)).y);

  // In order: the outline toggle, the name, page navigation, zoom, open, close.
  const controls = header.locator("button, input, [data-testid='viewer-title']");
  expect(
    await controls.evaluateAll((elements) =>
      elements.map((each) => each.getAttribute("data-testid")),
    ),
  ).toEqual([
    "pdf-outline-button",
    "viewer-title",
    "pdf-previous-page",
    "pdf-page-number",
    "pdf-next-page",
    "pdf-zoom-out",
    "pdf-zoom-in",
    "pdf-fit-width",
    "viewer-open-externally",
    "viewer-close",
  ]);
  await expect(window.getByTestId("viewer-title")).toHaveText("Guide");
  for (const control of await header.locator("button").all()) {
    await expect(control).toHaveAttribute("aria-label", /.+/);
    expect((await boxOf(control)).height).toBe(28);
  }

  // Every control is reached with Tab, in that order (Previous page is off on page 1).
  await header.getByTestId("pdf-outline-button").focus();
  const reached: (string | null)[] = [];
  for (let step = 0; step < 7; step++) {
    await window.keyboard.press("Tab");
    reached.push(
      await window.evaluate(() => document.activeElement?.getAttribute("data-testid") ?? null),
    );
  }
  expect(reached).toEqual([
    "pdf-page-number",
    "pdf-next-page",
    "pdf-zoom-out",
    "pdf-zoom-in",
    "pdf-fit-width",
    "viewer-open-externally",
    "viewer-close",
  ]);

  // "Open in default app" opens a copy named after the Document, as the sidebar's menu does.
  await interceptOpenPath(app);
  await header.getByTestId("viewer-open-externally").click();
  await expect.poll(() => pathsOpened(app)).toHaveLength(1);
  const [opened] = await pathsOpened(app);
  expect(basename(opened ?? "")).toBe("Guide.pdf");

  // It slides in as it opens, unless the system asks for less motion.
  const animation = () => viewer.evaluate((element) => getComputedStyle(element).animationName);
  expect(await animation()).toBe("viewer-pane-in");
  await window.getByTestId("viewer-close").click();
  await window.emulateMedia({ reducedMotion: "reduce" });
  await window.getByTestId("open-document").click();
  expect(await animation()).toBe("none");
  await app.close();
  // The app would remove the copy a day later; the test doesn't leave it behind.
  if (opened) await rm(dirname(opened), { recursive: true, force: true });
});

test("a Citation's quote is washed in its check's colour, with its mark beside the first highlighted line", async () => {
  const { report, notes } = await writeSources();
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [report, notes]);
  const viewer = window.getByTestId("viewer");
  const mark = viewer.getByTestId("viewer-quote-mark");

  // A found quote across a line break on page 2: green, its mark beside the first line.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Report"),
    pageFrom: 2,
    quote: "Revenue grew by ten percent in the third quarter",
    citation: { check: "found", number: 3 },
  });
  const page2 = viewer.locator('[data-page-number="2"]');
  const highlights = page2.locator("[data-quote-highlight]");
  await expect(highlights).toHaveCount(2);
  expect(await backgroundOf(highlights.first())).toBe(FOUND_WASH);
  await expect(mark).toHaveText("3");
  await expect(mark).toHaveAttribute("aria-label", "Citation 3: quote found");
  await expect(mark).toHaveAttribute("data-check", "found");
  await expectMarkBeside(mark, highlights.first(), page2);
  // It stays beside the quote through a zoom.
  await window.getByTestId("pdf-zoom-in").click();
  await expect(window.getByTestId("pdf-fit-width")).toHaveAttribute("aria-pressed", "false");
  await expect(async () => expectMarkBeside(mark, highlights.first(), page2)).toPass();

  // Opened anyway after "not found": amber, with the matching mark.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Report"),
    pageFrom: 2,
    quote: "Revenue grew by ten percent … led by exports. Costs stayed flat.",
    citation: { check: "not-found", number: 2 },
  });
  await expect(highlights).toHaveText([
    "Revenue grew by ten percent",
    "led by exports.",
    "Costs stayed flat.",
  ]);
  expect(await backgroundOf(highlights.first())).toBe(NOT_FOUND_WASH);
  await expect(mark).toHaveAttribute("data-check", "not-found");
  await expect(mark).toHaveText("2");
  await expect(async () => expectMarkBeside(mark, highlights.first(), page2)).toPass();

  // Without the Citation's number, the quote is washed green with no mark.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Report"),
    pageFrom: 2,
    quote: "Costs stayed flat.",
  });
  await expect(highlights).toHaveText(["Costs stayed flat."]);
  expect(await backgroundOf(highlights.first())).toBe(FOUND_WASH);
  await expect(mark).toHaveCount(0);

  // Markdown, in the Mind's serif on a page: each part of an ellipsis quote is washed,
  // and the mark sits beside the first.
  await openDocumentAt(window, {
    documentId: await documentIdOf(window, "Field notes"),
    quote: "Observation 119: nothing unusual today. ... Observation 120: nothing unusual today.",
    citation: { check: "found", number: 1 },
  });
  const text = window.getByTestId("viewer-text");
  const washed = text.locator("[data-quote-highlight]");
  await expect(washed).toHaveText([
    "Observation 119: nothing unusual today.",
    "Observation 120: nothing unusual today.",
  ]);
  expect(await backgroundOf(washed.first())).toBe(FOUND_WASH);
  await expect(mark).toHaveText("1");
  await expect(async () =>
    expectMarkBeside(mark, washed.first(), text.locator("article")),
  ).toPass();
  expect(
    await text.locator("h1").evaluate((element) => getComputedStyle(element).fontFamily),
  ).toMatch(/^"Source Serif 4"/);
  await app.close();
});
