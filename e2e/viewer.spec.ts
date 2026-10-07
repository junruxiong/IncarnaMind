import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  darkPixels,
  dismissChatSetup,
  documentIdOf,
  launchApp,
  openDocumentAt,
  removeDataFolder,
  widthOf,
} from "./app";

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
  await expect(window.getByTestId("pdf-toolbar")).toContainText("of 3");

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
  await writeFile(
    guide,
    buildPdf(
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
    ),
  );
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

  // Clicking an entry goes to its page.
  await entries.filter({ hasText: "Troubleshooting" }).click();
  await expect(pageNumber).toHaveValue("6");
  await expect(viewer.locator('[data-page-number="6"] .textLayer')).toContainText("Guide page 6");
  await entries.filter({ hasText: "Reference" }).click();
  await expect(pageNumber).toHaveValue("3");
  await expect(viewer.locator('[data-page-number="3"]')).toBeInViewport();

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

  await item.hover();
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
