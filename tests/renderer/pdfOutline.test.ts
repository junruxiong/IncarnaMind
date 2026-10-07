import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { describe, expect, onTestFinished, test } from "vitest";
import {
  destinationPage,
  loadOutline,
  type OutlineSource,
} from "../../src/renderer/src/viewer/outline";
import { buildPdf, type PdfOutlineEntry } from "../helpers/pdf";

/** Four pages, each saying which it is. */
const PAGES = [1, 2, 3, 4].map((page) => ({ lines: [`Page ${page}`] }));

/** Opens a PDF with pdf.js's Node build, as the viewer does with its browser build. */
async function open(bytes: Uint8Array) {
  const task = getDocument({ data: bytes, verbosity: 0 });
  onTestFinished(() => task.destroy());
  return task.promise;
}

describe("a PDF's outline", () => {
  test("is read with each entry's page, however the PDF says where it goes", async () => {
    const outline: PdfOutlineEntry[] = [
      { title: "Introduction", page: 1 },
      {
        title: "Results",
        page: 2,
        via: "named",
        open: true,
        items: [
          { title: "Revenue", page: 2, via: "action" },
          {
            title: "Costs  and\tspending",
            page: 3,
            items: [{ title: "Salaries", page: 4, via: "named" }],
          },
        ],
      },
      { title: "Project website", url: "https://example.com/project" },
    ];
    const pdf = await open(buildPdf(PAGES, { outline }));

    expect(await loadOutline(pdf)).toEqual([
      { title: "Introduction", page: 1, open: false, items: [] },
      {
        title: "Results",
        page: 2,
        open: true,
        items: [
          { title: "Revenue", page: 2, open: false, items: [] },
          {
            title: "Costs and spending",
            page: 3,
            open: false,
            items: [{ title: "Salaries", page: 4, open: false, items: [] }],
          },
        ],
      },
      // A web link goes to no page.
      { title: "Project website", page: null, open: false, items: [] },
    ]);
  });

  test("is empty for a PDF without one", async () => {
    const pdf = await open(buildPdf(PAGES));
    expect(await loadOutline(pdf)).toEqual([]);
  });
});

describe("an outline entry's destination", () => {
  /** A PDF of `numPages` pages whose page objects are numbered 10, 11, …, with one named destination. */
  const fakePdf = (numPages: number): OutlineSource => ({
    numPages,
    getOutline: async () => null,
    getDestination: async (id) =>
      id === "chapter-2" ? [{ num: 11, gen: 0 }, { name: "Fit" }] : null,
    getPageIndex: async ({ num }) => {
      if (num < 10 || num >= 10 + numPages) throw new Error("Invalid page reference.");
      return num - 10;
    },
  });

  test.each([
    {
      case: "an explicit destination's page reference",
      dest: [{ num: 12, gen: 0 }, { name: "XYZ" }],
      page: 3,
    },
    { case: "a page index, as some PDFs give it", dest: [1, { name: "Fit" }], page: 2 },
    { case: "a named destination", dest: "chapter-2", page: 2 },
  ])("resolves $case", async ({ dest, page }) => {
    expect(await destinationPage(fakePdf(4), dest)).toBe(page);
  });

  test.each([
    { case: "no destination", dest: null },
    { case: "an unknown named destination", dest: "appendix" },
    { case: "an empty destination", dest: [] },
    { case: "a reference to no page", dest: [{ num: 99, gen: 0 }, { name: "Fit" }] },
    { case: "a page index past the last page", dest: [4, { name: "Fit" }] },
    { case: "a negative page index", dest: [-1, { name: "Fit" }] },
    { case: "something that isn't a page", dest: [{ name: "XYZ" }] },
  ])("goes nowhere for $case", async ({ dest }) => {
    expect(await destinationPage(fakePdf(4), dest)).toBeNull();
  });
});
