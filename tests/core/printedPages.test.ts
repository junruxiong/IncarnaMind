/**
 * The name a PDF's page prints (#901): from the PDF's page labels, else from
 * the numbers its headers and footers print, when they keep a steady offset
 * from the PDF's page index. Otherwise the page is known by its index.
 */
import { describe, expect, test } from "vitest";
import { extractText } from "../../src/core/documents/extract";
import { printedNumbers, printedPageNames } from "../../src/core/documents/printedPages";
import { buildPdf, type PdfPage } from "../helpers/pdf";

const BODIES = [
  "Tides rise twice a day.",
  "Spring tides follow the full moon.",
  "Neap tides are the smallest.",
  "Harbours publish tide tables.",
  "Sailors read them every morning.",
  "Storm surges add to the tide.",
  "The Moon pulls on the oceans.",
  "The Sun pulls on them too.",
];

/** A page of text: a body line, then its footer lines. */
const page = (index: number, ...footer: string[]) =>
  [BODIES[index % BODIES.length] as string, ...footer].join("\n");

/** Pages whose footers print `printed`, one string for each page ("" for a page that prints none). */
const footed = (printed: readonly string[]) =>
  printed.map((footer, index) => page(index, ...(footer ? [footer] : [])));

describe("Page numbers read from footers and headers", () => {
  test("footers one lower than the PDF's pages give those numbers to the pages that print them", () => {
    expect(printedNumbers(footed(["", "1", "2", "3", "4"]))).toEqual([null, "1", "2", "3", "4"]);
  });

  test("every common way of printing a number is read", () => {
    const forms = ["- 1 -", "Page 2 of 40", "3 / 40", "第 4 页，共 40 页", "p. 5", "(6)", "| 7 |"];
    expect(printedNumbers(footed(["", ...forms]))).toEqual([
      null,
      ...forms.map((_, index) => String(index + 1)),
    ]);
  });

  test("a number beside text on a page's first or last line is read, headers as well as footers", () => {
    const pages = [
      page(0),
      ["Annual Report 2024 | 1", BODIES[1]].join("\n"),
      [BODIES[2], "2 | Annual Report 2024"].join("\n"),
      [BODIES[3], "Annual Report 2024    3"].join("\n"),
    ];
    expect(printedNumbers(pages)).toEqual([null, "1", "2", "3"]);
  });

  test("a page between the first and last that print a number gets the one the offset gives it", () => {
    expect(printedNumbers(footed(["", "1", "2", "", "4", "5", ""]))).toEqual([
      null,
      "1",
      "2",
      "3",
      "4",
      "5",
      null,
    ]);
  });

  test("a PDF that starts at a later printed page counts up from it", () => {
    expect(printedNumbers(footed(["100", "101", "102"]))).toEqual(["100", "101", "102"]);
  });

  test("fewer than three pages with the same offset are no steady offset", () => {
    expect(printedNumbers(footed(["", "1", "2", "", ""]))).toEqual([null, null, null, null, null]);
  });

  test("one page that prints another number than the offset gives it spoils it for every page", () => {
    expect(printedNumbers(footed(["", "1", "2", "3", "9", "5"]))).toEqual(Array(6).fill(null));
  });

  test("two numberings that run on at different offsets are no steady offset", () => {
    const printed = ["", "1", "2", "3", "1", "2", "3"];
    expect(printedNumbers(footed(printed))).toEqual(Array(7).fill(null));
  });

  test("numbers that don't follow the page index, and pages that print none, give nothing", () => {
    expect(printedNumbers(footed(["12", "4", "88", "7", "3"]))).toEqual(Array(5).fill(null));
    expect(printedNumbers(footed(["", "", "", ""]))).toEqual(Array(4).fill(null));
  });

  test("a figure's number, a year or a total in the body is not a page number", () => {
    const pages = [1, 2, 3, 4].map((number) =>
      [BODIES[number], `Figure ${number}`, "Turnover rose to 1,234 in 2024.", "Up 12%"].join("\n"),
    );
    expect(printedNumbers(pages)).toEqual(Array(4).fill(null));
  });

  test("Chinese pages are read as well", () => {
    const pages = ["", "第 1 页", "第 2 页，共 3 页", "第 3 页"].map((footer) =>
      ["大潮在新月和满月时出现。", footer].filter(Boolean).join("\n"),
    );
    expect(printedNumbers(pages)).toEqual([null, "1", "2", "3"]);
  });
});

describe("The name of each page", () => {
  const texts = footed(["", "1", "2", "3", "4"]);

  test("a PDF's page labels name its pages, roman numerals and all", () => {
    const labels = ["i", "ii", "iii", "1", "2"];
    expect(printedPageNames(labels, texts)).toEqual(labels);
  });

  test("labels win over what the footers print", () => {
    expect(printedPageNames(["A-1", "A-2", "A-3", "A-4", "A-5"], texts)).toEqual([
      "A-1",
      "A-2",
      "A-3",
      "A-4",
      "A-5",
    ]);
  });

  test("labels that give every page its own index are a default, and footers speak instead", () => {
    expect(printedPageNames(["1", "2", "3", "4", "5"], texts)).toEqual([null, "1", "2", "3", "4"]);
  });

  test("an empty label, or one too long to be a page's name, names nothing", () => {
    expect(printedPageNames(["", "ii", "x".repeat(40), "1", "\u0000"], texts)).toEqual([
      null,
      "ii",
      null,
      "1",
      null,
    ]);
  });

  test("a PDF with no labels, and footers with no steady offset, has no names", () => {
    expect(printedPageNames(null, footed(["", "", "", ""]))).toEqual([null, null, null, null]);
    expect(printedPageNames(null, footed(["2", "3", "9"]))).toEqual([null, null, null]);
  });

  test("footers that print the PDF's own page index name nothing new", () => {
    expect(printedPageNames(null, footed(["1", "2", "3", "4"]))).toEqual([null, null, null, null]);
  });
});

/** A PDF page of lines. */
const lines = (...text: string[]): PdfPage => ({ lines: text });

describe("A PDF's pages, as extracted", () => {
  test("page labels become each page's printed name: roman front matter, then 1, 2, 3", async () => {
    const pages = Array.from({ length: 15 }, (_, index) => lines(BODIES[index % 8] as string));
    const pdf = buildPdf(pages, {
      pageLabels: [
        { page: 1, style: "r" },
        { page: 13, style: "D" },
      ],
    });

    const { units } = await extractText("pdf", pdf);

    expect(units[11]).toMatchObject({ page: 12, label: { printed: "xii" } });
    expect(units[14]).toMatchObject({ page: 15, label: { printed: "3" } });
    expect(units[0]?.label).toEqual({ printed: "i" });
  });

  test("page labels pdf.js can't read are ignored, and the PDF is still read", async () => {
    const pages = Array.from({ length: 3 }, (_, index) => lines(BODIES[index] as string));
    const pdf = buildPdf(pages, { pageLabels: [{ page: 1, style: "X" }] });

    const { units } = await extractText("pdf", pdf);

    expect(units.map((unit) => [unit.text, unit.label])).toEqual(
      pages.map((each) => [each.lines?.[0], null]),
    );
  });

  test("a prefix and a start number are in the name", async () => {
    const pages = Array.from({ length: 4 }, (_, index) => lines(BODIES[index] as string));
    const pdf = buildPdf(pages, {
      pageLabels: [{ page: 1, style: "D", prefix: "A-", start: 3 }],
    });

    const { units } = await extractText("pdf", pdf);

    expect(units.map((unit) => unit.label?.printed)).toEqual(["A-3", "A-4", "A-5", "A-6"]);
  });

  test("with no labels, footers one lower than the PDF's pages give the pages their printed numbers", async () => {
    const pdf = buildPdf([
      lines("Annual Report", "The year in numbers"),
      lines(BODIES[1] as string, "Strategic report", "1"),
      lines(BODIES[2] as string, "Strategic report", "2"),
      lines(BODIES[3] as string, "Strategic report", "3"),
      lines(BODIES[4] as string, "Strategic report", "4"),
    ]);

    const { units } = await extractText("pdf", pdf);

    expect(units.map((unit) => unit.label?.printed ?? null)).toEqual([null, "1", "2", "3", "4"]);
    // The text is as it was extracted: the footer is for Processing to take off.
    expect(units[1]?.text).toContain("Strategic report");
  });

  test("a PDF with inconsistent numbers, or none, keeps its pages as they are, with no label", async () => {
    const inconsistent = buildPdf(
      ["1", "2", "3", "9", "5"].map((footer, index) => lines(BODIES[index] as string, footer)),
    );
    const bare = buildPdf(Array.from({ length: 4 }, (_, index) => lines(BODIES[index] as string)));

    for (const pdf of [inconsistent, bare]) {
      const { units } = await extractText("pdf", pdf);
      expect(units.map((unit) => unit.label)).toEqual(Array(units.length).fill(null));
    }
  });
});
