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

const NONE = (count: number) => Array(count).fill(null);

/** Pages with `lines` before the body (headers) and `footers` after it, one string each ("" for none). */
const framed = (headers: readonly string[], footers: readonly string[] = []) =>
  Array.from({ length: Math.max(headers.length, footers.length) }, (_, index) =>
    [headers[index], BODIES[index % BODIES.length], footers[index]]
      .filter((line): line is string => Boolean(line))
      .join("\n"),
  );

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

  test("a number beside the text of a running footer or header is read", () => {
    const running = (end: "header" | "footer", ...numbers: number[]) =>
      framed(
        end === "header" ? ["", ...numbers.map((n) => `Annual Report 2024 | ${n}`)] : [],
        end === "footer" ? ["", ...numbers.map((n) => `${n} | Annual Report 2024`)] : [],
      );
    expect(printedNumbers(running("footer", 1, 2, 3, 4))).toEqual([null, "1", "2", "3", "4"]);
    expect(printedNumbers(running("header", 1, 2, 3, 4))).toEqual([null, "1", "2", "3", "4"]);
  });

  test("a footer's year above the number, or beside it, is not the number", () => {
    const above = framed([], ["", ...[1, 2, 3, 4].map((n) => `Annual Report 2023\n${n}`)]);
    const beside = framed([], ["", ...[1, 2, 3, 4].map((n) => `Annual Report 2023 | ${n}`)]);
    expect(printedNumbers(above)).toEqual([null, "1", "2", "3", "4"]);
    expect(printedNumbers(beside)).toEqual([null, "1", "2", "3", "4"]);
  });

  test("a cover that prints a year alone is let off, and the pages after it are named", () => {
    const pages = ["Annual Report\n2023", ...[1, 2, 3, 4].map((n) => page(n, `${n}`))];
    expect(printedNumbers(pages)).toEqual([null, "1", "2", "3", "4"]);
  });

  test("roman front matter that prints no digits is skipped, and the pages after it are named", () => {
    const printed = ["", "i", "ii", "iii", "1", "2", "3", "4"];
    expect(printedNumbers(footed(printed))).toEqual([...NONE(4), "1", "2", "3", "4"]);
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

  test("a header and a footer that print the same numbers agree", () => {
    const both = framed(
      ["", ...[1, 2, 3, 4].map((n) => `${n}`)],
      ["", ...[1, 2, 3, 4].map((n) => `${n}`)],
    );
    expect(printedNumbers(both)).toEqual([null, "1", "2", "3", "4"]);
  });

  test("a header and a footer that print numbers at different offsets give nothing", () => {
    const both = framed(
      ["", ...[1, 2, 3, 4].map((n) => `${n}`)],
      ["", ...[2, 3, 4, 5].map((n) => `${n}`)],
    );
    expect(printedNumbers(both)).toEqual(NONE(5));
  });

  test("fewer than three pages with the same offset are no steady offset", () => {
    expect(printedNumbers(footed(["", "1", "2", "", ""]))).toEqual(NONE(5));
  });

  test("numbers at both ends, each end short of three pages, are not added together", () => {
    const pages = framed(["", "", "", "3", "4", ""], ["", "1", "2", "", "", ""]);
    expect(printedNumbers(pages)).toEqual(NONE(6));
  });

  test("numbers on a few pages scattered over many are not a steady offset", () => {
    const printed = Array(12).fill("");
    printed[1] = "1";
    printed[5] = "5";
    printed[10] = "10";
    expect(printedNumbers(footed(printed))).toEqual(NONE(12));
  });

  test("one page that prints another number than the offset gives it spoils it for every page", () => {
    expect(printedNumbers(footed(["", "1", "2", "3", "9", "5"]))).toEqual(NONE(6));
  });

  test("one end with numbers but no steady offset gives nothing, whatever the other end prints", () => {
    const stray = ["", "", "7", "", "3", "", ""];
    const steady = ["", "1", "2", "3", "4", "5", "6"];
    expect(printedNumbers(framed(stray, steady))).toEqual(NONE(7));
    expect(printedNumbers(framed(steady, stray))).toEqual(NONE(7));
  });

  test("an end whose only number is on the cover doesn't spoil the other end", () => {
    const pages = framed(["2023", "", "", "", "", ""], ["", "1", "2", "3", "4", "5"]);
    expect(printedNumbers(pages)).toEqual([null, "1", "2", "3", "4", "5"]);
  });

  test("a footnote's number before its text, one a page, is not a running footer", () => {
    const notes = ["", ...[1, 2, 3, 4, 5].map((n) => `${n} Ibid., p. 4`)];
    expect(printedNumbers(footed(notes))).toEqual(NONE(6));
  });

  test("a number beside text with only a space between them is not a page number", () => {
    const spaced = ["", ...[1, 2, 3, 4].map((n) => `Annual Report 2024 ${n}`)];
    expect(printedNumbers(footed(spaced))).toEqual(NONE(5));
    const before = ["", ...[1, 2, 3, 4].map((n) => `${n} Annual Report 2024`)];
    expect(printedNumbers(footed(before))).toEqual(NONE(5));
  });

  test("a page in a gap that prints its own name is not named from the offset", () => {
    for (const own of ["A-1", "i", "ii", "5-12"]) {
      const printed = ["", "1", "2", own, "4", "5"];
      expect(printedNumbers(footed(printed))).toEqual([null, "1", "2", null, "4", "5"]);
    }
  });

  test("a number beside text that disagrees spoils it as a bare one does", () => {
    const footers = ["", "Annual Report | 1", "Annual Report | 2", "Annual Report | 3"];
    expect(printedNumbers(footed([...footers, "Annual Report | 9", "Annual Report | 5"]))).toEqual(
      NONE(6),
    );
  });

  test("two numberings that run on at different offsets are no steady offset", () => {
    expect(printedNumbers(footed(["", "1", "2", "3", "1", "2", "3"]))).toEqual(NONE(7));
  });

  test("a second part that starts again at 1 spoils the first part's offset", () => {
    expect(printedNumbers(footed(["", "1", "2", "3", "4", "1", "2", "3"]))).toEqual(NONE(8));
  });

  test("two pages to a sheet, with two numbers in the footer, give nothing", () => {
    const sheets = [1, 2, 3, 4, 5, 6].map((sheet) =>
      page(sheet, `${2 * sheet}`, `${2 * sheet + 1}`),
    );
    expect(printedNumbers(sheets)).toEqual(NONE(6));
  });

  test("numbers that don't follow the page index, and pages that print none, give nothing", () => {
    expect(printedNumbers(footed(["12", "4", "88", "7", "3"]))).toEqual(NONE(5));
    expect(printedNumbers(footed(["", "", "", ""]))).toEqual(NONE(4));
  });

  test("a figure's number, a year or a total in the body is not a page number", () => {
    const pages = [1, 2, 3, 4].map((number) =>
      [BODIES[number], `Figure ${number}`, "Turnover rose to 1,234 in 2024.", "Up 12%"].join("\n"),
    );
    expect(printedNumbers(pages)).toEqual(NONE(4));
  });

  test("digits after a hyphen are never a page number", () => {
    const hyphenated = (label: (n: number) => string) => footed(["", ...[1, 2, 3, 4].map(label)]);
    expect(printedNumbers(hyphenated((n) => `Halden Manual | A-${n + 2}`))).toEqual(NONE(5));
    expect(printedNumbers(hyphenated((n) => `III-${n + 3}`))).toEqual(NONE(5));
    expect(printedNumbers(hyphenated((n) => `Page 2-${n + 13}`))).toEqual(NONE(5));
    expect(printedNumbers(hyphenated((n) => `Manual 5-${n + 11}`))).toEqual(NONE(5));
    expect(printedNumbers(hyphenated((n) => `5-${n + 11}`))).toEqual(NONE(5));
  });

  test("numbered headings on consecutive pages are not page numbers", () => {
    const headings = framed(["", "", "", "3 Methods", "4 Results", "5 Discussion", ""]);
    expect(printedNumbers(headings)).toEqual(NONE(7));
  });

  test("captions on consecutive pages are not page numbers", () => {
    const captions = ["", "", "", "Figure 3", "Figure 4", "Figure 5", ""];
    expect(printedNumbers(footed(captions))).toEqual(NONE(7));
    const tables = ["", "", "", "Table 3", "Table 4", "Table 5", ""];
    expect(printedNumbers(footed(tables))).toEqual(NONE(7));
  });

  test("footnotes at the foot of consecutive pages are not page numbers", () => {
    const notes = ["", "", "", "3 See Smith (2020).", "4 See Jones (2019).", "5 Ibid.", ""];
    expect(printedNumbers(footed(notes))).toEqual(NONE(7));
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

  test("a label with a hyphen is kept whole, as the PDF gives it", () => {
    const labels = ["5-12", "5-13", "5-14", "5-15", "5-16"];
    expect(printedPageNames(labels, texts)).toEqual(labels);
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

  test("hyphenated footers give no names: a hyphen's digits are never a page number", async () => {
    const pdf = buildPdf(
      [12, 13, 14, 15, 16].map((n, index) =>
        lines(BODIES[index] as string, "Manual", `Manual 5-${n}`),
      ),
    );

    const { units } = await extractText("pdf", pdf);

    expect(units.map((unit) => unit.label)).toEqual(NONE(5));
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
