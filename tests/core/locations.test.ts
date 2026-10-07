/**
 * Citation Locations (ADR-0011): a Citation names one Unit, or two
 * consecutive ones of one sheet or Document (a slide, a section, a block of
 * rows or lines), its quote is checked against their text, and its label
 * says where it points, narrowed to the rows or lines the quote covers.
 */
import { readFileSync } from "node:fs";
import { describe, expect, test } from "vitest";
import { checkCitation } from "../../src/core/answers/citations";
import { extractUnits } from "../../src/core/documents/formats";
import { lineUnits } from "../../src/core/documents/formats/text";
import {
  englishLocation,
  locationOf,
  parseRequestedLocation,
  quoteInUnits,
  resolveRequested,
} from "../../src/shared/locations";
import type { TextUnit } from "../../src/shared/units";
import {
  askAndFinish,
  citationsIn,
  citeFeedback,
  citingModel,
  type ShownPassage,
  setUpWithDocuments,
} from "../helpers/citations";

const fixture = (name: string) =>
  new Uint8Array(readFileSync(new URL(`../fixtures/formats/${name}`, import.meta.url)));

const WORKBOOK = await extractUnits("xlsx", fixture("Regional Revenue.xlsx"));
const DECK = await extractUnits("pptx", fixture("Quarterly Research Update.pptx"));
const REVIEW = await extractUnits("docx", fixture("Coastal Flood Risk Review.docx"));

const units = (all: readonly TextUnit[], from: number, to = from) =>
  all.filter((unit) => unit.page >= from && unit.page <= to);

/** The check, as the core runs it, of a quote in Units `from` to `to` of a Passage over all of them. */
function check(all: readonly TextUnit[], from: number, to: number, quote: string) {
  return checkCitation({
    quote,
    range: { pageFrom: from, pageTo: to },
    passage: { pageFrom: 1, pageTo: all.length },
    documentDeleted: false,
    pages: units(all, from, to),
  });
}

const FOUND = { check: "found", checkReason: null };
const NOT_FOUND = { check: "not-found", checkReason: "quote-not-on-pages" };

describe("The check of a quote in a spreadsheet's rows", () => {
  test("finds the cells' values in reading order, however their numbers are written", () => {
    expect(check(WORKBOOK, 1, 1, "West £350,200 £389,600")).toEqual(FOUND);
    expect(check(WORKBOOK, 1, 1, "West 350200 389600")).toEqual(FOUND);
    expect(check(WORKBOOK, 1, 1, "West 350,200.00 389 600")).toEqual(FOUND);
    expect(check(WORKBOOK, 1, 1, "Change on 2025 -4.2% 1.8%")).toEqual(FOUND);
  });

  test("doesn't find numbers that differ, a sign dropped, a cell skipped, or a paraphrase", () => {
    expect(check(WORKBOOK, 1, 1, "West £350,201 £389,600")).toEqual(NOT_FOUND);
    expect(check(WORKBOOK, 1, 1, "West £35,020 £389,600")).toEqual(NOT_FOUND);
    expect(check(WORKBOOK, 1, 1, "Change on 2025 4.2% 1.8%")).toEqual(NOT_FOUND);
    expect(check(WORKBOOK, 1, 1, "West £389,600")).toEqual(NOT_FOUND);
    expect(check(WORKBOOK, 1, 1, "The West earned £350,200 in Q1")).toEqual(NOT_FOUND);
  });

  test("doesn't find a quote in the wrong Unit, and two Units of different sheets break the rule", () => {
    const leeds = "Northern figures exclude the Leeds office, which reported late.";
    expect(check(WORKBOOK, 2, 2, leeds)).toEqual(FOUND);
    expect(check(WORKBOOK, 1, 1, leeds)).toEqual(NOT_FOUND);
    expect(check(WORKBOOK, 1, 2, leeds)).toEqual({
      check: "not-found",
      checkReason: "too-many-pages",
    });
  });

  test("is labelled with the rows the quote covers, not the whole block, the header left out", () => {
    const block = units(WORKBOOK, 1);
    const quote = "West £350,200 £389,600 £421,800 £466,100 £1,627,700 All regions";
    const location = locationOf(block, quoteInUnits(block, quote));

    expect(location).toEqual({ kind: "rows", sheet: "Revenue", from: 7, to: 8 });
    expect(englishLocation(location as never)).toBe("Revenue, rows 7–8");
    // Not found: the rows the record named, within the block, or else the whole block.
    expect(locationOf(block, null, { from: 4, to: 5 })).toMatchObject({ from: 4, to: 5 });
    expect(locationOf(block, null)).toMatchObject({ from: 1, to: 9 });
  });
});

describe("The check of a quote in a deck", () => {
  test("finds a quote from a slide's speaker notes on that slide", () => {
    expect(check(DECK, 3, 3, "Point at the red bar: that is the west.")).toEqual(FOUND);
    expect(check(DECK, 3, 3, "The western region grew fastest")).toEqual(FOUND);
  });

  test("doesn't find a quote on the wrong slide, or reworded", () => {
    expect(check(DECK, 4, 4, "The western region grew fastest")).toEqual(NOT_FOUND);
    expect(check(DECK, 3, 3, "The western region grew the fastest")).toEqual(NOT_FOUND);
    expect(check(DECK, 3, 4, "The western region grew fastest")).toEqual(FOUND);
    expect(check(DECK, 3, 5, "The western region grew fastest")).toEqual({
      check: "not-found",
      checkReason: "too-many-pages",
    });
  });
});

describe("The check of a quote in a Word file's sections", () => {
  const SENSITIVITY =
    "Raising the barrier crest by 40 centimetres halves the expected annual damage";

  test("finds it in its section, and is labelled with the heading it sits under", () => {
    expect(check(REVIEW, 5, 5, SENSITIVITY)).toEqual(FOUND);
    const both = units(REVIEW, 4, 5);
    expect(locationOf(both, quoteInUnits(both, SENSITIVITY))).toEqual({
      kind: "section",
      heading: "2.1 Sensitivity",
    });
  });

  test("doesn't find it in another section, or paraphrased", () => {
    expect(check(REVIEW, 4, 4, SENSITIVITY)).toEqual(NOT_FOUND);
    expect(check(REVIEW, 5, 5, "Raising the crest by 40 cm halves the damage")).toEqual(NOT_FOUND);
  });
});

describe("Lines of a text file", () => {
  test("are labelled with the lines the quote covers", () => {
    const text = Array.from({ length: 140 }, (_, index) => `Entry ${index + 1}: calm.`).join("\n");
    const block = lineUnits(text).filter((unit) => unit.page === 3);
    const ranges = quoteInUnits(block, "Entry 120: calm. Entry 121: calm.");

    expect(locationOf(block, ranges)).toEqual({ kind: "lines", from: 120, to: 121 });
    expect(englishLocation({ kind: "lines", from: 120, to: 134 })).toBe("lines 120–134");
  });
});

describe("A Location as the model writes it", () => {
  test("is read from its label, its plural, a cell range, or the old page form", () => {
    expect(parseRequestedLocation("p. 4")).toEqual({ kind: "page", from: 4, to: 4 });
    expect(parseRequestedLocation("pp. 4–5")).toEqual({ kind: "page", from: 4, to: 5 });
    expect(parseRequestedLocation("[slide 4]")).toEqual({ kind: "slide", from: 4, to: 4 });
    expect(parseRequestedLocation("slides 3-4")).toEqual({ kind: "slide", from: 3, to: 4 });
    expect(parseRequestedLocation("slide 3 (notes)")).toEqual({ kind: "slide", from: 3, to: 3 });
    expect(parseRequestedLocation("Revenue, rows 12–14")).toEqual({
      kind: "rows",
      sheet: "Revenue",
      from: 12,
      to: 14,
    });
    expect(parseRequestedLocation("rows 2 to 3")).toEqual({
      kind: "rows",
      sheet: null,
      from: 2,
      to: 3,
    });
    expect(parseRequestedLocation("Revenue!A7:F8")).toEqual({
      kind: "rows",
      sheet: "Revenue",
      from: 7,
      to: 8,
    });
    expect(parseRequestedLocation("§ 2.1 Sensitivity")).toEqual({
      kind: "section",
      heading: "2.1 Sensitivity",
    });
    expect(parseRequestedLocation("lines 120–134")).toEqual({ kind: "lines", from: 120, to: 134 });
    expect(parseRequestedLocation("somewhere")).toBeNull();
  });

  test("names the Units it covers among a Passage's, or none", () => {
    const resolve = (text: string, all: readonly TextUnit[]) =>
      resolveRequested(parseRequestedLocation(text) as never, all);

    expect(resolve("Revenue, rows 7–8", WORKBOOK)).toEqual({ from: 1, to: 1 });
    expect(resolve("Notes, rows 2–3", WORKBOOK)).toEqual({ from: 2, to: 2 });
    expect(resolve("Notes, rows 20–30", WORKBOOK)).toBeNull();
    expect(resolve("§ 2.1 Sensitivity", REVIEW)).toEqual({ from: 5, to: 5 });
    expect(resolve("§ 2.1", REVIEW)).toEqual({ from: 5, to: 5 });
    expect(resolve("§ Results › 2.1 Sensitivity", REVIEW)).toBeNull();
    expect(resolve("§ 2 Results 2.1 Sensitivity", REVIEW)).toEqual({ from: 5, to: 5 });
    expect(resolve("§ Notes", REVIEW)).toEqual({ from: 8, to: 8 });
    // A deck's "page 4" is its slide 4; pages mean nothing in a sheet.
    expect(resolve("p. 4", DECK)).toEqual({ from: 4, to: 4 });
    expect(resolve("p. 4", WORKBOOK)).toBeNull();
  });
});

describe("Citing Word, PowerPoint and Excel Documents in an Answer", { timeout: 30_000 }, () => {
  const byDocument = (passages: ShownPassage[], name: string) =>
    passages.find((passage) => passage.document === name)?.id ?? "none";

  test("the model sees where each Passage is, cites a Location, and each Citation stores its label", async () => {
    let shown: ShownPassage[] = [];
    const model = citingModel({
      query: "western region revenue barrier crest",
      records: (passages) => {
        shown = passages;
        return [
          {
            marker: 1,
            passage: byDocument(passages, "Deck"),
            location: "slide 3",
            quote: "Point at the red bar: that is the west.",
          },
          {
            marker: 2,
            passage: byDocument(passages, "Revenue"),
            location: "Revenue, rows 7–8",
            quote: "West 350,200 389,600",
          },
          {
            marker: 3,
            passage: byDocument(passages, "Review"),
            location: "§ 2.1 Sensitivity",
            quote: "Raising the barrier crest by 40 centimetres",
          },
          // The wrong slide: not found.
          {
            marker: 4,
            passage: byDocument(passages, "Deck"),
            location: "slide 4",
            quote: "The western region grew fastest",
          },
        ];
      },
      answer: "The west grew [^1], led revenue [^2], and the crest helps [^3]. Or not [^4].",
    });
    const { core, client, mind } = await setUpWithDocuments(model, [
      { name: "Deck.pptx", contents: fixture("Quarterly Research Update.pptx") },
      { name: "Revenue.xlsx", contents: fixture("Regional Revenue.xlsx") },
      { name: "Review.docx", contents: fixture("Coastal Flood Risk Review.docx") },
    ]);

    const { answerId } = await askAndFinish(core, client, mind.id, "Who grew fastest?");

    // The Passages say where they are, and where each new slide or section starts.
    const deck = shown.find((passage) => passage.document === "Deck");
    expect(deck?.pages).toBeNull();
    expect(deck?.location).toMatch(/^slides? \d/);
    expect(deck?.text).toContain("\n\n[slide 3] Regional growth");
    expect(deck?.text).toContain("[speaker notes] Point at the red bar");
    expect(shown.find((passage) => passage.document === "Revenue")?.location).toBe(
      "Revenue, rows 1–9",
    );

    const citations = citationsIn(client, answerId);
    expect(citations.map((citation) => [citation.location, citation.check])).toEqual([
      [{ kind: "slide", from: 3, to: 3 }, "found"],
      [{ kind: "rows", sheet: "Revenue", from: 7, to: 7 }, "found"],
      [{ kind: "section", heading: "2.1 Sensitivity" }, "found"],
      [{ kind: "slide", from: 4, to: 4 }, "not-found"],
    ]);
    expect(citations[1]).toMatchObject({ pageFrom: 1, pageTo: 1, documentName: "Revenue" });
    expect(citeFeedback(model)).toMatch(/\[\^4\]: the quote isn't word for word at slide 4/);
  });

  test("a Location the Passage doesn't have is 'not found', and the model is told where it can cite", async () => {
    const model = citingModel({
      query: "west revenue",
      records: (passages) => [
        {
          marker: 1,
          passage: byDocument(passages, "Revenue"),
          location: "Revenue, rows 40–41",
          quote: "West 350,200",
        },
      ],
      answer: "Revenue [^1].",
    });
    const { core, client, mind } = await setUpWithDocuments(model, [
      { name: "Revenue.xlsx", contents: fixture("Regional Revenue.xlsx") },
    ]);

    const { answerId } = await askAndFinish(core, client, mind.id, "What did the West earn?");

    expect(citationsIn(client, answerId)[0]).toMatchObject({
      check: "not-found",
      checkReason: "pages-outside-passage",
    });
    expect(citeFeedback(model)).toMatch(/give a location inside P\d \(Revenue, rows 1–9\)/);
  });
});
