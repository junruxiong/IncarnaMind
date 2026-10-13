import { describe, expect, test } from "vitest";
import { excerptOf, findTerms, searchTerms } from "../../src/core/wordSearch";

const marked = (text: string, ranges: readonly { start: number; end: number }[]) =>
  ranges.map((range) => text.slice(range.start, range.end));

describe("finding the words of a search in text", () => {
  const long = `The harbour pilot boarded at dawn. ${"Tides were logged. ".repeat(500)}The harbour pilot left.`;

  test("finds every place in the whole text, unless told to stop soon after the first", () => {
    const terms = searchTerms("harbour pilot");
    expect(marked(long, findTerms(long, terms))).toEqual(["harbour", "pilot", "harbour", "pilot"]);
    expect(marked(long, findTerms(long, terms, 160))).toEqual(["harbour", "pilot"]);
  });

  test("a term of several words that starts before one found first still counts", () => {
    // As "聯合國大會" folds to two words: a term can be a run of them.
    const text = "The night shift harbour pilot boarded";
    expect(marked(text, findTerms(text, [["night", "shift", "harbour"], ["shift"]], 20))).toEqual([
      "night shift harbour",
    ]);
  });

  test("an excerpt of a long Passage marks the words in it, with the rest cut", () => {
    const { excerpt, highlights } = excerptOf(long, searchTerms("pilot"));
    expect(excerpt.startsWith("The harbour pilot boarded at dawn.")).toBe(true);
    expect(excerpt.endsWith("…")).toBe(true);
    expect(marked(excerpt, highlights)).toEqual(["pilot"]);
  });
});
