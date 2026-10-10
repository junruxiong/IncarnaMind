/**
 * Keyword search's words (ADR-0009): the one place the keyword index's text
 * and a query's terms are made, where traditional Chinese characters are read
 * as simplified ones.
 */
import { describe, expect, test } from "vitest";
import { keywordQuery, keywordTerms, keywordText } from "../../src/core/documents/keywords";
import { SIMPLIFIED } from "../../src/shared/hanVariants";
import { unfoldedKeywords } from "../helpers/documents";

const TRADITIONAL = "可持續發展目標是聯合國制定的十七個全球發展目標。";
const SIMPLIFIED_TEXT = "可持续发展目标是联合国制定的十七个全球发展目标。";

/** Each traditional character as its simplified one, after the text was split. */
const foldedAfter = (words: string) =>
  [...words].map((char) => SIMPLIFIED.get(char) ?? char).join("");

describe("Keyword search's words", () => {
  test("traditional Chinese characters are read as simplified ones, in the index and in queries alike", () => {
    expect(keywordText(TRADITIONAL)).toBe(keywordText(SIMPLIFIED_TEXT));
    expect(keywordText(TRADITIONAL)).not.toMatch(/[續發標聯國個]/);
    expect(keywordQuery("聯合國的目標")).toBe(keywordQuery("联合国的目标"));
  });

  test("the characters are folded before the text is split, so both scripts split into the same words", () => {
    const traditional = "聯合國大會通過了這項決議";
    const simplified = "联合国大会通过了这项决议";
    expect(keywordText(traditional)).toBe(keywordText(simplified));
    // Split first, they don't: the traditional text has other words to match.
    expect(foldedAfter(unfoldedKeywords(traditional))).not.toBe(keywordText(simplified));
  });

  test("a traditional stopword is left out of a query like its simplified one", () => {
    expect(keywordTerms("什麼是聯合國？")).toEqual(keywordTerms("什么是联合国？"));
    expect(keywordTerms("什麼")).toEqual([]);
  });

  test("text with no traditional characters gives the words it gave before", () => {
    for (const text of [
      SIMPLIFIED_TEXT,
      "The Transformer relies entirely on self-attention.",
      "カタカナとひらがなの文",
      "한국어 문장은 띄어 씁니다",
    ]) {
      expect(keywordText(text)).toBe(unfoldedKeywords(text));
    }
  });
});
