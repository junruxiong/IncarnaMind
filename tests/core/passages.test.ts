import { describe, expect, test } from "vitest";
import {
  approximateTokens,
  buildPassages,
  PASSAGE_PARAMETERS,
} from "../../src/core/documents/passages";

const sentences = (count: number, from = 0) =>
  Array.from({ length: count }, (_, index) => `Sentence number ${from + index} fills the page.`);

/**
 * Lines as pdf.js gives them, each `characters` long with its line break and
 * labelled "Line N", so a Passage's lines can be read back.
 */
const lines = (count: number, characters = 40, from = 0) =>
  Array.from({ length: count }, (_, index) =>
    `Line ${from + index} `.padEnd(characters - 2, "x").concat("."),
  );

/** The numbers of the lines in a Passage, in order. */
const lineNumbers = (text: string) =>
  [...text.matchAll(/Line (\d+) /g)].map((match) => Number(match[1]));

/** The part of `previous` that `next` repeats: `next` starts inside `previous`. */
function overlapOf(previous: string, next: string): string {
  for (let start = 0; start < previous.length; start++) {
    const tail = previous.slice(start);
    if (next.startsWith(tail.trim())) return tail.trim();
  }
  return "";
}

describe("Passages", () => {
  test("approximate tokens: one per CJK character, one per four other characters, spaces included", () => {
    expect(approximateTokens("")).toBe(0);
    expect(approximateTokens("abcd")).toBe(1);
    expect(approximateTokens("ab  \n\t cd")).toBe(2.25);
    expect(approximateTokens("如 GPT-3 含")).toBe(3.75);
    expect(approximateTokens("注意力")).toBe(3);
  });

  test("a short text is one Passage", () => {
    expect(buildPassages([{ page: null, text: "  Hello, world.  " }])).toEqual([
      {
        position: 0,
        pageFrom: null,
        pageTo: null,
        windowFrom: 0,
        windowTo: 0,
        text: "Hello, world.",
      },
    ]);
  });

  test("no text gives no Passages", () => {
    expect(buildPassages([])).toEqual([]);
    expect(
      buildPassages([
        { page: 1, text: " \n " },
        { page: 2, text: "" },
      ]),
    ).toEqual([]);
  });

  // The retrieval prototype's boundaries (ADR-0009, #31): whole lines fill a
  // Passage up to 500 tokens, and the next one starts with the last lines of
  // it, at most 200 tokens of them. A different builder at the same sizes
  // lost two of the ten English evaluation questions.
  test("Passages are whole lines up to 500 tokens, each starting with at most 200 tokens of the one before", () => {
    // 200 lines of 10 tokens: 50 lines a Passage, starting every 30 lines.
    const passages = buildPassages([{ page: 1, text: lines(200).join("\n") }]);

    expect(passages.map((passage) => lineNumbers(passage.text))).toEqual(
      [0, 30, 60, 90, 120, 150].map((first) =>
        Array.from({ length: 50 }, (_, index) => first + index),
      ),
    );
    for (const passage of passages) {
      expect(approximateTokens(passage.text)).toBeLessThanOrEqual(PASSAGE_PARAMETERS.maxTokens);
    }
  });

  test("a Passage doesn't stop at a page break or a sentence end while lines still fit", () => {
    // Pages of 30 lines (300 tokens), every line ending a sentence.
    const pages = [0, 1, 2, 3].map((page) => ({
      page: page + 1,
      text: lines(30, 40, page * 30).join("\n"),
    }));
    const passages = buildPassages(pages);

    expect(lineNumbers(passages[0]?.text ?? "")).toEqual(
      Array.from({ length: 50 }, (_, index) => index),
    );
    expect(passages[0]).toMatchObject({ pageFrom: 1, pageTo: 2 });
    // Pages are joined by a blank line: the Citation check finds where a page starts by it.
    expect(passages[0]?.text).toContain(`\n\n${pages[1]?.text.split("\n")[0]}`);
    expect(passages[1]).toMatchObject({ pageFrom: 2, pageTo: 3 });
    expect(lineNumbers(passages[1]?.text ?? "")[0]).toBe(30);
  });

  test("a line counts whole tokens, as the prototype counted it", () => {
    // 41 characters with the line break: 10.25 tokens, counted as 11. So a
    // Passage holds 45 lines (495 tokens), not 48, and the next starts 18 lines back.
    const passages = buildPassages([{ page: null, text: lines(100, 41).join("\n") }]);

    expect(lineNumbers(passages[0]?.text ?? "")).toEqual(
      Array.from({ length: 45 }, (_, index) => index),
    );
    expect(lineNumbers(passages[1]?.text ?? "")[0]).toBe(27);
  });

  test("a line longer than a Passage is split into words, keeping the overlap", () => {
    // A TXT paragraph on a single line.
    const all = sentences(200);
    const passages = buildPassages([{ page: null, text: all.join(" ") }]);

    expect(passages.length).toBeGreaterThan(3);
    for (const passage of passages) {
      expect(approximateTokens(passage.text)).toBeLessThanOrEqual(PASSAGE_PARAMETERS.maxTokens);
    }
    for (let index = 1; index < passages.length; index++) {
      const overlap = overlapOf(passages[index - 1]?.text ?? "", passages[index]?.text ?? "");
      expect(approximateTokens(overlap)).toBeLessThanOrEqual(PASSAGE_PARAMETERS.overlapTokens);
      expect(approximateTokens(overlap)).toBeGreaterThanOrEqual(
        PASSAGE_PARAMETERS.overlapTokens / 2,
      );
    }
    // Nothing is lost.
    for (const sentence of all) {
      expect(passages.some((passage) => passage.text.includes(sentence))).toBe(true);
    }
  });

  test("each Passage records a sliding window of itself and the two before it", () => {
    const passages = buildPassages([{ page: null, text: lines(300).join("\n") }]);

    expect(passages.length).toBeGreaterThan(3);
    expect(passages.map(({ windowFrom, windowTo }) => [windowFrom, windowTo])).toEqual(
      passages.map((_, position) => [Math.max(0, position - 2), position]),
    );
  });

  test("Passages record the pages they cover, crossing page breaks and skipping empty pages", () => {
    const passages = buildPassages([
      { page: 1, text: sentences(20, 100).join("\n") },
      { page: 2, text: "   " },
      { page: 3, text: sentences(20, 200).join("\n") },
      { page: 4, text: sentences(20, 300).join("\n") },
    ]);

    // Sentences 100–119 are on page 1, 200–219 on page 3 and 300–319 on page 4.
    const pageOf = (sentence: string | undefined) =>
      [0, 1, 3, 4][Math.floor(Number(sentence) / 100)];
    expect(passages[0]?.pageFrom).toBe(1);
    expect(passages.at(-1)?.pageTo).toBe(4);
    for (const passage of passages) {
      const numbers = [...passage.text.matchAll(/Sentence number (\d+)/g)].map((match) => match[1]);
      expect(passage.pageFrom).toBe(pageOf(numbers[0]));
      expect(passage.pageTo).toBe(pageOf(numbers.at(-1)));
    }
    // One runs on from page 1 past the empty page 2.
    expect(passages[0]?.pageTo).toBeGreaterThanOrEqual(3);
  });

  test("Chinese text counts about one token per character", () => {
    const text = Array.from(
      { length: 60 },
      () => "注意力机制让模型能够关注输入中最重要的部分。",
    ).join("\n");
    const passages = buildPassages([{ page: null, text }]);

    expect(passages.length).toBeGreaterThan(2);
    for (const passage of passages) {
      expect(approximateTokens(passage.text)).toBeLessThanOrEqual(PASSAGE_PARAMETERS.maxTokens);
      // Only the full stops and line breaks count less than one.
      expect(Array.from(passage.text).length).toBeLessThan(PASSAGE_PARAMETERS.maxTokens * 1.1);
      expect(passage.text.endsWith("。")).toBe(true);
    }
  });

  test("a 'word' longer than a Passage is cut into pieces", () => {
    const blob = "x".repeat(5000);
    const passages = buildPassages([{ page: null, text: `Before. ${blob} After.` }]);

    expect(passages.length).toBeGreaterThan(3);
    for (const passage of passages) {
      expect(approximateTokens(passage.text)).toBeLessThanOrEqual(PASSAGE_PARAMETERS.maxTokens);
    }
    expect(passages.map((passage) => passage.text).join("")).toContain("After.");
  });
});
