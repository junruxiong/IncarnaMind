import { describe, expect, test } from "vitest";
import {
  approximateTokens,
  buildPassages,
  PASSAGE_PARAMETERS,
} from "../../src/core/documents/passages";

const sentences = (count: number, from = 0) =>
  Array.from({ length: count }, (_, index) => `Sentence number ${from + index} fills the page.`);

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

  test("long text gives overlapping Passages within the token budget, ending at sentences", () => {
    const all = sentences(200);
    const passages = buildPassages([{ page: null, text: all.join(" ") }]);

    expect(passages.length).toBeGreaterThan(3);
    for (const passage of passages) {
      expect(approximateTokens(passage.text)).toBeLessThanOrEqual(PASSAGE_PARAMETERS.maxTokens);
      expect(passage.text.endsWith(".")).toBe(true);
    }
    for (let index = 1; index < passages.length; index++) {
      const overlap = overlapOf(passages[index - 1]?.text ?? "", passages[index]?.text ?? "");
      expect(overlap).not.toBe("");
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
    const passages = buildPassages([{ page: null, text: sentences(150).join(" ") }]);

    expect(passages.map(({ windowFrom, windowTo }) => [windowFrom, windowTo])).toEqual(
      passages.map((_, position) => [Math.max(0, position - 2), position]),
    );
  });

  test("Passages record the pages they cover, crossing page breaks and skipping empty pages", () => {
    const passages = buildPassages([
      { page: 1, text: sentences(20, 100).join(" ") },
      { page: 2, text: "   " },
      { page: 3, text: sentences(20, 200).join(" ") },
      { page: 4, text: sentences(20, 300).join(" ") },
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
    expect(passages.some((passage) => passage.pageFrom === 1 && passage.pageTo === 3)).toBe(true);
  });

  test("Chinese text counts about one token per character", () => {
    const text = "注意力机制让模型能够关注输入中最重要的部分。".repeat(60);
    const passages = buildPassages([{ page: null, text }]);

    expect(passages.length).toBeGreaterThan(2);
    for (const passage of passages) {
      expect(approximateTokens(passage.text)).toBeLessThanOrEqual(PASSAGE_PARAMETERS.maxTokens);
      // Only the full stops count less than one.
      expect(Array.from(passage.text).length).toBeLessThan(PASSAGE_PARAMETERS.maxTokens * 1.05);
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
