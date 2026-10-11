import { describe, expect, test } from "vitest";
import { readEditReply, sameText } from "../../src/core/edits/prompt";
import { diffSteps, hunksOf, splitWords } from "../../src/core/proposals/diff";

/** The hunks of two texts as "old → new" pairs, joining across kept runs of `join` characters or fewer. */
function changes(before: string, after: string, join = 0): string[] {
  const a = splitWords(before);
  const b = splitWords(after);
  const hunks = hunksOf(
    diffSteps(a, b),
    (from, to) => a.slice(from, to).join("").replace(/\s+/g, "").length <= join,
  );
  return hunks.map(
    (hunk) =>
      `${a.slice(hunk.oldFrom, hunk.oldTo).join("")} → ${b.slice(hunk.newFrom, hunk.newTo).join("")}`,
  );
}

describe("Comparing a text with its new version", () => {
  test("cuts words, spaces and signs apart, and Chinese and Japanese into characters", () => {
    expect(splitWords("Spring tides, don’t wait.")).toEqual([
      "Spring",
      " ",
      "tides",
      ",",
      " ",
      "don’t",
      " ",
      "wait",
      ".",
    ]);
    expect(splitWords("潮水涨落 twice")).toEqual(["潮", "水", "涨", "落", " ", "twice"]);
  });

  test("finds the shortest change, a removed word taking the space after it", () => {
    expect(
      changes("The tides are very strong and really regular.", "The tides are strong and regular."),
    ).toEqual(["very  → ", "really  → "]);
    expect(changes("潮水每天非常规律", "潮水每天很规律")).toEqual(["非常 → 很"]);
    expect(changes("same text", "same text")).toEqual([]);
  });

  test("joins changes a few kept characters apart into one", () => {
    // Only a space apart: one change.
    expect(changes("the cat sat on the mat", "a dog sat on a rug")).toEqual([
      "the cat → a dog",
      "the mat → a rug",
    ]);
    // "and" apart: one change when three letters join; two when none do.
    expect(changes("moon and at full", "new and full", 3)).toEqual(["moon and at  → new and "]);
    expect(changes("moon and at full", "new and full", 0)).toEqual(["moon → new", "at  → "]);
  });

  test("past its limit, replaces the whole text rather than search on", () => {
    const a = splitWords("one two three four five");
    const b = splitWords("uno dos tres cuatro cinco");
    expect(diffSteps(a, b, 2)).toEqual([...a.map(() => "delete"), ...b.map(() => "insert")]);
  });
});

describe("Reading the model's new text", () => {
  test("drops thinking, a code fence, the passage's tags and quotation marks it added", () => {
    expect(readEditReply("<think>Shorter.</think>\nTides are strong.", "x")).toBe(
      "Tides are strong.",
    );
    expect(readEditReply("```text\nTides are strong.\n```", "x")).toBe("Tides are strong.");
    expect(readEditReply("<passage>\nTides.\n</passage>", "x")).toBe("Tides.");
    expect(readEditReply("“Tides are strong.”", "Tides are very strong.")).toBe(
      "Tides are strong.",
    );
    // Quotation marks the passage had stay.
    expect(readEditReply('"Tides," she said.', '"Tides," she said, "rise."')).toBe(
      '"Tides," she said.',
    );
  });

  test("a text given back with only its spaces changed is the same", () => {
    expect(sameText("Spring  tides\n", "Spring tides")).toBe(true);
    expect(sameText("Spring tides", "Spring tide")).toBe(false);
  });
});
