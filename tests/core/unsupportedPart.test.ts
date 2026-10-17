import { describe, expect, test } from "vitest";
import { partOfSentence, readVerdict } from "../../src/core/answers/checks/prompt";

const SENTENCE =
  "GraphST's variational adaptive graph augmentation has complexity O(|V|² × d), creating a potential scaling constraint.";

describe("The part of a sentence a reply names as unsupported", () => {
  test("is the line after the reason, whatever the label's markup", () => {
    expect(
      readVerdict(
        "Verdict: unsupported\nThe Passage gives the complexity, not what it causes.\nPart: creating a potential scaling constraint",
      ),
    ).toEqual({
      verdict: "unsupported",
      says: "The Passage gives the complexity, not what it causes.",
      part: "creating a potential scaling constraint",
    });
    expect(readVerdict("**Verdict:** unsupported\n**Part:** “a scaling constraint”")?.part).toBe(
      "“a scaling constraint”",
    );
    expect(readVerdict("Verdict: unsupported\n部分：一个扩展瓶颈")?.part).toBe("一个扩展瓶颈");
  });

  test("never takes the place of the reason", () => {
    expect(readVerdict("Verdict: unsupported\nPart: a scaling constraint")).toEqual({
      verdict: "unsupported",
      says: null,
      part: "a scaling constraint",
    });
    expect(
      readVerdict("Verdict: unsupported\nPart of it is missing.\nPart: a scaling constraint"),
    ).toMatchObject({ says: "Part of it is missing.", part: "a scaling constraint" });
  });

  test("is none when the reply names none", () => {
    expect(readVerdict("Verdict: unsupported")).toEqual({
      verdict: "unsupported",
      says: null,
      part: null,
    });
    expect(readVerdict("Verdict: unsupported\nNo Passage mentions penguins.")).toEqual({
      verdict: "unsupported",
      says: "No Passage mentions penguins.",
      part: null,
    });
    expect(readVerdict("Verdict: unsupported\nPart:")?.part).toBeNull();
  });

  test("comes with a JSON reply's verdict", () => {
    expect(readVerdict('{"verdict": "unsupported", "says": "Not stated.", "part": "d)"}')).toEqual({
      verdict: "unsupported",
      says: "Not stated.",
      part: "d)",
    });
  });

  test("a supported reply's explanation is still its line", () => {
    expect(readVerdict("Verdict: supported\nThe Document says so [^1].")).toEqual({
      verdict: "supported",
      says: "The Document says so.",
      part: null,
    });
  });
});

describe("The part, as the sentence has it", () => {
  test("is the words copied from the sentence, without quotes or a closing stop", () => {
    expect(partOfSentence(SENTENCE, "creating a potential scaling constraint")).toBe(
      "creating a potential scaling constraint",
    );
    expect(partOfSentence(SENTENCE, "“creating a potential scaling constraint.”")).toBe(
      "creating a potential scaling constraint",
    );
    expect(partOfSentence(SENTENCE, ", creating a potential scaling constraint")).toBe(
      "creating a potential scaling constraint",
    );
  });

  test("keeps the sentence's own case and spacing", () => {
    expect(partOfSentence(SENTENCE, "Creating  a potential\nscaling constraint")).toBe(
      "creating a potential scaling constraint",
    );
    expect(partOfSentence("Sales rose 5%\tin 2025 (a record).", "rose 5% in 2025")).toBe(
      "rose 5% in 2025",
    );
  });

  test("holds words with regular-expression marks as plain words", () => {
    expect(partOfSentence(SENTENCE, "O(|V|² × d)")).toBe("O(|V|² × d)");
  });

  test("works in Chinese, which has no spaces", () => {
    expect(
      partOfSentence("该方法的复杂度为 O(n²)，因此难以扩展到大规模图。", "因此难以扩展到大规模图"),
    ).toBe("因此难以扩展到大规模图");
  });

  test.each([
    ["words the sentence doesn't hold", "causing a scaling bottleneck"],
    ["words with an ellipsis in them", "creating … constraint"],
    ["nothing", ""],
    ["only quotes", "“”"],
  ])("is left out for %s", (_name, named) => {
    expect(partOfSentence(SENTENCE, named)).toBeNull();
  });

  test("is left out when it is the whole sentence, which is no part", () => {
    expect(partOfSentence(SENTENCE, SENTENCE)).toBeNull();
    expect(partOfSentence(SENTENCE, `“${SENTENCE.slice(0, -1)}”`)).toBeNull();
    expect(partOfSentence(SENTENCE, null)).toBeNull();
  });
});
