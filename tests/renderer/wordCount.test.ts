import { getSchema, type JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import { CITATION_NODE } from "../../src/core/api";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { countMind } from "../../src/renderer/src/editor/wordCountText";
import {
  countBasisOf,
  setCountBasis,
  setWordTarget,
  wordTargetOf,
} from "../../src/shared/mindWordCount";
import { countFor, countText, mostlyCjk, parseTarget } from "../../src/shared/wordCount";

const schema = getSchema(noteExtensions());

const text = (value: string): JSONContent => ({ type: "text", text: value });
const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });
const citation = (): JSONContent => ({
  type: CITATION_NODE,
  attrs: { documentId: "tides", documentName: "Tides", check: "found" },
});
const question = (value: string): JSONContent => ({ type: "question", content: [text(value)] });

const docOf = (...blocks: JSONContent[]) => schema.nodeFromJSON({ type: "doc", content: blocks });

const words = (value: string) => countFor(countText(value), "words");

describe("counting text", () => {
  test("English is counted in words", () => {
    expect(words("The tide turns at noon.")).toBe(5);
    expect(words("")).toBe(0);
    expect(words("  \n ")).toBe(0);
    expect(words("well-known isn't")).toBe(2);
  });

  test("Chinese is counted by character, punctuation not counted", () => {
    expect(words("今天的潮水很高。")).toBe(7);
    expect(countText("今天的潮水很高。").characters).toBe(8);
  });

  test("mixed text counts words and characters together", () => {
    const count = countText("用 Python 写了 3 个脚本");
    expect(count.words).toBe(2);
    expect(count.cjk).toBe(6);
    expect(countFor(count, "words")).toBe(8);
    expect(mostlyCjk(count)).toBe(true);
    expect(mostlyCjk(countText("a few words 和"))).toBe(false);
  });

  test("a number with separators is one word", () => {
    expect(words("Revenue was 1,234.56 in 2026")).toBe(5);
    expect(words("共 1,234 人")).toBe(3);
  });

  test("characters are counted with and without spaces, line breaks never", () => {
    const count = countText("ab cd\nef");
    expect(countFor(count, "characters")).toBe(7);
    expect(countFor(count, "charactersNoSpaces")).toBe(6);
  });

  test("an emoji is one character", () => {
    expect(countText("a😀").characters).toBe(2);
  });
});

describe("counting a Mind", () => {
  test("a Citation marker is not counted", () => {
    const withMarker = docOf(paragraph(text("Tides rise twice a day"), citation(), text(" here.")));
    const without = docOf(paragraph(text("Tides rise twice a day here.")));
    const a = countMind(withMarker, { from: 0, to: 0 }).total;
    const b = countMind(without, { from: 0, to: 0 }).total;
    expect(countFor(a, "words")).toBe(6);
    expect(a.characters).toBe(b.characters - 0);
  });

  test("Blocks are counted, Questions are not", () => {
    const doc = docOf(
      paragraph(text("One two three.")),
      question("What are the tides?"),
      paragraph(text("Four five.")),
    );
    expect(countFor(countMind(doc, { from: 0, to: 0 }).total, "words")).toBe(5);
  });

  test("the selection is counted beside the whole", () => {
    const doc = docOf(paragraph(text("One two three.")), paragraph(text("Four five six seven.")));
    // Inside the first paragraph: "One two" is positions 1 to 8.
    const inFirst = countMind(doc, { from: 1, to: 8 });
    expect(inFirst.selection && countFor(inFirst.selection, "words")).toBe(2);
    expect(countFor(inFirst.total, "words")).toBe(7);

    // Across both paragraphs: from "three." to "Four five".
    const across = countMind(doc, { from: 9, to: 26 });
    expect(across.selection && countFor(across.selection, "words")).toBe(3);
  });

  test("no selection, or one with no text, is not a selection", () => {
    const doc = docOf(paragraph(text("One two.")), paragraph(text("Three.")));
    expect(countMind(doc, { from: 3, to: 3 }).selection).toBeNull();
    // Only the boundary between two paragraphs.
    expect(countMind(doc, { from: 9, to: 10 }).selection).toBeNull();
  });

  test("the selection counts Chinese by character", () => {
    const doc = docOf(paragraph(text("潮水每天涨落两次。")));
    const { selection } = countMind(doc, { from: 1, to: 5 });
    expect(selection && countFor(selection, "words")).toBe(4);
  });

  test("Blocks a seam names are left out", () => {
    const doc = docOf(paragraph(text("Body text.")), paragraph(text("Footnote text here.")));
    const total = countMind(doc, { from: 0, to: 0 }, ["footnote"]).total;
    expect(countFor(total, "words")).toBe(5);
    expect(countFor(countMind(doc, { from: 0, to: 0 }, ["paragraph"]).total, "words")).toBe(0);
  });
});

describe("the target", () => {
  test.each([
    ["8000", 8000],
    ["8,000 words", 8000],
    ["3万", 30000],
    ["3万字", 30000],
    ["1.5万", 15000],
    ["５千", 5000],
    ["5k", 5000],
  ])("%s is %i", (input, expected) => {
    expect(parseTarget(input)).toBe(expected);
  });

  test.each(["", "words", "0", "-5", "万"])("%j is no target", (input) => {
    expect(parseTarget(input)).toBeNull();
  });

  test("the Mind remembers its target and basis, and forgets them", () => {
    const doc = new Y.Doc();
    expect(wordTargetOf(doc)).toBeNull();
    expect(countBasisOf(doc)).toBe("words");
    setWordTarget(doc, 8000);
    setCountBasis(doc, "charactersNoSpaces");
    expect(wordTargetOf(doc)).toBe(8000);
    expect(countBasisOf(doc)).toBe("charactersNoSpaces");
    setWordTarget(doc, null);
    setCountBasis(doc, "words");
    expect(wordTargetOf(doc)).toBeNull();
    expect(countBasisOf(doc)).toBe("words");
  });
});
