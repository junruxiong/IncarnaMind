import { getSchema, type JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import { CITATION_NODE } from "../../src/core/api";
import { findMatches } from "../../src/renderer/src/editor/findInMind";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";

const schema = getSchema(noteExtensions());

const text = (value: string, marks?: string[]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks && { marks: marks.map((type) => ({ type })) }),
});

const docOf = (...paragraphs: JSONContent[][]) =>
  schema.nodeFromJSON({
    type: "doc",
    content: paragraphs.map((content) => ({ type: "paragraph", content })),
  });

/** What each match covers, as the document's text between its parts. */
const found = (doc: ReturnType<typeof docOf>, query: string) =>
  findMatches(doc, query).map((match) =>
    match.parts.map((part) => doc.textBetween(part.from, part.to)).join("|"),
  );

describe("findMatches", () => {
  test("finds every occurrence in order, whatever the case", () => {
    const doc = docOf([text("Tides and tides and TIDES")], [text("No tide here")]);
    expect(found(doc, "tides")).toEqual(["Tides", "tides", "TIDES"]);
    expect(found(doc, "TIDE")).toEqual(["Tide", "tide", "TIDE", "tide"]);
  });

  test("finds Chinese text as it comes, with no idea of words", () => {
    const doc = docOf([text("潮汐是海水的周期性涨落，大潮和小潮交替出现。")]);
    expect(found(doc, "潮")).toHaveLength(3);
    expect(found(doc, "大潮和")).toEqual(["大潮和"]);
    // Chinese and Latin letters side by side, with no space between them.
    expect(found(docOf([text("用Mind记录潮汐")]), "mind记录")).toEqual(["Mind记录"]);
  });

  test("a match across marks covers each stretch, and a node in between breaks it", () => {
    const marked = docOf([text("spring "), text("tides", ["bold"]), text(" rise")]);
    const [match] = findMatches(marked, "ng tides r");
    expect(match?.parts).toHaveLength(3);
    expect(found(marked, "ng tides r")).toEqual(["ng |tides| r"]);

    const cited = docOf([
      text("tide"),
      { type: CITATION_NODE, attrs: { documentId: "d", documentName: "D" } },
      text("s"),
    ]);
    expect(found(cited, "tides")).toEqual([]);
    expect(found(cited, "tide")).toEqual(["tide"]);
  });

  test("an empty query finds nothing", () => {
    expect(findMatches(docOf([text("tides")]), "")).toEqual([]);
  });

  test("matches don't overlap into the next paragraph", () => {
    const doc = docOf([text("tid")], [text("es")]);
    expect(found(doc, "tides")).toEqual([]);
  });
});
