import { getSchema, type JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import { CITATION_NODE, FOOTNOTE_NODE } from "../../src/core/api";
import { citationsIn, textWithSources } from "../../src/renderer/src/editor/copyNotes";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";

const schema = getSchema(noteExtensions());
const context = { language: "en" as const, documents: [{ id: "tides", contentHash: "h1" }] };

const cite = (pageFrom: number): JSONContent => ({
  type: CITATION_NODE,
  attrs: {
    documentId: "tides",
    documentName: "Tides",
    contentHash: "h1",
    pageFrom,
    pageTo: pageFrom,
  },
});

const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });
const text = (value: string): JSONContent => ({ type: "text", text: value });

const fragmentOf = (...blocks: JSONContent[]) =>
  schema.nodeFromJSON({ type: "doc", content: blocks }).content;

describe("copying Citations as text", () => {
  test("numbered markers in the text and a list of Sources after it", () => {
    const copied = textWithSources(
      fragmentOf(paragraph(text("Spring tides"), cite(2)), paragraph(text("Neap tides"), cite(5))),
      schema,
      context,
    );
    expect(copied).toBe(
      "Spring tides[1]\n\nNeap tides[2]\n\nSources\n[1] Tides, p. 2\n[2] Tides, p. 5",
    );
  });

  test("in Chinese", () => {
    expect(
      textWithSources(fragmentOf(paragraph(text("大潮"), cite(2))), schema, {
        ...context,
        language: "zh-CN",
      }),
    ).toBe("大潮[1]\n\n来源\n[1] Tides，第 2 页");
  });

  test("a copy with no Citation is left to the editor's own text", () => {
    expect(textWithSources(fragmentOf(paragraph(text("Plain"))), schema, context)).toBe("");
  });

  test("a Citation in a footnote the User wrote takes no number", () => {
    const footnote: JSONContent = { type: FOOTNOTE_NODE, content: [text("See "), cite(9)] };
    const fragment = fragmentOf(paragraph(text("Claim"), footnote, cite(2)));
    expect(citationsIn(fragment)).toHaveLength(1);
    expect(textWithSources(fragment, schema, context)).toContain("Claim^[See [Tides, p. 9]][1]");
  });
});
