import { getSchema, type JSONContent } from "@tiptap/core";
import { parseHTML } from "linkedom";
import { describe, expect, test } from "vitest";
import { CITATION_NODE, FOOTNOTE_NODE } from "../../src/core/api";
import {
  addFootnotes,
  citationsIn,
  textWithSources,
  wordFootnoteHtml,
} from "../../src/renderer/src/editor/copyNotes";
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

describe("copying Citations as Word footnotes", () => {
  const copied = () => {
    const { document } = parseHTML("<!doctype html><html><body></body></html>");
    const root = document.createElement("div");
    root.innerHTML =
      '<p>Spring tides<span data-type="citation"></span></p><p>Neap tides<span data-type="citation"></span></p>';
    const citations = citationsIn(
      fragmentOf(paragraph(text("a"), cite(2)), paragraph(text("b"), cite(5))),
    );
    addFootnotes(root, citations, context);
    return wordFootnoteHtml(root.innerHTML);
  };

  test("each marker is Word's own footnote number, with the visible [n] in a block Word skips", () => {
    const html = copied();
    for (const number of [1, 2]) {
      expect(html).toContain(
        `<span style="mso-special-character:footnote"><![if !supportFootnotes]><span class="MsoFootnoteReference">[${number}]</span><![endif]></span>`,
      );
    }
    // Both ends of each footnote: the reference in the text and the mark opening its note.
    expect(html.match(/mso-special-character:footnote/g)).toHaveLength(4);
    expect(html.match(/<!\[if !supportFootnotes\]>/g)).toHaveLength(4);
    expect(html.match(/<!\[endif\]>/g)).toHaveLength(4);
    expect(html).toContain("mso-footnote-id:ftn1");
    expect(html).toContain("mso-footnote-id:ftn2");
  });

  test("the footnote list and its Sources stay, and no fixed mark is left as a comment", () => {
    const html = copied();
    expect(html).toContain("mso-element:footnote-list");
    expect(html).toContain("Tides, p. 2");
    expect(html).toContain("Tides, p. 5");
    expect(html).not.toContain("<!--");
  });
});
