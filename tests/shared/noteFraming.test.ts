import { describe, expect, test } from "vitest";
import type { NodeJSON } from "../../src/shared/blockJson";
import { citedTables, withoutFraming } from "../../src/shared/noteBlocks";

const p = (text: string): NodeJSON => ({ type: "paragraph", content: [{ type: "text", text }] });
const citation: NodeJSON = { type: "citation", attrs: { documentId: "d" } };
const cell = (type: string, text: string): NodeJSON => ({ type, content: [p(text)] });
const table: NodeJSON = {
  type: "table",
  content: [
    { type: "tableRow", content: [cell("tableHeader", "Year"), cell("tableHeader", "Revenue")] },
    { type: "tableRow", content: [cell("tableCell", "FY2020"), cell("tableCell", "1,000")] },
  ],
};

describe("withoutFraming", () => {
  test("leaves out a lead-in and a closing note, in English and Chinese", () => {
    expect(withoutFraming([p("Here is a draft:"), table, p("Note: only a draft.")])).toEqual([
      table,
    ]);
    expect(withoutFraming([p("以下是草稿："), table, p("注：以上仅为草稿。")])).toEqual([table]);
  });
  test("keeps a lone paragraph, a Citation-bearing one and ordinary prose", () => {
    expect(withoutFraming([p("Here is a draft")])).toEqual([p("Here is a draft")]);
    const cited = { type: "paragraph", content: [{ type: "text", text: "Note: " }, citation] };
    expect(withoutFraming([table, cited])).toEqual([table, cited]);
    expect(withoutFraming([p("Tides vary."), p("Springs are largest.")]).length).toBe(2);
  });
});

describe("citedTables", () => {
  test("puts the Citation under a table onto its figure cells and drops the paragraph", () => {
    const [done, ...rest] = citedTables([table, { type: "paragraph", content: [citation] }]);
    expect(rest).toEqual([]);
    const row = done?.content?.[1]?.content;
    expect(row?.[0]?.content?.[0]?.content?.some((n) => n.type === "citation")).toBe(false);
    expect(row?.[1]?.content?.[0]?.content?.at(-1)).toEqual(citation);
  });
  test("leaves a table that is cited already, and other paragraphs, as they are", () => {
    const cited = {
      ...table,
      content: [
        {
          type: "tableRow",
          content: [{ type: "tableCell", content: [{ type: "paragraph", content: [citation] }] }],
        },
      ],
    };
    const under = { type: "paragraph", content: [citation] };
    expect(citedTables([cited, under])).toEqual([cited, under]);
    expect(citedTables([p("a"), under])).toEqual([p("a"), under]);
  });
});
