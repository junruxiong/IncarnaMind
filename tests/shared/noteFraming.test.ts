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
  test("leaves out the model explaining how it went about the request (#1061)", () => {
    const explaining = p("你没有说明是哪一份合同，所以我把四份文档各写了一段简短摘要。");
    const summary = p("合同 A 规定租期为两年。");
    expect(withoutFraming([explaining, summary, p("合同 B 规定月租为八千元。")])).toEqual([
      summary,
      p("合同 B 规定月租为八千元。"),
    ]);
    const english = p("You didn't say which contract, so I've summarised all four.");
    expect(withoutFraming([english, summary, p("Tell me if you meant one of them.")])).toEqual([
      summary,
    ]);
  });
  test("keeps a first or last paragraph that is the content's own, in English and Chinese", () => {
    const body = p("The landlord has not replied.");
    for (const first of [
      p("I have written to you twice about the damp in the bedroom and nothing has been done."),
      p("我先后给物业写了三封信，反映卧室漏水的问题。"),
    ]) {
      expect(withoutFraming([first, body])).toEqual([first, body]);
    }
    for (const last of [
      p("If you want to renew, write to the landlord before the term ends."),
      p("如果你想续租，需要在租期结束前书面通知房东。"),
      p("如需开具发票，请扫描小票底部二维码。"),
      p("If you don't pay by Friday, I'll start proceedings."),
      p("如果你不按时付款，我会提起诉讼。"),
      p("I can be reached on 07700 900123 if you need to discuss it."),
    ]) {
      expect(withoutFraming([body, last])).toEqual([body, last]);
    }
  });
  test("leaves out the model's offer or question to the User after the content, whatever it holds", () => {
    const body = p("The landlord has not replied.");
    for (const last of [
      p("Tell me if you meant the 2019 lease."),
      p("Let me know if you want the 2 other leases too."),
      p("Would you like a shorter version?"),
      p("If you'd like, I can add the dates."),
      p("If you want me to cite page 4 as well, say so."),
      p("Want me to draft the reply?"),
      p("需要的话，我可以把摘要写得更详细。"),
      p("如果需要我补充 2019 年的数据，请说一声。"),
      p("要不要我再写一版？"),
      p("告诉我你指的是哪一份合同。"),
      p("如有需要，我可以补充租金的细节。"),
    ]) {
      expect(withoutFraming([body, last])).toEqual([body]);
    }
  });
  test("keeps framing that says something: units, figures, and a sentence that starts 这是", () => {
    const units = p("Revenue and operating profit, FY2020–24 (GBP m):");
    const note = p("Note: figures in GBP millions.");
    expect(withoutFraming([units, table, note])).toEqual([units, table, note]);
    const real = p("这是公司首次亏损");
    expect(withoutFraming([real, table])).toEqual([real, table]);
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
