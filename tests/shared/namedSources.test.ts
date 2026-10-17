import { describe, expect, test } from "vitest";
import { citesSource, linkSources, sourcesAsText } from "../../src/shared/namedSources";

/**
 * Sources a model writes in a table's cells as a name in brackets (#961):
 * each links to the Citation of that Document at the place it names, by page,
 * slide, sheet cell or section, or stays as text and is counted.
 */

const CITATIONS = [
  { documentName: "Annual report.pdf", pageFrom: 3, pageTo: 4 },
  { documentName: "Deck.pptx", location: { kind: "slide", from: 2, to: 2 } },
  { documentName: "Model.xlsx", location: { kind: "rows", sheet: "Revenue", from: 10, to: 14 } },
  { documentName: "Policy.docx", location: { kind: "section", heading: "4. Termination" } },
  { documentName: "Notes.md" },
];

const link = (text: string) =>
  linkSources(text, (source) => {
    const index = CITATIONS.findIndex((citation) => citesSource(citation, source));
    return index < 0 ? null : `[${index + 1}]`;
  });

describe("A source written as a name in brackets", () => {
  // protects=page, slide, sheet cell and section sources link to the Citation there, in English and Chinese; fails_when=one of them stays as text though a Citation is there; no_claim=none beyond protects.
  test.each([
    ["Revenue rose [Annual report, p. 4]", "Revenue rose [1]"],
    ["Revenue rose [Annual report，第 3 页]", "Revenue rose [1]"],
    ["Pricing [Deck, slide 2]", "Pricing [2]"],
    ["Q3 [Model.xlsx, B12]", "Q3 [3]"],
    ["Q3 [Model, Revenue!B12]", "Q3 [3]"],
    ["Q3 [Model.xlsx, Revenue, rows 11–12]", "Q3 [3]"],
    ["Notice [Policy, §4]", "Notice [4]"],
    ["Notice [Policy, § 4]", "Notice [4]"],
    ["Ideas [Notes]", "Ideas [5]"],
    ["Ideas [Notes, p. 9]", "Ideas [5]"],
    ["Notice [Policy, p. 2]", "Notice [4]"],
  ])("%s", (text, linked) => {
    expect(link(text)).toEqual({ text: linked, unlinked: 0 });
  });

  // protects=a source at a place no Citation is, or of a Document none is of, stays as text and is counted, while a bracket that names no source isn't counted; fails_when=it links to the wrong place, or "[sic]" counts as a source; no_claim=none beyond protects.
  test.each([
    ["[Annual report, p. 9]", 1],
    ["[Deck, slide 5]", 1],
    ["[Model.xlsx, B40]", 1],
    ["[Model.xlsx, Costs!B12]", 1],
    ["[Policy, §7]", 1],
    ["[Atlas of the Sea, p. 2]", 1],
    ["[sic]", 0],
    ["[^1] and [2]", 0],
    ["[a link](https://example.com)", 0],
  ])("%s stays as text", (text, unlinked) => {
    expect(link(text)).toEqual({ text, unlinked });
  });
});

describe("Sources written as text in what the assistant writes into a Mind (#1075)", () => {
  const known = (name: string) => (source: { name: string }) => source.name === name;

  // protects=the owner's two shapes, a place alone, several sources in one bracket, full-width brackets and a bare Document name are found as written, while markers, links, "[sic]" and a cell reference alone aren't; fails_when=one of them is missed, or text that names no source is taken for one; no_claim=none beyond protects.
  test("finds each bracket that holds a source, as written", () => {
    const text = [
      "Offices of 1,200 m² [5_Rue_Berryer_Paris, slide 3].",
      "收入每年增长百分之二十[执行概要, § 6.3.1 未来三年财务预测]。",
      "Scope [§ 1.1 Scope] and [p. 4], from [Deck, slide 2; Policy, § 4] and 【执行概要，第 3 页】.",
      "Ideas [Notes], a marker [^1], a figure [=2], a link [site](https://example.com), [sic] and [B12].",
    ].join("\n\n");
    expect(sourcesAsText(text, known("Notes"))).toEqual([
      "[5_Rue_Berryer_Paris, slide 3]",
      "[执行概要, § 6.3.1 未来三年财务预测]",
      "[§ 1.1 Scope]",
      "[p. 4]",
      "[Deck, slide 2; Policy, § 4]",
      "【执行概要，第 3 页】",
      "[Notes]",
    ]);
  });

  // protects=with unmatched sources left out, each source becomes its Citation's marker or goes with the space before it, and one bracket of two sources keeps the one it can; fails_when=a bracket stays as text, or a space is left before the punctuation; no_claim=none beyond protects.
  test("drops what matches no Citation, with the space before it", () => {
    const linked = linkSources(
      "Pricing [Deck, slide 2]. Notice [Policy, § 9]. Both [Deck, slide 2; Atlas, p. 2].",
      (source) => (citesSource(CITATIONS[1] ?? {}, source) ? "[^2]" : null),
      "drop",
    );
    expect(linked).toEqual({ text: "Pricing [^2]. Notice. Both [^2].", unlinked: 2 });
  });
});
