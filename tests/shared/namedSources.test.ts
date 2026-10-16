import { describe, expect, test } from "vitest";
import { citesSource, linkSources } from "../../src/shared/namedSources";

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
