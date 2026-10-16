import { describe, expect, it } from "vitest";
import { withCitationGlue, withoutCitationGlue } from "../../src/renderer/src/editor/citationGlue";
import type { NodeJSON } from "../../src/shared/blockJson";

const citation: NodeJSON = { type: "citation", attrs: { passageId: "p1" } };
const text = (value: string): NodeJSON => ({ type: "text", text: value });
const paragraph = (...content: NodeJSON[]): NodeJSON => ({ type: "paragraph", content });
const glue = { type: "citationGlue" };

describe("withCitationGlue", () => {
  it("marks the last word, the Citations after it and the punctuation after them", () => {
    const [block] = withCitationGlue([
      paragraph(text("Your Documents answer this "), citation, citation, text(". Then more")),
    ]);
    expect(block?.content).toEqual([
      text("Your Documents answer "),
      { ...text("this "), marks: [glue] },
      { ...citation, marks: [glue] },
      { ...citation, marks: [glue] },
      { ...text("."), marks: [glue] },
      text(" Then more"),
    ]);
  });

  it("keeps only the first Citations with the word, and the last with the full stop, so a long run can wrap", () => {
    const [block] = withCitationGlue([
      paragraph(text("this "), ...Array.from({ length: 6 }, () => citation), text(". Then")),
    ]);
    const marked = block?.content?.map((node) => node.marks?.length === 1);
    expect(marked).toEqual([true, true, true, true, false, false, true, true, false]);
  });

  it("marks the word before the next sentence's Citation, after the full stop of the last", () => {
    const [block] = withCitationGlue([
      paragraph(text("One "), citation, text(". Two words "), citation, text(".")),
    ]);
    expect(block?.content).toEqual([
      { ...text("One "), marks: [glue] },
      { ...citation, marks: [glue] },
      { ...text("."), marks: [glue] },
      text(" Two "),
      { ...text("words "), marks: [glue] },
      { ...citation, marks: [glue] },
      { ...text("."), marks: [glue] },
    ]);
  });

  it("takes one character of Chinese", () => {
    const [block] = withCitationGlue([paragraph(text("增长了"), citation, text("。"))]);
    expect(block?.content).toEqual([
      text("增长"),
      { ...text("了"), marks: [glue] },
      { ...citation, marks: [glue] },
      { ...text("。"), marks: [glue] },
    ]);
  });

  it("leaves text without a Citation, and code, as it is", () => {
    const plain = paragraph(text("No Citation here."));
    const code: NodeJSON = { type: "codeBlock", content: [text("a[1]")] };
    expect(withCitationGlue([plain, code])).toEqual([plain, code]);
  });

  it("is undone by withoutCitationGlue, except that text stays split", () => {
    const glued = withCitationGlue([paragraph(text("see this "), citation, text("."))]);
    const clean = withoutCitationGlue(glued);
    expect(JSON.stringify(clean)).not.toContain("citationGlue");
    expect(clean[0]?.content?.map((node) => node.text ?? node.type).join("|")).toBe(
      "see |this |citation|.",
    );
  });
});
