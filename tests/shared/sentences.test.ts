import { describe, expect, test } from "vitest";
import { INLINE_NODE, isClaim, sentenceSpans } from "../../src/shared/sentences";

const sentences = (text: string) => sentenceSpans(text).map(({ from, to }) => text.slice(from, to));

describe("The sentences of a paragraph", () => {
  test("are split at their full stops, and keep their figures whole", () => {
    expect(
      sentences(
        `Net leverage was 3.1× at the end of the quarter, ${INLINE_NODE} inside the 3.5× limit in clause 7.1. Online sales grew 12% in the quarter.`,
      ),
    ).toEqual([
      `Net leverage was 3.1× at the end of the quarter, ${INLINE_NODE} inside the 3.5× limit in clause 7.1.`,
      "Online sales grew 12% in the quarter.",
    ]);
  });

  test("a Citation right after a full stop belongs to the sentence before it", () => {
    expect(
      sentences(`Spring tides come at full moon.${INLINE_NODE} Neap tides come between.`),
    ).toEqual([`Spring tides come at full moon.${INLINE_NODE}`, "Neap tides come between."]);
  });

  test("a court decision's citation stays in one sentence, with its court and year", () => {
    expect(
      sentences(
        "See Hallmark Builders v. Rio Grande Supply, 512 S.W.3d 101 (Tex. 2017). Penguins nest here. Cf. Acme Corp. v. Widget Co., 25 F. Supp. 2d 345, 350 (S.D.N.Y. 1998). Donoghue v Stevenson [1932] AC 562 is older.",
      ),
    ).toEqual([
      "See Hallmark Builders v. Rio Grande Supply, 512 S.W.3d 101 (Tex. 2017).",
      "Penguins nest here.",
      "Cf. Acme Corp. v. Widget Co., 25 F. Supp. 2d 345, 350 (S.D.N.Y. 1998).",
      "Donoghue v Stevenson [1932] AC 562 is older.",
    ]);
  });

  test.each([
    ["The duty arises under the Act.", "See Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017)."],
    ["The lien attached in Texas.", "See Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017)."],
    ["Revenue fell in Q3.", "See Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017)."],
    ["The supplier was owned by IBM.", "See Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017)."],
    ["Yes.", "See Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017)."],
    [
      "It was formed in Delaware.",
      "Under Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017), it attaches.",
    ],
    [
      "The plan is Plan A.",
      "In Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017), it was upheld.",
    ],
    ["The case arose in the U.S.", "Brown v. Board of Education, 347 U.S. 483 (1954), decided it."],
    ["The buyer was Acme Co.", "Smith v. Jones, 12 F.3d 5 (1st Cir. 1994), is cited."],
  ])("the sentence before a citation stays apart from it: %j", (before, cited) => {
    expect(sentences(`${before} ${cited}`)).toEqual([before, cited]);
  });

  test("Chinese sentences end at 。！？", () => {
    expect(sentences("今天的潮水很高。明天会更高！真的吗？")).toEqual([
      "今天的潮水很高。",
      "明天会更高！",
      "真的吗？",
    ]);
  });

  test("a title doesn't end a sentence, and a line break does", () => {
    expect(sentences("Mr. Smith signed it. Fig. 2 shows why.\nNext line")).toEqual([
      "Mr. Smith signed it.",
      "Fig. 2 shows why.",
      "Next line",
    ]);
  });

  test("nothing but spaces or a lone Citation is no sentence", () => {
    expect(sentences("   ")).toEqual([]);
    expect(sentences(` ${INLINE_NODE} `)).toEqual([]);
  });
});

describe("A sentence that makes a claim", () => {
  test("has three words, or four characters of Chinese", () => {
    expect(isClaim("Online sales grew 12% in the quarter.")).toBe(true);
    expect(isClaim("Next steps")).toBe(false);
    expect(isClaim("Thanks!")).toBe(false);
    expect(isClaim("销售额增长了。")).toBe(true);
    expect(isClaim("谢谢。")).toBe(false);
  });
});
