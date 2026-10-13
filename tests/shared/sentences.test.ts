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
