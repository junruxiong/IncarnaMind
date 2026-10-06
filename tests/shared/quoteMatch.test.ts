import { describe, expect, test } from "vitest";
import { findQuote, findQuoteInPieces, normaliseForMatching } from "../../src/shared/quoteMatch";

/** The original text a match covers. */
const covered = (text: string, quote: string) => {
  const range = findQuote(text, quote);
  return range && text.slice(range.start, range.end);
};

describe("normalising text for matching quotes", () => {
  test("collapses whitespace, line breaks included, and trims it", () => {
    expect(normaliseForMatching("  Revenue grew\n  by ten\tpercent.  ")).toBe(
      "Revenue grew by ten percent.",
    );
  });

  test("applies NFKC and unifies quote marks and dashes", () => {
    expect(normaliseForMatching("“Ｆｕｌｌ” ‘width’ — ﬁne – yes")).toBe(
      "\"Full\" 'width' - fine - yes",
    );
  });

  test("removes end-of-line hyphenation, but keeps hyphens within a line", () => {
    expect(normaliseForMatching("inter-\nnational well-known")).toBe("international well-known");
  });

  test("removes whitespace between CJK characters, CJK punctuation included", () => {
    expect(normaliseForMatching("卷积神经\n网络擅长处理图像。\n循环 神经网络，\n擅长")).toBe(
      "卷积神经网络擅长处理图像。循环神经网络,擅长",
    );
    expect(normaliseForMatching("使用 GPU 训练")).toBe("使用 GPU 训练");
  });

  test("drops invisible characters", () => {
    expect(normaliseForMatching("soft­hyphen zero​width")).toBe("softhyphen zerowidth");
  });
});

describe("finding a quote", () => {
  test("finds it across line breaks and returns the span of the original text", () => {
    const text = "Results\nRevenue grew by ten percent\nin the third quarter, led by exports.";
    expect(covered(text, "Revenue grew by ten percent in the third quarter")).toBe(
      "Revenue grew by ten percent\nin the third quarter",
    );
  });

  test("finds Chinese quotes broken across lines", () => {
    const text = "第一章\n卷积神经网络擅长处理图像。\n循环神经网络擅长处理序列数据。";
    expect(covered(text, "擅长处理图像。循环神经网络")).toBe("擅长处理图像。\n循环神经网络");
  });

  test("maps back through characters NFKC changes", () => {
    const text = "The ﬁrst “quoted” word";
    expect(covered(text, 'first "quoted"')).toBe("ﬁrst “quoted”");
  });

  test("is an exact match: a near miss, or an empty quote, isn't found", () => {
    expect(findQuote("Revenue grew by ten percent.", "Revenue grew by 10 percent")).toBeNull();
    expect(findQuote("Revenue grew.", "revenue grew")).toBeNull();
    expect(findQuote("Revenue grew.", "  \n ")).toBeNull();
  });
});

describe("finding a quote across pieces of text", () => {
  test("returns the part of each piece the quote covers", () => {
    const pieces = [
      { text: "Results", breakAfter: true },
      { text: "Revenue grew by ten percent", breakAfter: true },
      { text: "in the third quarter, led by exports.", breakAfter: true },
      { text: "Next page", breakAfter: false },
    ];
    expect(findQuoteInPieces(pieces, "ten percent in the third quarter")).toEqual([
      { piece: 1, start: 16, end: 27 },
      { piece: 2, start: 0, end: 20 },
    ]);
    expect(findQuoteInPieces(pieces, "Results Revenue")).toEqual([
      { piece: 0, start: 0, end: 7 },
      { piece: 1, start: 0, end: 7 },
    ]);
  });

  test("joins pieces without a break directly, as a PDF's text runs on one line", () => {
    const pieces = [{ text: "Reve" }, { text: "nue grew" }];
    expect(findQuoteInPieces(pieces, "Revenue")).toEqual([
      { piece: 0, start: 0, end: 4 },
      { piece: 1, start: 0, end: 3 },
    ]);
    expect(findQuoteInPieces(pieces, "Profit")).toBeNull();
  });
});
