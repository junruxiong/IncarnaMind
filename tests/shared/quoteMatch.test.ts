import { describe, expect, test } from "vitest";
import { findQuote, findQuoteInPieces } from "../../src/shared/quoteMatch";

/** The original text a match covers. */
const covered = (text: string, quote: string) => {
  const range = findQuote(text, quote);
  return range && text.slice(range.start, range.end);
};

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

  // The cases the retrieval prototype (#21) found in real PDFs.
  test("matches CJK radical look-alikes with the ideographs they stand for", () => {
    // A browser-made PDF encodes 大 and 言 as Kangxi radicals, and 长 as a CJK Radicals Supplement character.
    const text = "⼤型语⾔模型的上下⽂⻓度";
    expect(covered(text, "大型语言模型的上下文长度")).toBe(text);
    expect(covered("大型语言模型", "⼤型语⾔模型")).toBe("大型语言模型");
  });

  test("ignores the spaces pdf.js puts around Latin words and numbers in Chinese", () => {
    const text = "例如 GPT-3 含 1750 亿参数，是一个 大型语言模型。";
    expect(covered(text, "GPT-3含1750亿参数")).toBe("GPT-3 含 1750 亿参数");
    expect(covered(text, "例如GPT-3含1750亿参数,是一个大型语言模型")).toBe(
      "例如 GPT-3 含 1750 亿参数，是一个 大型语言模型",
    );
    expect(covered("统 计 语 言 模 型", "统计语言模型")).toBe("统 计 语 言 模 型");
  });

  test("matches a hyphenated compound broken at a line end, with or without its hyphen", () => {
    const text = "On the WMT 2014 English-\nto-German translation task";
    expect(covered(text, "English-to-German translation")).toBe("English-\nto-German translation");
    expect(covered(text, "Englishto-German")).toBe("English-\nto-German");
  });

  test("matches a word split at a line end, with or without the hyphen", () => {
    const text = "growth in inter-\nnational trade";
    expect(covered(text, "international trade")).toBe("inter-\nnational trade");
    expect(covered(text, "inter-national")).toBe("inter-\nnational");
    expect(findQuote(text, "inter national")).toBeNull();
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

  test("matches across a line-end hyphen between pieces", () => {
    const pieces = [
      { text: "the English-", breakAfter: true },
      { text: "to-German task", breakAfter: true },
    ];
    expect(findQuoteInPieces(pieces, "English-to-German")).toEqual([
      { piece: 0, start: 4, end: 12 },
      { piece: 1, start: 0, end: 9 },
    ]);
  });
});
