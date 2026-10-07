import { describe, expect, test } from "vitest";
import { normaliseText, normaliseWithOffsets } from "../../src/shared/text";

describe("normalising text", () => {
  test("collapses whitespace, line breaks included, and trims it", () => {
    expect(normaliseText("  Revenue grew\n  by ten\tpercent.  ")).toBe(
      "Revenue grew by ten percent.",
    );
    expect(normaliseText("a\u00a0b\r\nc")).toBe("a b c");
  });

  test("applies NFKC and unifies quote marks, dashes and CJK full stops and commas", () => {
    expect(normaliseText("“Ｆｕｌｌ” ‘width’ — ﬁne – yes")).toBe("\"Full\" 'width' - fine - yes");
    expect(normaliseText("「引号」、逗号，句号。")).toBe('"引号",逗号,句号.');
    expect(normaliseText("ｅ\u0301 = é")).toBe("é = é");
  });

  test("folds the CJK radical look-alikes that browser-made PDFs use", () => {
    // Kangxi radicals (NFKC folds them) and CJK Radicals Supplement characters (it doesn't).
    expect(normaliseText("⼤型语⾔模型")).toBe("大型语言模型");
    expect(normaliseText("⻓城和⻔⼝")).toBe("长城和门口");
  });

  test("drops invisible characters", () => {
    expect(normaliseText("soft\u00adhyphen zero\u200bwidth\ufeff")).toBe("softhyphen zerowidth");
  });

  test("removes whitespace next to CJK characters and CJK punctuation, but not Hangul", () => {
    expect(normaliseText("如 GPT-3 含 1750 亿参数")).toBe("如GPT-3含1750亿参数");
    expect(normaliseText("统 计 语 言 模 型")).toBe("统计语言模型");
    expect(normaliseText("卷积神经\n网络擅长处理图像。\n循环 神经网络，\n擅长")).toBe(
      "卷积神经网络擅长处理图像.循环神经网络,擅长",
    );
    expect(normaliseText("使用 GPU 训练")).toBe("使用GPU训练");
    expect(normaliseText("日本語の テキスト")).toBe("日本語のテキスト");
    expect(normaliseText("안녕 하세요")).toBe("안녕 하세요");
  });

  describe("line-break hyphenation", () => {
    test("joins a word split by a hyphen at the end of a line", () => {
      expect(normaliseText("inter-\nnational")).toBe("international");
      expect(normaliseText("The trans-\n  former model")).toBe("The transformer model");
      expect(normaliseText("inter- \r\n national")).toBe("international");
      expect(normaliseText("INTER-\nNATIONAL")).toBe("INTERNATIONAL");
    });

    test("keeps the hyphen of a compound broken after it", () => {
      expect(normaliseText("the WMT 2014 English-\nto-German translation task")).toBe(
        "the WMT 2014 English-to-German translation task",
      );
      expect(normaliseText("state-of-\nthe-art")).toBe("state-of-the-art");
      expect(normaliseText("a non-\nEnglish speaker")).toBe("a non-English speaker");
    });

    test("keeps a hyphen or dash that ends a line before a number, without a space", () => {
      expect(normaliseText("COVID-\n19")).toBe("COVID-19");
      expect(normaliseText("from 1990–\n1995")).toBe("from 1990-1995");
      expect(normaliseText("one thing—\nanother")).toBe("one thing-another");
    });

    test("leaves hyphens within a line and free-standing dashes alone", () => {
      expect(normaliseText("well-known self-attention")).toBe("well-known self-attention");
      expect(normaliseText("the result -\nwhich")).toBe("the result - which");
      expect(normaliseText("- first\n- second")).toBe("- first - second");
    });

    test("marks each line-end hyphen as optional, and the joined ones as removed", () => {
      const joined = normaliseWithOffsets("inter-\nnational");
      expect(joined.text).toBe("international");
      expect(joined.units.filter((unit) => unit.optional)).toEqual([
        { char: "-", start: 5, end: 6, optional: true, removed: true },
      ]);

      const kept = normaliseWithOffsets("English-\nto-German");
      expect(kept.text).toBe("English-to-German");
      expect(kept.units.filter((unit) => unit.optional)).toEqual([
        { char: "-", start: 7, end: 8, optional: true },
      ]);
    });
  });

  test("maps every normalised character back to the span of the original it came from", () => {
    const original = "The ﬁrst  “⼤”\nword";
    const { text, units } = normaliseWithOffsets(original);

    expect(text).toBe('The first "大" word');
    expect(units.map((unit) => original.slice(unit.start, unit.end))).toEqual([
      "T",
      "h",
      "e",
      " ",
      "ﬁ",
      "ﬁ",
      "r",
      "s",
      "t",
      "  ",
      "“",
      "⼤",
      "”",
      "\n",
      "w",
      "o",
      "r",
      "d",
    ]);
  });
});
