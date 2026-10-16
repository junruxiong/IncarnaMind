import { describe, expect, test } from "vitest";
import { cleanSpeech, textOfResult, wordsOnly } from "../../src/core/transcription/speechText";

/** A result as sherpa-onnx writes it: its text goes in as it is, apart from quotes and backslashes. */
const resultOf = (text: string, tokens: string[] = []) =>
  `{"lang": "", "emotion": "", "event": "", "text": "${text}", "timestamps": [], "tokens": ${JSON.stringify(
    tokens,
  ).replace(/\\u([0-9a-f]{4})/g, (_, code: string) =>
    String.fromCharCode(Number.parseInt(code, 16)),
  )}, "words": []}`;

describe("A result's text, as sherpa-onnx writes it (#981)", () => {
  test("is read when it holds a newline, a tab or another control character", () => {
    expect(textOfResult(resultOf("two\nlines"))).toBe("two\nlines");
    expect(textOfResult(resultOf("a\ttab"))).toBe("a\ttab");
    expect(textOfResult(resultOf("\u0001\u0007bell and DEL\u007f"))).toBe(
      "\u0001\u0007bell and DEL\u007f",
    );
    expect(() => JSON.parse(resultOf("two\nlines"))).toThrow(/control character/);
  });

  test("is read when it is in Chinese, or has quotes and backslashes, or its tokens hold control characters", () => {
    expect(textOfResult(resultOf("发布日期\n推迟了。"))).toBe("发布日期\n推迟了。");
    expect(textOfResult(resultOf('He said \\"hi\\" and a\\\\b\n'))).toBe('He said "hi" and a\\b\n');
    expect(textOfResult(resultOf("fine", ["fi", "ne\n", "\t"]))).toBe("fine");
  });

  test("is as it was for a result that reads as it is, and empty when it has none", () => {
    expect(textOfResult('{"text": "plain"}')).toBe("plain");
    expect(textOfResult('{"lang": ""}')).toBe("");
    expect(textOfResult('{"text": 3}')).toBe("");
    expect(() => textOfResult("not json")).toThrow(SyntaxError);
  });
});

describe("Speech kept as one line of words (#981)", () => {
  test("a break is a space, another control character is left out, and the line is trimmed", () => {
    expect(cleanSpeech("The launch\nmoves\tto the\u0007 eighteenth.\r\n\r\nDone")).toBe(
      "The launch moves to the eighteenth. Done",
    );
    expect(cleanSpeech("\n\n  发布日期\u0000推迟。\n")).toBe("发布日期推迟。");
    expect(cleanSpeech("\u0001\u0002\n")).toBe("");
  });

  test("text with no control characters is left as it is", () => {
    expect(cleanSpeech("  Plain words, 你好。 ")).toBe("Plain words, 你好。");
  });
});

describe("The words after the model's header (#981)", () => {
  test("the header the model wrote before them is left out, wherever it begins", () => {
    expect(wordsOnly("提uggestion\nlanguage English<asr_text>Local authorities are warning.")).toBe(
      "Local authorities are warning.",
    );
    expect(wordsOnly("language English<asr_text>")).toBe("");
  });

  test("words with no header are as they are", () => {
    expect(wordsOnly("Local authorities are warning.")).toBe("Local authorities are warning.");
    expect(wordsOnly("")).toBe("");
  });
});
