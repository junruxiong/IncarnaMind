import { describe, expect, test } from "vitest";
import {
  characters,
  editDistance,
  errorRate,
  normalise,
  words,
} from "../../eval/transcription/scoring";

describe("Scoring transcription: the normalisation rules", () => {
  test("case, punctuation and symbols go, in both scripts; full-width letters and digits become ASCII", () => {
    expect(normalise("When you call someone, you're using a SATELLITE!")).toBe(
      "when you call someone youre using a satellite",
    );
    expect(normalise("报告警告称，没有人能保证……“目前”")).toBe("报告警告称 没有人能保证 目前");
    expect(normalise("ＷｉＦｉ　门铃，２１比２０")).toBe("wifi 门铃 21比20");
  });

  test("English is cut into words, Chinese into Han characters and runs of letters or digits", () => {
    expect(words("The U.N. also hopes")).toEqual(["the", "u", "n", "also", "hopes"]);
    expect(characters("他称，他制作了一个 WiFi 门铃。")).toEqual([
      "他",
      "称",
      "他",
      "制",
      "作",
      "了",
      "一",
      "个",
      "wifi",
      "门",
      "铃",
    ]);
    // Numbers are left as written: #85 fixes a rule for them.
    expect(characters("可达6英里")).toEqual(["可", "达", "6", "英", "里"]);
    expect(characters("可达六英里")).toEqual(["可", "达", "六", "英", "里"]);
  });

  test("error rates count substitutions, deletions and insertions over the reference", () => {
    expect(editDistance(["a", "b", "c"], ["a", "x", "c", "d"])).toBe(2);
    expect(errorRate(words("the cat sat"), words("the cat sat"))).toEqual({
      errors: 0,
      total: 3,
      rate: 0,
    });
    expect(errorRate(characters("可达6英里"), characters("可达六英里")).rate).toBeCloseTo(0.2);
    expect(errorRate([], ["a"]).rate).toBe(0);
  });
});
