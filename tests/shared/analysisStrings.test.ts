import { describe, expect, test } from "vitest";
import { type AnalysisUnavailableReason, UNAVAILABLE_REASONS } from "../../src/shared/analysis";
import { translate } from "../../src/shared/i18n";

/*
 * Where the User would use an Analysis that can't run on their computer, they read why, in plain
 * words, in English and in Chinese (ADR-0023).
 */

const REASONS = Object.keys(UNAVAILABLE_REASONS) as AnalysisUnavailableReason[];

describe("why an Analysis can't run", () => {
  test.each(REASONS)("%s reads as one sentence in English and in Chinese", (reason) => {
    const english = translate("en", "analysis.unavailable", {
      reason: translate("en", UNAVAILABLE_REASONS[reason]),
    });
    const chinese = translate("zh-CN", "analysis.unavailable", {
      reason: translate("zh-CN", UNAVAILABLE_REASONS[reason]),
    });

    expect(english).toMatch(/^Analysis can't run on this computer: [a-z].+\.$/);
    expect(chinese).toMatch(/^这台电脑上无法运行数据分析：\p{Script=Han}.+。$/u);
    expect(chinese).not.toContain("沙箱");
  });

  test("a sandbox that can't start says so", () => {
    expect(
      translate("en", "analysis.unavailable", {
        reason: translate("en", UNAVAILABLE_REASONS.start),
      }),
    ).toBe("Analysis can't run on this computer: its sandbox couldn't start.");
  });
});
