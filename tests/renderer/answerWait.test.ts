import { afterEach, describe, expect, test } from "vitest";
import { setSilentWaitMs, waitText } from "../../src/renderer/src/answerWaitWords";
import { translate } from "../../src/shared/i18n";

/*
 * What an Answer's line says once it has been silent for 15 seconds (#899):
 * how long its model usually takes, in rough measures, or that it is still working.
 */

type Key = Parameters<typeof translate>[1];
type Params = Parameters<typeof translate>[2];
const en = (key: Key, params?: Params) => translate("en", key, params);
const zh = (key: Key, params?: Params) => translate("zh-CN", key, params);
const SONNET = "Claude Sonnet 5.5";

afterEach(() => setSilentWaitMs(15_000));

describe("The line of an Answer with no words", () => {
  // protects=a usual wait is said in seconds under a minute, as a minute near 60 seconds and in minutes after, in both languages; fails_when=75 seconds reads "75 seconds", or Chinese keeps English word order; no_claim=the rounding is for reading, not exact.
  test.each([
    [
      40,
      "Usually about 40 seconds with Claude Sonnet 5.5",
      "用 Claude Sonnet 5.5 通常要 40 秒左右",
    ],
    [
      43,
      "Usually about 40 seconds with Claude Sonnet 5.5",
      "用 Claude Sonnet 5.5 通常要 40 秒左右",
    ],
    [58, "Usually about a minute with Claude Sonnet 5.5", "用 Claude Sonnet 5.5 通常要 1 分钟左右"],
    [75, "Usually about a minute with Claude Sonnet 5.5", "用 Claude Sonnet 5.5 通常要 1 分钟左右"],
    [
      190,
      "Usually about 3 minutes with Claude Sonnet 5.5",
      "用 Claude Sonnet 5.5 通常要 3 分钟左右",
    ],
  ])("%s seconds", (seconds, english, chinese) => {
    expect(waitText(en, seconds, SONNET)).toBe(english);
    expect(waitText(zh, seconds, SONNET)).toBe(chinese);
  });

  // protects=with no history, a usual wait shorter than the wait so far, or no model, the line only says it is still working; fails_when=it claims "usually about 10 seconds" after 15 seconds of silence; no_claim=none beyond protects.
  test("says it is still working with no usual wait, one shorter than the wait so far, or no model", () => {
    expect(waitText(en, null, SONNET)).toBe("Still working");
    expect(waitText(zh, null, SONNET)).toBe("仍在处理");
    expect(waitText(en, 10, SONNET)).toBe("Still working");
    expect(waitText(en, 15, SONNET)).toBe("Still working");
    expect(waitText(en, 60, null)).toBe("Still working");
  });

  test("counts the wait so far as the test build sets it", () => {
    setSilentWaitMs(1_000);
    expect(waitText(en, 10, SONNET)).toBe("Usually about 10 seconds with Claude Sonnet 5.5");
  });
});
