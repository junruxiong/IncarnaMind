import { describe, expect, test } from "vitest";
import { clockTime, durationLabel } from "../../src/renderer/src/meetingTime";
import { translate } from "../../src/shared/i18n";

describe("A Meeting's time", () => {
  test("reads as a clock: minutes and seconds, then hours", () => {
    expect(clockTime(0)).toBe("0:00");
    expect(clockTime(4_999)).toBe("0:04");
    expect(clockTime(24 * 60_000 + 16_000)).toBe("24:16");
    expect(clockTime(3_725_000)).toBe("1:02:05");
  });

  test("a gap's length reads in words, in English and Chinese", () => {
    const en = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
      translate("en", key, params);
    const zh = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
      translate("zh-CN", key, params);
    expect(durationLabel(45_000, en)).toBe("45 s");
    expect(durationLabel(184_000, en)).toBe("3 min");
    expect(durationLabel(2 * 3_600_000 + 5 * 60_000, en)).toBe("2 h 5 min");
    expect(durationLabel(45_000, zh)).toBe("45 秒");
    expect(durationLabel(184_000, zh)).toBe("3 分钟");
    expect(durationLabel(2 * 3_600_000 + 5 * 60_000, zh)).toBe("2 小时 5 分钟");
  });
});
