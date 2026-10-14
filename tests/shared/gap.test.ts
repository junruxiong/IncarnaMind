import { describe, expect, test } from "vitest";
import { gap, translate } from "../../src/shared/i18n";

describe("The gap between two pieces of text on a screen", () => {
  test("is a space in English", () => {
    expect(gap("en", "Audio stays here.", "Learn more")).toBe(" ");
  });

  test("is none in Chinese after or before full-width punctuation, or between Chinese characters", () => {
    expect(gap("zh-CN", "会议中其他人的声音。", "音频只保存在这台 Mac 上")).toBe("");
    expect(gap("zh-CN", "数据使用：", "除非你选择加入")).toBe("");
    expect(gap("zh-CN", "无需密钥。", "了解更多")).toBe("");
    expect(gap("zh-CN", "更多信息见", "了解更多")).toBe("");
    expect(gap("zh-CN", "开始使用", "")).toBe("");
  });

  test("is a space in Chinese between Chinese and Latin letters, numbers or symbols", () => {
    expect(gap("zh-CN", "格式来自项目", "citationstyles.org")).toBe(" ");
    expect(gap("zh-CN", "开始使用", "· 0/5")).toBe(" ");
    expect(gap("zh-CN", "Mac", "上的音频")).toBe(" ");
  });

  test("joins the recording notice as the interface language says", () => {
    const join = (language: "en" | "zh-CN") => {
      const before = translate(language, "meetings.notice.both");
      const after = translate(language, "meetings.notice.kept.mac");
      return `${before}${gap(language, before, after)}${after}`;
    };
    expect(join("zh-CN")).toContain("声音。音频");
    expect(join("en")).toContain("meeting. The audio");
  });
});
