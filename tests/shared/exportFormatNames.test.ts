import { describe, expect, test } from "vitest";
import { EXPORT_FORMATS } from "../../src/core/api";
import { translate } from "../../src/shared/i18n";

describe("an export format's name in a sentence", () => {
  const exportAs = (language: "en" | "zh-CN") =>
    EXPORT_FORMATS.map((format) =>
      translate(language, "export.as", {
        format: translate(language, `export.format.${format}`),
      }),
    );

  test("in Chinese, has a space beside a Latin name and none beside a Chinese one", () => {
    expect(exportAs("zh-CN")).toEqual([
      "导出为 Word（.docx）",
      "导出为 PDF",
      "导出为 PDF 引用页面合集",
      "导出为 HTML",
      "导出为可分享的网页（.html）",
      "导出为 Markdown（.md）",
    ]);
    expect(
      translate("zh-CN", "export.saving.line", {
        format: translate("zh-CN", "export.format.share-page"),
      }),
    ).toBe("正在导出为可分享的网页（.html）…");
  });

  test("in English, reads as it always did", () => {
    expect(exportAs("en")).toContain("Export as Web page to share (.html)");
    expect(exportAs("en")).toContain("Export as PDF");
  });
});
