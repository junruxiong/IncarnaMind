import { describe, expect, test } from "vitest";
import { fileDate, infoLabelKey, showFileLabelKey } from "../../src/renderer/src/fileInfo";
import { translate } from "../../src/shared/i18n";

describe("a file's own date", () => {
  test("is shown at the precision the file gave", () => {
    expect(fileDate("2019", "en")).toBe("2019");
    expect(fileDate("2019-03", "en")).toBe("March 2019");
    expect(fileDate("2019-03-04", "en")).toBe("March 4, 2019");
    expect(fileDate("2019-03-04", "zh-CN")).toBe("2019年3月4日");
  });

  test("is the day the file names, whatever its offset from UTC", () => {
    expect(fileDate("2019-03-04T23:30:00-08:00", "en")).toBe("March 4, 2019");
    expect(fileDate("2019-03-04T00:30:00+09:00", "en")).toBe("March 4, 2019");
  });

  test("is left as it is when it isn't a date", () => {
    expect(fileDate("sometime", "en")).toBe("sometime");
  });
});

test("every action that shows a file says Finder on macOS and folder on Windows and Linux, in both languages", () => {
  const says = (platform: string, language: "en" | "zh-CN") =>
    translate(language, showFileLabelKey(platform));
  expect(says("MacIntel", "en")).toBe("Show in Finder");
  expect(says("MacIntel", "zh-CN")).toBe("在访达中显示");
  for (const platform of ["Win32", "Linux x86_64"]) {
    expect(says(platform, "en")).toBe("Show in folder");
    expect(says(platform, "zh-CN")).toBe("在文件夹中显示");
  }
});

test("Get info is Properties in Chinese on Windows and Linux only", () => {
  expect(infoLabelKey("MacIntel")).toBe("info.get");
  expect(infoLabelKey("Win32")).toBe("info.get.other");
  expect(infoLabelKey("Linux x86_64")).toBe("info.get.other");
});
