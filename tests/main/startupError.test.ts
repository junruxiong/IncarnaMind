import { describe, expect, it } from "vitest";
import { DataFromNewerVersionError } from "../../src/core/errors";
import { startupDialog } from "../../src/main/startupError";

describe("startupDialog", () => {
  const newer = new DataFromNewerVersionError(58);

  it("tells English users to update, with no schema number", () => {
    const dialog = startupDialog(newer, "en");
    expect(dialog.title).toBe("Update IncarnaMind to open your data");
    expect(dialog.detail).toBe(
      "A newer version of IncarnaMind saved this data folder, so this version can't open it. Your data is safe and unchanged.",
    );
    expect(dialog.buttons).toEqual(["Download the latest version", "Quit"]);
    expect(`${dialog.message}${dialog.detail}`).not.toMatch(/58|migration/);
  });

  it("tells Chinese users to update, with no schema number", () => {
    const dialog = startupDialog(newer, "zh-CN");
    expect(dialog.title).toBe("更新 IncarnaMind 才能打开你的数据");
    expect(dialog.detail).toBe(
      "较新版本的 IncarnaMind 保存了这个数据文件夹，当前版本无法打开。你的数据完好无损，没有任何改动。",
    );
    expect(dialog.buttons).toEqual(["下载最新版本", "退出"]);
    expect(`${dialog.message}${dialog.detail}`).not.toMatch(/58|migration/);
  });

  it("keeps the message of any other start-up failure", () => {
    const dialog = startupDialog(new Error("disk full"), "en");
    expect(dialog.title).toBe("IncarnaMind couldn't start");
    expect(dialog.detail).toContain("disk full");
    expect(dialog.buttons).toEqual([]);
  });
});
