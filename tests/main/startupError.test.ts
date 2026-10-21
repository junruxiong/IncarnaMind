import { describe, expect, it } from "vitest";
import { DataFromNewerVersionError } from "../../src/core/errors";
import { busyDialog, startupDialog } from "../../src/main/startupError";

describe("startupDialog", () => {
  const newer = new DataFromNewerVersionError(58);

  it("tells English users to update, with no schema number", () => {
    const dialog = startupDialog(newer, "en");
    expect(dialog.title).toBe("Update IncarnaMind to open your data");
    expect(dialog.detail).toBe(
      "A newer IncarnaMind saved this data. Update to open it; nothing changed.",
    );
    expect(dialog.buttons).toEqual(["Download the latest version", "Quit"]);
    expect(`${dialog.message}${dialog.detail}`).not.toMatch(/58|migration/);
  });

  it("tells Chinese users to update, with no schema number", () => {
    const dialog = startupDialog(newer, "zh-CN");
    expect(dialog.title).toBe("更新 IncarnaMind 才能打开你的数据");
    expect(dialog.detail).toBe(
      "较新版本的 IncarnaMind 保存了这些数据。更新后即可打开，数据未改动。",
    );
    expect(dialog.buttons).toEqual(["下载最新版本", "退出"]);
    expect(`${dialog.message}${dialog.detail}`).not.toMatch(/58|migration/);
  });

  it("keeps the message of any other start-up failure", () => {
    const dialog = startupDialog(new Error("disk full"), "en");
    expect(dialog.title).toBe("IncarnaMind couldn’t start");
    expect(dialog.detail).toContain("disk full");
    expect(dialog.buttons).toEqual([]);
  });

  it("says in plain words that another copy may still be starting, in both languages", () => {
    expect(busyDialog("en").detail).toBe(
      "IncarnaMind couldn’t open its data folder. Another copy may still be starting. Try again in a moment.",
    );
    expect(busyDialog("zh-CN").detail).toBe(
      "IncarnaMind 无法打开数据文件夹。可能另一个副本仍在启动，请稍后再试。",
    );
  });
});
