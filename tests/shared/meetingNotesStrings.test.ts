import { describe, expect, test } from "vitest";
import { type MessageKey, translate } from "../../src/shared/i18n";

const KEYS: readonly MessageKey[] = [
  "meetings.command.mark",
  "meetings.command.markPaused",
  "meetings.mark",
  "meetings.mark.title",
  "meetings.mark.noNote",
  "meetings.command.addLine",
  "meetings.command.noLine",
  "transcript.addToNote",
  "transcript.addToNote.line",
  "transcript.addToNote.added",
  "transcript.addToNote.already",
  "transcript.addToNote.undo",
  "transcript.addToNote.failed",
  "transcript.line.hint",
];

describe("the strings of notes during a meeting", () => {
  test.each(KEYS)("%s is in English and in Chinese, and the Chinese is not the English", (key) => {
    const en = translate("en", key, { shortcut: "⌘D", time: "0:01", text: "x" });
    const zh = translate("zh-CN", key, { shortcut: "⌘D", time: "0:01", text: "x" });
    expect(en).not.toBe("");
    expect(zh).not.toBe("");
    expect(zh).not.toBe(en);
    expect(zh).toMatch(/[一-鿿]/);
  });

  test("the undo line says a line of the transcript was added, in Chinese", () => {
    expect(translate("zh-CN", "transcript.addToNote.undo")).toBe("已将转写稿中的一句加入笔记");
  });
});
