import { describe, expect, test } from "vitest";
import { readsLine } from "../../src/renderer/src/assistantContext";
import type { ScopeChip } from "../../src/renderer/src/scope";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

/*
 * The line under the assistant's header (#253, #258): what the next Question
 * reads and searches, in English and Chinese.
 */

const en = (key: MessageKey, params?: MessageParams) => translate("en", key, params);
const zh = (key: MessageKey, params?: MessageParams) => translate("zh-CN", key, params);

const chip = (name: string | null, deleted = false): ScopeChip => ({
  kind: "folder",
  id: name ?? "gone",
  name,
  deleted,
  documentKind: null,
});

describe("what the assistant reads", () => {
  test("with no Mind in view, it searches All Documents", () => {
    expect(readsLine(en, null, [chip("Thesis")])).toBe("Searches All Documents");
    expect(readsLine(zh, null, [])).toBe("搜索全部文档");
  });

  test("a Mind and its Folder", () => {
    expect(readsLine(en, "Chapter 3", [chip("Thesis")])).toBe(
      "Reads “Chapter 3” · searches “Thesis”",
    );
    expect(readsLine(zh, "第三章", [chip("论文")])).toBe("正在读“第三章” · 搜索“论文”");
  });

  test("a Mind with no scope searches All Documents; deleted items are left out", () => {
    expect(readsLine(en, "Notes", [])).toBe("Reads “Notes” · searches All Documents");
    expect(readsLine(en, "Notes", [chip(null, true), chip("Old", true)])).toBe(
      "Reads “Notes” · searches All Documents",
    );
  });

  test("a scope of several items names the first and counts the rest", () => {
    const scope = [chip("Thesis"), chip("Leeds data"), chip(null, true), chip("Report.pdf")];
    expect(readsLine(en, "Notes", scope)).toBe("Reads “Notes” · searches “Thesis” and 2 more");
    expect(readsLine(zh, "笔记", scope)).toBe("正在读“笔记” · 搜索“Thesis”等 3 项");
  });
});
