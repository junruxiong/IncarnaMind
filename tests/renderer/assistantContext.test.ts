import { describe, expect, test } from "vitest";
import { askedFromLine, stepsScope } from "../../src/renderer/src/assistantContext";
import type { ScopeChip } from "../../src/renderer/src/scope";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

/*
 * Where the next Question is asked from and what it searches, which an
 * empty conversation says (#253, #258, #391), and what a Question searched, in its Answer's
 * steps (#391); in English and Chinese.
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

describe("where the next Question is asked from, and what it searches", () => {
  test("with no Mind in view, it searches All Documents", () => {
    expect(askedFromLine(en, null, [chip("Thesis")])).toBe("Searches All Documents");
    expect(askedFromLine(zh, null, [])).toBe("搜索全部文档");
  });

  test("a Mind and its Folder", () => {
    expect(askedFromLine(en, "Chapter 3", [chip("Thesis")])).toBe(
      "Asked from “Chapter 3” · searches “Thesis”",
    );
    expect(askedFromLine(zh, "第三章", [chip("论文")])).toBe("从“第三章”提问 · 搜索“论文”");
  });

  test("a Mind with no scope searches All Documents; deleted items are left out", () => {
    expect(askedFromLine(en, "Notes", [])).toBe("Asked from “Notes” · searches All Documents");
    expect(askedFromLine(en, "Notes", [chip(null, true), chip("Old", true)])).toBe(
      "Asked from “Notes” · searches All Documents",
    );
  });

  test("a scope of several items names the first and counts the rest", () => {
    const scope = [chip("Thesis"), chip("Leeds data"), chip(null, true), chip("Report.pdf")];
    expect(askedFromLine(en, "Notes", scope)).toBe(
      "Asked from “Notes” · searches “Thesis” and 2 more",
    );
    expect(askedFromLine(zh, "笔记", scope)).toBe("从“笔记”提问 · 搜索“Thesis”等 3 项");
  });
});

describe("what an Answer's steps say was searched", () => {
  const document = (name: string): ScopeChip => ({
    kind: "document",
    id: name,
    name,
    deleted: false,
    documentKind: "pdf",
  });

  // protects=the steps name the scope plainly in English and quoted in Chinese, count Documents alone as files, and add further scopes; fails_when=a scope reads otherwise in either language; no_claim=none beyond protects.
  test("a Folder by its name, Documents alone as files, All Documents, and further scopes", () => {
    expect(stepsScope(en, [chip("Thesis")])).toBe("Thesis");
    expect(stepsScope(zh, [chip("论文")])).toBe("“论文”");
    expect(stepsScope(en, [])).toBe("All Documents");
    expect(stepsScope(zh, [])).toBe("全部文档");
    expect(stepsScope(en, [document("Report")])).toBe("Report");
    expect(stepsScope(en, [document("A"), document("B"), document("C")])).toBe("3 files");
    expect(stepsScope(zh, [document("A"), document("B"), document("C")])).toBe("所选的 3 个文件");
    expect(stepsScope(en, [chip("Thesis"), document("Report")], 1)).toBe("Thesis +2");
    expect(stepsScope(zh, [chip("Thesis")], 1)).toBe("“Thesis”等 2 项");
    expect(stepsScope(en, [chip(null, true)])).toBe("Deleted Folder");
  });
});
