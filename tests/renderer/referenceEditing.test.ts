/**
 * The "Cite as" panel's words (#286): labels that follow the record's type,
 * what the User typed read back as values, and what is still missing, in
 * English and in Chinese.
 */
import { describe, expect, test } from "vitest";
import type { ReferenceFields } from "../../src/core/api";
import {
  fieldInSentence,
  fieldLabel,
  languageText,
  missingList,
  readTyped,
  valueText,
} from "../../src/renderer/src/referenceEditing";
import { emptyReferenceFields } from "../../src/shared/references";

const fields = (changes: Partial<ReferenceFields>) => ({ ...emptyReferenceFields(), ...changes });

describe("The Cite as panel", () => {
  test("where a work was published is named for its type", () => {
    expect(fieldLabel("en", "containerTitle", "article-journal")).toBe("Journal");
    expect(fieldLabel("en", "containerTitle", "webpage")).toBe("Website");
    expect(fieldLabel("en", "containerTitle", "book")).toBe("Published in");
    expect(fieldLabel("en", "publisher", "thesis")).toBe("University");
    expect(fieldLabel("zh-CN", "containerTitle", "article-journal")).toBe("期刊");
    expect(fieldLabel("zh-CN", "publisher", "thesis")).toBe("学位授予单位");
  });

  test("names show as they are typed back, with the language's separator", () => {
    const names = fields({ authors: [{ family: "Bowler", given: "Diana" }, { literal: "张三" }] });
    expect(valueText("en", "authors", names)).toBe("Bowler, Diana; 张三");
    expect(valueText("zh-CN", "authors", names)).toBe("Bowler, Diana；张三");
  });

  test("what is typed is read back, or says why it can't be", () => {
    expect(readTyped("authors", "张三；Pullin, Andrew")).toEqual({
      ok: true,
      value: [{ literal: "张三" }, { family: "Pullin", given: "Andrew" }],
    });
    expect(readTyped("issued", "2010年8月")).toEqual({ ok: true, value: "2010-08" });
    expect(readTyped("issued", "soon")).toEqual({
      ok: false,
      problem: "reference.issued.invalid",
    });
    expect(readTyped("doi", "https://doi.org/10.1234/x")).toEqual({ ok: true, value: "10.1234/x" });
    expect(readTyped("doi", "x")).toEqual({ ok: false, problem: "reference.doi.invalid" });
    expect(readTyped("title", "  ")).toEqual({ ok: true, value: "" });
  });

  test("what is missing is said as one list, in each language", () => {
    const thin = { fields: fields({ type: "article-journal" }) };
    expect(missingList("en", thin)).toBe("title, authors, and date");
    expect(missingList("zh-CN", thin)).toBe("标题、作者和日期");
    expect(missingList("en", { fields: fields({ type: "document" }) })).toBeNull();
  });

  test("a language shows its name, with the tag the record keeps", () => {
    expect(languageText("en", "en")).toBe("English (en)");
    expect(languageText("zh-CN", "zh")).toBe("中文（zh）");
    expect(languageText("en", "")).toBe("");
    expect(languageText("en", "not a tag!")).toBe("not a tag!");
  });

  test("a field inside a sentence: the title, but the DOI", () => {
    expect(fieldInSentence("en", "title", "book")).toBe("title");
    expect(fieldInSentence("en", "doi", "book")).toBe("DOI");
    expect(fieldInSentence("zh-CN", "title", "book")).toBe("标题");
  });
});
