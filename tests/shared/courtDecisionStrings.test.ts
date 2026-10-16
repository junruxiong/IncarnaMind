import { describe, expect, test } from "vitest";
import { translate } from "../../src/shared/i18n";

describe("The court decision card's words", () => {
  test("when no Document holds the decision", () => {
    expect(translate("en", "claim.courtDecision.title")).toBe("This cites a court decision.");
    expect(translate("en", "claim.courtDecision.body")).toBe(
      "IncarnaMind can't check that the case exists or that it says this. Look it up in your legal research service before you rely on it.",
    );
    expect(translate("zh-CN", "claim.courtDecision.title")).toBe("本句引用了法院判决。");
    expect(translate("zh-CN", "claim.courtDecision.body")).toBe(
      "IncarnaMind 无法核对这个判决是否存在、是否这样说。依赖它之前，请在你使用的法律数据库中查一下。",
    );
  });

  test("when the opinion is among the Documents, naming the Document and where in it", () => {
    const given = { document: "Hallmark Builders v. Rio Grande Supply", location: "p. 3" };
    expect(translate("en", "claim.courtDecision.held", given)).toBe(
      "This cites a court decision. The quoted words were found in your saved copy of the opinion (Hallmark Builders v. Rio Grande Supply, p. 3). Whether it has been overruled or changed isn't checked: look it up in your legal research service before you rely on it.",
    );
    expect(translate("zh-CN", "claim.courtDecision.held", { ...given, location: "第 3 页" })).toBe(
      "本句引用了法院判决。引用的文字已在你保存的判决书副本中找到（Hallmark Builders v. Rio Grande Supply，第 3 页）。之后有没有法院推翻或修改这个判决，尚未核对；依赖它之前，请在你使用的法律数据库中查一下。",
    );
  });

  test.each(["en", "zh-CN"] as const)(
    "never calls a decision verified, valid or good law (%s)",
    (language) => {
      const words = [
        translate(language, "claim.courtDecision.title"),
        translate(language, "claim.courtDecision.body"),
        translate(language, "claim.courtDecision.held", { document: "X", location: "p. 1" }),
        translate(language, "claim.citedCaseUnchecked"),
        translate(language, "export.comment.sentence.citedCaseUnchecked"),
      ].join(" ");
      expect(words).not.toMatch(/verified|\bvalid|good law|已核实|有效判例/i);
    },
  );

  test("when a Supported sentence cites a case in no Document: the card line and the export's short form", () => {
    expect(translate("en", "claim.citedCaseUnchecked")).toBe(
      "The case this sentence cites isn't in your Documents, so it hasn't been checked. Look it up in your legal research service before you rely on it.",
    );
    expect(translate("zh-CN", "claim.citedCaseUnchecked")).toBe(
      "本句引用的判决不在你的文档中，因此尚未核对。依赖它之前，请在你使用的法律数据库中查一下。",
    );
    expect(translate("en", "export.comment.sentence.citedCaseUnchecked")).toBe(
      "Cited case not checked: not in the Documents.",
    );
    expect(translate("zh-CN", "export.comment.sentence.citedCaseUnchecked")).toBe(
      "所引判决尚未核对：不在文档中。",
    );
  });

  test("a saved web page's source line", () => {
    const given = { address: "https://example.com/opinion", date: "12 Oct" };
    expect(translate("en", "webPages.source", given)).toBe(
      "Saved web page · https://example.com/opinion · saved 12 Oct",
    );
    expect(translate("zh-CN", "webPages.source", { ...given, date: "10月12日" })).toBe(
      "保存的网页 · https://example.com/opinion · 保存于 10月12日",
    );
  });
});
