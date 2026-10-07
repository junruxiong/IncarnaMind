import { describe, expect, test } from "vitest";
import type { CitationAttributes } from "../../src/core/api";
import type { Language } from "../../src/core/language";
import {
  badgeMessage,
  citationReference,
  citationState,
  citedLocation,
  citedTextKept,
} from "../../src/shared/citations";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";
import { findQuoteInPages } from "../../src/shared/quoteMatch";

/** A Citation from before Locations: its pages say where it points. */
const found: CitationAttributes = {
  passageId: "passage",
  documentId: "tides",
  documentName: "Tides",
  contentHash: "abc",
  pageFrom: 12,
  pageTo: 13,
  location: null,
  quote: "Spring tides happen at new moon.",
  check: "found",
  checkReason: null,
};

/** Words in a language, as the renderer's `t` does. */
const in_ = (language: Language) => (key: MessageKey, params?: MessageParams) =>
  translate(language, key, params);

const words = (
  attributes: Partial<CitationAttributes>,
  documents = [{ id: "tides", contentHash: "abc" }],
) => {
  const state = citationState(attributes, documents);
  const en = badgeMessage(state, attributes, in_("en"));
  const zh = badgeMessage(state, attributes, in_("zh-CN"));
  return {
    en: translate("en", en.key, en.params),
    zh: translate("zh-CN", zh.key, zh.params),
  };
};

describe("A Citation's badge", () => {
  test("says where the quote was found or not found, in words, and never that the sentence is verified", () => {
    expect(words(found)).toEqual({ en: "Quote found on p. 12–13", zh: "已在第 12–13 页找到引文" });
    expect(
      words({ ...found, pageTo: 12, check: "not-found", checkReason: "quote-not-on-pages" }),
    ).toEqual({
      en: "Quote not found on p. 12",
      zh: "未在第 12 页找到引文",
    });
    // Documents without pages (TXT, Markdown) are named instead.
    expect(words({ ...found, pageFrom: null, pageTo: null })).toEqual({
      en: "Quote found in Tides",
      zh: "已在《Tides》中找到引文",
    });
    expect(words({ ...found, check: "checking" }).en).toBe("checking…");
    expect(words({ ...found, check: "cant-check", checkReason: "no-text" }).en).toBe("Can't check");
    for (const check of ["found", "not-found", "cant-check", "checking"] as const) {
      expect(words({ ...found, check }).en).not.toMatch(/verif/i);
    }
  });

  test("turns to 'can't check' once the Document is deleted, unless the same file was added again", () => {
    expect(citationState(found, [])).toEqual({
      check: "cant-check",
      reason: "document-removed",
      documentId: null,
      changedAfterCited: false,
    });
    // Added again, the same content is a new Document: the Citation opens it.
    expect(citationState(found, [{ id: "tides-again", contentHash: "abc" }])).toEqual({
      check: "found",
      reason: null,
      documentId: "tides-again",
      changedAfterCited: false,
    });
    // While the Documents are loading, the stored result stands.
    expect(citationState(found, null)).toMatchObject({ check: "found", documentId: "tides" });
  });

  test("keeps its check once its Document is unlinked with its folder, while every Unit it cites is kept", () => {
    const text = { documentId: "tides", contentHash: "abc", units: [12, 13] };
    const kept = [text];
    expect(citationState(found, [], kept)).toEqual({
      check: "found",
      reason: null,
      documentId: null,
      changedAfterCited: false,
    });
    const notFound = {
      ...found,
      check: "not-found" as const,
      checkReason: "quote-not-on-pages" as const,
    };
    expect(citationState(notFound, [], kept)).toMatchObject({
      check: "not-found",
      reason: "quote-not-on-pages",
    });
    // Linked again, the same content is a live Document again: the Citation opens it.
    expect(citationState(found, [{ id: "tides-again", contentHash: "abc" }], kept)).toMatchObject({
      check: "found",
      documentId: "tides-again",
    });
    // Not when one of its Units, its version or its Document isn't the one kept.
    const removed = { check: "cant-check", reason: "document-removed" };
    expect(citationState(found, [], [{ ...text, units: [12] }])).toMatchObject(removed);
    expect(citationState(found, [], [{ ...text, contentHash: "def" }])).toMatchObject(removed);
    expect(citationState(found, [], [{ ...text, documentId: "rivers" }])).toMatchObject(removed);
    // A Citation that doesn't say which version it quotes takes any; one of a whole file, its text.
    expect(citationState({ ...found, contentHash: null }, [], kept).check).toBe("found");
    const whole = { ...found, pageFrom: null, pageTo: null };
    expect(citationState(whole, [], [{ ...text, units: [1] }]).check).toBe("found");
  });

  test("the viewer tells an unlinked Document whose quoted text was kept from a deleted one", () => {
    const kept = [{ documentId: "tides", contentHash: "abc", units: [12, 13] }];
    // As the viewer opens a Citation: its Document, version and Units.
    const opened = { documentId: "tides", contentHash: "abc", pageFrom: 12, pageTo: 13 };
    expect(citedTextKept(opened, kept)).toBe(true);
    expect(citedTextKept({ ...opened, contentHash: null }, kept)).toBe(true);
    // Deleted by the User, nothing is kept; nor of another version, or Unit.
    expect(citedTextKept(opened, [])).toBe(false);
    expect(citedTextKept({ ...opened, contentHash: "def" }, kept)).toBe(false);
    expect(citedTextKept({ ...opened, pageFrom: 14, pageTo: 14 }, kept)).toBe(false);
  });

  test("says when the Document changed after it was cited: the check stands for the version quoted", () => {
    expect(citationState(found, [{ id: "tides", contentHash: "abc" }])).toMatchObject({
      check: "found",
      changedAfterCited: false,
    });
    expect(citationState(found, [{ id: "tides", contentHash: "def" }])).toEqual({
      check: "found",
      reason: null,
      documentId: "tides",
      changedAfterCited: true,
    });
    // Not while it is being written, nor for a Citation that doesn't say which version it quotes.
    expect(
      citationState({ ...found, check: "checking" }, [{ id: "tides", contentHash: "def" }]),
    ).toMatchObject({ changedAfterCited: false });
    expect(
      citationState({ ...found, contentHash: null }, [{ id: "tides", contentHash: "def" }]),
    ).toMatchObject({ changedAfterCited: false });
  });

  test("names its source in plain text, for copying and for Question context", () => {
    expect(citedLocation(found, in_("en"))).toBe("p. 12–13");
    expect(citationReference(found)).toBe("[Tides, p. 12–13]");
    expect(citationReference({ ...found, pageFrom: null, pageTo: null })).toBe("[Tides]");
  });
});

describe("A Citation's Location label (ADR-0011)", () => {
  const at = (location: CitationAttributes["location"], name = "Deck") => ({
    ...found,
    documentName: name,
    location,
  });

  test("shows a slide, a range of rows, a section and lines, in English and Chinese", () => {
    const cases: [CitationAttributes["location"], string, string][] = [
      [{ kind: "page", from: 4, to: 4 }, "p. 4", "第 4 页"],
      [{ kind: "slide", from: 4, to: 4 }, "slide 4", "第 4 张幻灯片"],
      [{ kind: "slide", from: 3, to: 4 }, "slides 3–4", "第 3–4 张幻灯片"],
      [
        { kind: "rows", sheet: "Revenue", from: 12, to: 14 },
        "Revenue, rows 12–14",
        "Revenue，第 12–14 行",
      ],
      [{ kind: "rows", sheet: "Revenue", from: 7, to: 7 }, "Revenue, row 7", "Revenue，第 7 行"],
      [{ kind: "rows", sheet: null, from: 2, to: 3 }, "rows 2–3", "第 2–3 行"],
      [{ kind: "section", heading: "2.1 Sensitivity" }, "§ 2.1 Sensitivity", "§ 2.1 Sensitivity"],
      [{ kind: "section", heading: null }, "§ Start", "§ 开头"],
      [{ kind: "section", heading: null, notes: true }, "§ Notes", "§ 注释"],
      [{ kind: "lines", from: 120, to: 134 }, "lines 120–134", "第 120–134 行"],
    ];
    for (const [location, en, zh] of cases) {
      expect(citedLocation(at(location), in_("en"))).toBe(en);
      expect(citedLocation(at(location), in_("zh-CN"))).toBe(zh);
    }
  });

  test("words the badge with the label: on a page, slide or lines; in rows or a section", () => {
    expect(words(at({ kind: "slide", from: 4, to: 4 }))).toEqual({
      en: "Quote found on slide 4",
      zh: "已在第 4 张幻灯片找到引文",
    });
    expect(words(at({ kind: "rows", sheet: "Revenue", from: 12, to: 14 })).en).toBe(
      "Quote found in Revenue, rows 12–14",
    );
    expect(
      words({
        ...at({ kind: "section", heading: "2.1 Sensitivity" }),
        check: "not-found",
        checkReason: "quote-not-on-pages",
      }),
    ).toEqual({
      en: "Quote not found in § 2.1 Sensitivity",
      zh: "未在 § 2.1 Sensitivity 中找到引文",
    });
    expect(words(at({ kind: "lines", from: 120, to: 134 })).en).toBe(
      "Quote found on lines 120–134",
    );
  });

  test("copies as text with its label, and a stored Location that isn't one falls back to the pages", () => {
    expect(citationReference(at({ kind: "slide", from: 4, to: 4 }))).toBe("[Deck, slide 4]");
    expect(
      citationReference(at({ kind: "rows", sheet: "Revenue", from: 12, to: 14 }, "Model")),
    ).toBe("[Model, Revenue, rows 12–14]");
    const broken = {
      ...found,
      location: { kind: "slide", from: "x" },
    } as unknown as CitationAttributes;
    expect(citationReference(broken)).toBe("[Tides, p. 12–13]");
  });
});

describe("Finding a quote across pages in the viewer", () => {
  const line = (text: string) => ({ text, breakAfter: true });

  test("a quote across a page break is found though a footer, page number and header come between", () => {
    const pages = [
      [
        line("Annual Report"),
        line("Neap tides occur when the Sun"),
        line("and the Moon"),
        line("2"),
      ],
      [line("Annual Report"), line("pull at right angles."), line("3")],
    ];

    const found = findQuoteInPages(
      pages,
      "Neap tides occur when the Sun and the Moon pull at right angles.",
    );

    expect(found).toEqual([
      { page: 0, piece: 1, start: 0, end: 29 },
      { page: 0, piece: 2, start: 0, end: 12 },
      { page: 1, piece: 1, start: 0, end: 21 },
    ]);
  });

  test("a quote that isn't on the pages isn't found", () => {
    const pages = [[line("Spring tides happen at new moon.")], [line("Neap tides are small.")]];

    expect(findQuoteInPages(pages, "Spring tides are small.")).toBeNull();
    expect(findQuoteInPages(pages, "Neap tides are small.")).toEqual([
      { page: 1, piece: 0, start: 0, end: 21 },
    ]);
  });
});
