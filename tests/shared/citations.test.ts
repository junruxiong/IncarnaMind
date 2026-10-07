import { describe, expect, test } from "vitest";
import type { CitationAttributes } from "../../src/core/api";
import {
  badgeMessage,
  citationReference,
  citationState,
  citedPages,
} from "../../src/shared/citations";
import { translate } from "../../src/shared/i18n";
import { findQuoteInPages } from "../../src/shared/quoteMatch";

const found: CitationAttributes = {
  passageId: "passage",
  documentId: "tides",
  documentName: "Tides",
  contentHash: "abc",
  pageFrom: 12,
  pageTo: 13,
  quote: "Spring tides happen at new moon.",
  check: "found",
  checkReason: null,
};

const words = (
  attributes: Partial<CitationAttributes>,
  documents = [{ id: "tides", contentHash: "abc" }],
) => {
  const message = badgeMessage(citationState(attributes, documents), attributes);
  return {
    en: translate("en", message.key, message.params),
    zh: translate("zh-CN", message.key, message.params),
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
    expect(citedPages(found)).toBe("12–13");
    expect(citationReference(found)).toBe("[Tides, p. 12–13]");
    expect(citationReference({ ...found, pageFrom: null, pageTo: null })).toBe("[Tides]");
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
