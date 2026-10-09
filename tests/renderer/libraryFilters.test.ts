import { describe, expect, test } from "vitest";
import type { Document } from "../../src/core/api";
import {
  documentFacets,
  documentYear,
  type FilterSelection,
  filterLibrary,
  formatFacet,
  isFiltering,
  type LibraryFacet,
  libraryStatus,
  NO_DATE,
  statusFacet,
  toggleFilterValue,
  yearFacet,
} from "../../src/renderer/src/libraryFilters";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

const en = (key: MessageKey, params?: MessageParams) => translate("en", key, params);
const zh = (key: MessageKey, params?: MessageParams) => translate("zh-CN", key, params);

type Shape = Pick<Document, "id" | "kind" | "creationDate" | "status" | "fileStatus">;

let next = 0;
/** A ready PDF from 2023 whose file is where it was, unless said otherwise. */
const doc = (changes: Partial<Shape> = {}): Shape => ({
  id: `doc-${++next}`,
  kind: "pdf",
  creationDate: "2023-05-01",
  status: "ready",
  fileStatus: "available",
  ...changes,
});

const options = <T>(
  items: readonly T[],
  facet: LibraryFacet<T>,
  selection: FilterSelection = {},
  facets: readonly LibraryFacet<T>[] = documentFacets as readonly LibraryFacet<T>[],
) =>
  filterLibrary(items, facets, selection)
    .options.get(facet.id)
    ?.map(({ value, count, selected }) => [value, count, selected]);

describe("a Document's year", () => {
  test("is the first four characters of its creation date, at any precision", () => {
    expect(documentYear({ creationDate: "2019-03-04T10:30:00+01:00" })).toBe("2019");
    expect(documentYear({ creationDate: "2019-03-04" })).toBe("2019");
    expect(documentYear({ creationDate: "2019" })).toBe("2019");
  });

  test("without a creation date (or with one that isn't a date) it has none", () => {
    expect(documentYear({ creationDate: null })).toBe(NO_DATE);
    expect(documentYear({ creationDate: "" })).toBe(NO_DATE);
    expect(documentYear({ creationDate: "soon" })).toBe(NO_DATE);
  });
});

describe("a Document's status in the Library", () => {
  test("its file comes first: missing or unavailable, whatever its processing", () => {
    expect(libraryStatus(doc({ fileStatus: "missing", status: "failed" }))).toBe("missing");
    expect(libraryStatus(doc({ fileStatus: "unavailable", status: "embedding" }))).toBe(
      "unavailable",
    );
  });

  test("then its processing: ready is available, any step before it is not indexed yet", () => {
    expect(libraryStatus(doc())).toBe("available");
    for (const status of ["queued", "extracting", "waiting-for-model", "embedding"] as const) {
      expect(libraryStatus(doc({ status }))).toBe("not-indexed");
    }
    expect(libraryStatus(doc({ status: "failed" }))).toBe("failed");
    expect(libraryStatus(doc({ status: "no-text" }))).toBe("no-text");
  });
});

describe("filtering the Library", () => {
  const library = [
    doc({ kind: "pdf", creationDate: "2023-01-02" }),
    doc({ kind: "pdf", creationDate: "2021", status: "failed" }),
    doc({ kind: "docx", creationDate: "2023-11-30T08:00:00Z" }),
    doc({ kind: "docx", creationDate: null, fileStatus: "missing" }),
    doc({ kind: "text", creationDate: null, status: "queued" }),
  ];

  test("with nothing chosen, every Document shows and each option counts its Documents", () => {
    const { shown } = filterLibrary(library, documentFacets, {});
    expect(shown).toEqual(library);
    // Newest year first, Documents without a date last.
    expect(options(library, yearFacet)).toEqual([
      ["2023", 2, false],
      ["2021", 1, false],
      [NO_DATE, 2, false],
    ]);
    expect(options(library, formatFacet)).toEqual([
      ["pdf", 2, false],
      ["docx", 2, false],
      ["text", 1, false],
    ]);
    expect(options(library, statusFacet)).toEqual([
      ["available", 2, false],
      ["not-indexed", 1, false],
      ["failed", 1, false],
      ["missing", 1, false],
    ]);
  });

  test("options of one filter are either-or; different filters must all match", () => {
    const years = { year: ["2023", "2021"] };
    expect(filterLibrary(library, documentFacets, years).shown).toEqual(library.slice(0, 3));
    const pdfs = { ...years, format: ["pdf"] };
    expect(filterLibrary(library, documentFacets, pdfs).shown).toEqual(library.slice(0, 2));
    const failed = { ...pdfs, status: ["failed"] };
    expect(filterLibrary(library, documentFacets, failed).shown).toEqual([library[1]]);
    expect(filterLibrary(library, documentFacets, { ...failed, format: ["docx"] }).shown).toEqual(
      [],
    );
  });

  test("each option counts what choosing it would show, given the other filters", () => {
    const selection = { format: ["docx"] };
    // The years among Word files; the formats among every year (none chosen), the chosen one checked.
    expect(options(library, yearFacet, selection)).toEqual([
      ["2023", 1, false],
      ["2021", 0, false],
      [NO_DATE, 1, false],
    ]);
    expect(options(library, formatFacet, selection)).toEqual([
      ["pdf", 2, false],
      ["docx", 2, true],
      ["text", 1, false],
    ]);
    expect(options(library, statusFacet, selection)).toEqual([
      ["available", 1, false],
      ["not-indexed", 0, false],
      ["failed", 0, false],
      ["missing", 1, false],
    ]);
  });

  test("an option chosen keeps its place even when nothing in view has it", () => {
    const selection = { year: ["1999"] };
    expect(filterLibrary(library, documentFacets, selection).shown).toEqual([]);
    expect(options(library, yearFacet, selection)?.[2]).toEqual(["1999", 0, true]);
  });

  test("another kind of filter, e.g. one with several values per Document, fits in", () => {
    type Tagged = Shape & { tags: string[] };
    const tagFacet: LibraryFacet<Tagged> = {
      id: "tag",
      title: "tags.title",
      values: (item) => item.tags,
      label: (value) => value,
      compare: (a, b) => a.localeCompare(b),
    };
    const tagged: Tagged[] = [
      { ...doc(), tags: ["Report", "Finance"] },
      { ...doc({ kind: "docx" }), tags: ["Report"] },
      { ...doc(), tags: [] },
    ];
    const facets = [...(documentFacets as readonly LibraryFacet<Tagged>[]), tagFacet];
    expect(options(tagged, tagFacet, {}, facets)).toEqual([
      ["Finance", 1, false],
      ["Report", 2, false],
    ]);
    const { shown } = filterLibrary(tagged, facets, { tag: ["Report"], format: ["pdf"] });
    expect(shown).toEqual([tagged[0]]);
  });
});

describe("choosing filters", () => {
  test("toggling adds an option, then takes it away; a filter left empty is dropped", () => {
    let selection: FilterSelection = {};
    expect(isFiltering(selection)).toBe(false);
    selection = toggleFilterValue(selection, "year", "2023");
    selection = toggleFilterValue(selection, "year", "2021");
    selection = toggleFilterValue(selection, "format", "pdf");
    expect(selection).toEqual({ year: ["2023", "2021"], format: ["pdf"] });
    expect(isFiltering(selection)).toBe(true);
    selection = toggleFilterValue(selection, "format", "pdf");
    expect(selection).toEqual({ year: ["2023", "2021"] });
    selection = toggleFilterValue(toggleFilterValue(selection, "year", "2023"), "year", "2021");
    expect(selection).toEqual({});
    expect(isFiltering(selection)).toBe(false);
  });
});

describe("the filters' words", () => {
  test("say each option in English and Chinese", () => {
    expect(yearFacet.label("2023", en)).toBe("2023");
    expect(yearFacet.label(NO_DATE, en)).toBe("No date");
    expect(yearFacet.label(NO_DATE, zh)).toBe("无日期");
    expect(formatFacet.label("docx", en)).toBe("Word");
    expect(formatFacet.label("text", zh)).toBe("纯文本");
    expect(statusFacet.label("not-indexed", en)).toBe("Not indexed yet");
    expect(statusFacet.label("unavailable", zh)).toBe("无法访问");
    expect(en(yearFacet.title)).toBe("Year");
    expect(zh(statusFacet.title)).toBe("状态");
  });
});

describe("a large Library", () => {
  test("filters and counts 20,000 Documents in one pass, quickly", () => {
    const many = Array.from({ length: 20_000 }, (_, index) =>
      doc({
        kind: (["pdf", "docx", "text"] as const)[index % 3],
        creationDate: index % 5 === 0 ? null : String(2000 + (index % 20)),
      }),
    );
    const started = performance.now();
    const { shown, options: all } = filterLibrary(many, documentFacets, { format: ["pdf"] });
    expect(performance.now() - started).toBeLessThan(250);
    expect(shown).toHaveLength(6667);
    expect(all.get("year")?.reduce((sum, option) => sum + option.count, 0)).toBe(6667);
  });
});
