import { describe, expect, test } from "vitest";
import type { Document, DocumentGroupAssignment, LibraryGroup } from "../../src/core/api";
import {
  addedLabel,
  folderFacet,
  matchesSearch,
  NOT_IN_A_FOLDER,
  organizeModel,
  organizeState,
  problemOf,
  problemText,
  RECENTLY_DELETED,
  searchWords,
  sectionShows,
  sectionsOf,
} from "../../src/renderer/src/libraryPage";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

const en = (key: MessageKey, params?: MessageParams) => translate("en", key, params);
const zh = (key: MessageKey, params?: MessageParams) => translate("zh-CN", key, params);

let next = 0;
/** A ready Document whose file is where it was, unless said otherwise. */
const doc = (changes: Partial<Document> = {}): Document =>
  ({
    id: `doc-${++next}`,
    name: `Document ${next}`,
    title: null,
    kind: "pdf",
    status: "ready",
    fileStatus: "available",
    tags: [],
    createdAt: "2026-10-09T10:00:00.000Z",
    ...changes,
  }) as Document;

const folder = (id: string, name = id): LibraryGroup => ({
  id,
  name,
  description: "",
  countsOwnFiles: false,
  createdAt: "2026-10-01T00:00:00.000Z",
  updatedAt: "2026-10-01T00:00:00.000Z",
});

describe("sections", () => {
  const board = folder("board", "Board");
  const finance = folder("finance", "Finance");
  const a = doc();
  const b = doc();
  const c = doc();
  const where = new Map<string, string | null>([
    [a.id, "finance"],
    [b.id, null],
    [c.id, "gone"],
  ]);
  const folderOf = (id: string) => where.get(id) ?? null;

  test("one per Folder in the order given, then Not in a Folder, each with its Documents in order", () => {
    const sections = sectionsOf([board, finance], [a, b, c], folderOf, false);
    expect(sections.map((section) => section.key)).toEqual(["board", "finance", NOT_IN_A_FOLDER]);
    expect(sections[1]?.documents).toEqual([a]);
    // A Document in a Folder that isn't there any more is in none.
    expect(sections[2]?.documents).toEqual([b, c]);
  });

  test("an empty Folder still shows; Not in a Folder only while it holds something", () => {
    expect(sectionsOf([board, finance], [a], folderOf, false).map((s) => s.key)).toEqual([
      "board",
      "finance",
    ]);
  });

  test("while narrowed, sections with no match are left out", () => {
    expect(sectionsOf([board, finance], [a], folderOf, true).map((s) => s.key)).toEqual([
      "finance",
    ]);
    expect(sectionsOf([board, finance], [], folderOf, true)).toEqual([]);
  });

  test("a section asked for is there only if it would show, so the page never waits to scroll to it", () => {
    const assignments = [
      { documentId: a.id, groupId: "finance" },
      { documentId: c.id, groupId: "gone" },
    ];
    const shows = (key: string, documents: Document[], deleted: number | null = null) =>
      sectionShows(key, documents, [board, finance], assignments, deleted);
    expect(shows("board", [])).toBe(true);
    expect(shows("gone", [a, c])).toBe(false);
    expect(shows(NOT_IN_A_FOLDER, [a])).toBe(false);
    // A Document whose Folder has gone is Not in a Folder.
    expect(shows(NOT_IN_A_FOLDER, [a, c])).toBe(true);
    expect(shows(RECENTLY_DELETED, [], 0)).toBe(false);
    expect(shows(RECENTLY_DELETED, [], 2)).toBe(true);
    // Not read yet: it may be there.
    expect(shows(RECENTLY_DELETED, [], null)).toBe(true);
  });
});

describe("search", () => {
  const tagNames = new Map([["t1", "Contract"]]);
  const lease = doc({
    name: "lease_2026",
    title: "Lease: Rotterdam warehouse",
    tags: [{ tagId: "t1" } as Document["tags"][number]],
  });

  test("every word must be found, in the name, the title or a Tag, whatever the case", () => {
    expect(searchWords("  Rotterdam   CONTRACT ")).toEqual(["rotterdam", "contract"]);
    expect(matchesSearch(lease, searchWords("rotterdam contract"), tagNames)).toBe(true);
    expect(matchesSearch(lease, searchWords("lease_2026"), tagNames)).toBe(true);
    expect(matchesSearch(lease, searchWords("rotterdam invoice"), tagNames)).toBe(false);
    expect(matchesSearch(lease, [], tagNames)).toBe(true);
  });

  test("Chinese names are found by their characters", () => {
    const minutes = doc({ name: "会议纪要 0402" });
    expect(matchesSearch(minutes, searchWords("纪要"), tagNames)).toBe(true);
    expect(matchesSearch(minutes, searchWords("财务"), tagNames)).toBe(false);
  });
});

describe("the problem line", () => {
  test("none while every Document can be used, or is still being read", () => {
    expect(problemOf([doc(), doc({ status: "embedding" })])).toBeNull();
  });

  test("one kind is named, in the singular and the plural, in English and Chinese", () => {
    const one = problemOf([doc(), doc({ status: "no-text" })]);
    expect(one).toEqual({ kind: "no-text", count: 1, statuses: ["no-text"] });
    if (!one) throw new Error("No problem found.");
    expect(problemText(one, en)).toBe("1 Document has no text: a scan");
    expect(problemText(one, zh)).toBe("1 个文档没有文字：扫描件");
    const two = problemOf([doc({ fileStatus: "missing" }), doc({ fileStatus: "missing" })]);
    if (!two) throw new Error("No problem found.");
    expect(problemText(two, en)).toBe("2 Documents' files are missing");
    expect(problemText(two, zh)).toBe("2 个文档的文件缺失");
  });

  test("several kinds are counted together, and Show filters to all of them", () => {
    const mixed = problemOf([
      doc({ status: "failed" }),
      doc({ status: "no-text" }),
      doc({ fileStatus: "unavailable" }),
    ]);
    expect(mixed).toEqual({
      kind: "mixed",
      count: 3,
      statuses: ["no-text", "failed", "unavailable"],
    });
    if (!mixed) throw new Error("No problem found.");
    expect(problemText(mixed, en)).toBe("3 Documents need attention");
  });
});

describe("dates", () => {
  const now = new Date("2026-10-13T12:00:00Z");

  test("this year's as a day and a month, earlier ones with the year", () => {
    expect(addedLabel("2026-10-09T10:00:00Z", now, "en")).toBe("9 Oct");
    expect(addedLabel("2025-03-04T10:00:00Z", now, "en")).toBe("4 Mar 2025");
    expect(addedLabel("2026-10-09T10:00:00Z", now, "zh-CN")).toBe("10月9日");
    expect(addedLabel("2025-03-04T10:00:00Z", now, "zh-CN")).toBe("2025年3月4日");
    expect(addedLabel("not a date", now, "en")).toBe("");
  });
});

describe("what Organize did, as Get info says it", () => {
  const assignment = (changes: Partial<DocumentGroupAssignment>): DocumentGroupAssignment => ({
    documentId: "d",
    groupId: "finance",
    source: "automatic",
    suggested: false,
    status: "classified",
    error: null,
    model: null,
    ...changes,
  });

  test("its state in a word", () => {
    expect(en(organizeState(undefined))).toBe("Not organized yet");
    expect(en(organizeState(assignment({ status: "classifying" })))).toBe("Organizing…");
    expect(en(organizeState(assignment({ status: "failed" })))).toBe("Needs attention");
    expect(en(organizeState(assignment({ source: "user" })))).toBe("Your choice");
    expect(en(organizeState(assignment({ suggested: true })))).toBe("Suggested");
    expect(en(organizeState(assignment({})))).toBe("Organized");
  });

  test("the model it used, what it read, and why it took a smaller one", () => {
    expect(organizeModel({ id: "tev1:4b", images: false, reason: "text" }, en)).toBe(
      "tev1:4b · Text",
    );
    expect(organizeModel({ id: "clef-flash", images: true, reason: "visual" }, en)).toBe(
      "clef-flash · Text + PDF pages",
    );
    expect(organizeModel({ id: "tev1:0.8b", images: false, reason: "memory" }, en)).toBe(
      "tev1:0.8b · Text · Smaller model for memory limits",
    );
  });
});

describe("the Folder filter", () => {
  test("each Document under its Folder or Not in a Folder, the Folders in order and none last", () => {
    const where = new Map([["d1", "finance"]]);
    const facet = folderFacet(
      [folder("board", "Board"), folder("finance", "Finance")],
      (id) => where.get(id) ?? null,
    );
    expect(facet.values(doc({ id: "d1" }))).toEqual(["finance"]);
    expect(facet.values(doc({ id: "d2" }))).toEqual([NOT_IN_A_FOLDER]);
    expect([NOT_IN_A_FOLDER, "finance", "board"].sort(facet.compare)).toEqual([
      "board",
      "finance",
      NOT_IN_A_FOLDER,
    ]);
    expect(facet.label(NOT_IN_A_FOLDER, en)).toBe("Not in a Folder");
    expect(facet.label(NOT_IN_A_FOLDER, zh)).toBe("不在文件夹中");
    expect(facet.label("finance", en)).toBe("Finance");
  });
});
