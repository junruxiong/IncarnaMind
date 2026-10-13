/**
 * The Library page, grouped by Folder (#240): its sections, its search, its
 * problem line and its dates. Pure: no store, no bridge, so the tests import it.
 */

import type { Document, DocumentGroupAssignment, LibraryGroup } from "../../core/api";
import type { Language } from "../../core/language";
import { displayName } from "../../shared/documentNames";
import type { MessageKey, MessageParams } from "../../shared/i18n";
import { modelName } from "../../shared/modelName";
import { type LibraryFacet, type LibraryStatus, libraryStatus } from "./libraryFilters";
import { matchName, queryWords } from "./matcher";

type Translate = (key: MessageKey, params?: MessageParams) => string;

/** Not in a Folder's section, as a key beside the Folders' ids. */
export const NOT_IN_A_FOLDER = "not-in-a-folder";

/** The Recently deleted section at the page's foot. */
export const RECENTLY_DELETED = "recently-deleted";

/** How many Documents a section lists before "Show all N". */
export const FIRST_ROWS = 8;

export interface LibrarySection {
  /** The Folder's id, or `NOT_IN_A_FOLDER`. */
  key: string;
  folder: LibraryGroup | null;
  documents: Document[];
}

/**
 * One section per Folder, in the order given, then Not in a Folder, each with
 * its Documents in the order given. A Document whose Folder isn't among them
 * is Not in a Folder. Every Folder shows, empty or not, unless the page is
 * narrowed, when a section with no match is left out; Not in a Folder shows
 * only while it holds something.
 */
export function sectionsOf(
  folders: readonly LibraryGroup[],
  documents: readonly Document[],
  folderOf: (documentId: string) => string | null,
  narrowing: boolean,
): LibrarySection[] {
  const byFolder = new Map<string, Document[]>(folders.map((folder) => [folder.id, []]));
  const loose: Document[] = [];
  for (const document of documents) {
    const list = byFolder.get(folderOf(document.id) ?? "");
    if (list) list.push(document);
    else loose.push(document);
  }
  const sections: LibrarySection[] = folders.map((folder) => ({
    key: folder.id,
    folder,
    documents: byFolder.get(folder.id) ?? [],
  }));
  sections.push({ key: NOT_IN_A_FOLDER, folder: null, documents: loose });
  return sections.filter(
    (section) => section.documents.length > 0 || (section.folder !== null && !narrowing),
  );
}

/** The words of a search, folded as names are (see ./matcher): each must be found. */
export const searchWords = queryWords;

/**
 * Whether a Document's name, title or Tags hold every word, as ⌘K matches
 * names: a Chinese name by its pinyin too, and a long word with one typo.
 */
export function matchesSearch(
  document: Document,
  words: readonly string[],
  tagNames: ReadonlyMap<string, string>,
): boolean {
  if (words.length === 0) return true;
  const text = [
    displayName(document),
    document.name,
    document.title ?? "",
    ...document.tags.map((link) => tagNames.get(link.tagId) ?? ""),
  ].join(" ");
  return matchName(words, text) !== null;
}

/** What can be wrong with a Document that the problem line announces, in the order it names them. */
const PROBLEMS = ["no-text", "failed", "missing", "unavailable"] as const;
export type ProblemKind = (typeof PROBLEMS)[number];

export interface LibraryProblem {
  /** One kind, or "mixed" when the Documents have different ones. */
  kind: ProblemKind | "mixed";
  count: number;
  /** The status filter's options that show them. */
  statuses: LibraryStatus[];
}

/** The one line the page says about Documents it can't use, or null when there are none. */
export function problemOf(documents: readonly Document[]): LibraryProblem | null {
  const counts = new Map<ProblemKind, number>();
  for (const document of documents) {
    const status = libraryStatus(document);
    const kind = PROBLEMS.find((each) => each === status);
    if (kind) counts.set(kind, (counts.get(kind) ?? 0) + 1);
  }
  const kinds = PROBLEMS.filter((kind) => counts.has(kind));
  const [only] = kinds;
  if (!only) return null;
  const count = kinds.reduce((sum, kind) => sum + (counts.get(kind) ?? 0), 0);
  return { kind: kinds.length === 1 ? only : "mixed", count, statuses: kinds };
}

const PROBLEM_WORDS: Record<LibraryProblem["kind"], { one: MessageKey; many: MessageKey }> = {
  "no-text": { one: "library.problem.noText.one", many: "library.problem.noText" },
  failed: { one: "library.problem.failed.one", many: "library.problem.failed" },
  missing: { one: "library.problem.missing.one", many: "library.problem.missing" },
  unavailable: { one: "library.problem.unavailable.one", many: "library.problem.unavailable" },
  mixed: { one: "library.problem.mixed", many: "library.problem.mixed" },
};

/** The problem in words, in the singular or the plural. */
export function problemText(problem: LibraryProblem, t: Translate): string {
  const words = PROBLEM_WORDS[problem.kind];
  return t(problem.count === 1 ? words.one : words.many, { count: problem.count });
}

/**
 * When a Document was added, as its row says it: "Oct 9" this year, "Oct 9,
 * 2025" before (in Chinese "10月9日", "2025/10/9").
 */
export function addedLabel(iso: string, now: Date, language: Language): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const options: Intl.DateTimeFormatOptions =
    date.getFullYear() === now.getFullYear()
      ? { month: "short", day: "numeric" }
      : language === "zh-CN"
        ? { year: "numeric", month: "numeric", day: "numeric" }
        : { year: "numeric", month: "short", day: "numeric" };
  return new Intl.DateTimeFormat(language, options).format(date);
}

/** Where Organize is with a Document, in a word, for Get info. */
export function organizeState(assignment: DocumentGroupAssignment | undefined): MessageKey {
  if (assignment?.status === "pending" || assignment?.status === "classifying") {
    return "library.working";
  }
  if (assignment?.status === "waiting" || assignment?.status === "failed") {
    return "library.needsAttention";
  }
  if (assignment?.source === "user") return "library.manual";
  if (assignment?.status === "classified") {
    return assignment.suggested ? "library.suggested" : "library.saved";
  }
  return "library.notClassified";
}

/** Which model Organize used and what it read: "tev1:4b · Text", with why it took a smaller one. */
export function organizeModel(
  model: NonNullable<DocumentGroupAssignment["model"]>,
  t: Translate,
): string {
  const route =
    model.reason === "slow"
      ? "library.route.slow"
      : model.reason === "memory"
        ? "library.route.memory"
        : model.reason === "unavailable"
          ? "library.route.unavailable"
          : null;
  return [
    modelName(model.id),
    t(model.images ? "library.usedPages" : "library.usedText"),
    route && t(route),
  ]
    .filter(Boolean)
    .join(" · ");
}

/** By Folder: the Folders in the order given, then Not in a Folder. */
export function folderFacet(
  folders: readonly Pick<LibraryGroup, "id" | "name">[],
  folderOf: (documentId: string) => string | null,
): LibraryFacet<Document> {
  const names = new Map(folders.map((folder) => [folder.id, folder.name]));
  const order = [...folders.map((folder) => folder.id), NOT_IN_A_FOLDER];
  return {
    id: "folder",
    title: "library.filter.folder",
    values: (document) => {
      const id = folderOf(document.id);
      return [id !== null && names.has(id) ? id : NOT_IN_A_FOLDER];
    },
    label: (value, t) =>
      value === NOT_IN_A_FOLDER ? t("library.unsorted") : (names.get(value) ?? value),
    compare: (a, b) => order.indexOf(a) - order.indexOf(b),
  };
}
