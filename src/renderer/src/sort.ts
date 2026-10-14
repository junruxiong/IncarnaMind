import type { Language } from "../../core/language";

export const SORT_KEYS = ["name", "added", "modified", "kind"] as const;
export type SortKey = (typeof SORT_KEYS)[number];

export interface Sort {
  key: SortKey;
  direction: "asc" | "desc";
  /** Folders and Minds before Documents, each sorted among themselves. */
  foldersFirst: boolean;
}

/** What the menu shows chosen where nothing was: Documents newest added first. */
export const DEFAULT_SORT: Sort = { key: "added", direction: "desc", foldersFirst: true };

/** What an object is sorted by. */
export interface SortFields {
  name: string;
  /** The kind as the User reads it ("PDF", "Word"), so the order is the one on screen. */
  kind: string;
  /** ISO 8601, UTC. */
  added: string;
  /** ISO 8601, UTC; none for an object with no time of its own, which then sorts as added. */
  modified: string | null;
}

/**
 * Names compare in the interface language: Chinese by pinyin, ignoring case
 * and the order of digits ("2" before "10"), and a letter with an accent after
 * the one without it.
 */
const collators = new Map<Language, Intl.Collator>();
function collatorFor(language: Language): Intl.Collator {
  let collator = collators.get(language);
  if (!collator) {
    collator = new Intl.Collator(language === "zh-CN" ? "zh-CN-u-co-pinyin" : language, {
      sensitivity: "accent",
      numeric: true,
    });
    collators.set(language, collator);
  }
  return collator;
}

const byText = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** A comparison for `sort`: the key either way, ties by name, then by when it was added. */
export function comparer(sort: Sort, language: Language): (a: SortFields, b: SortFields) => number {
  const collator = collatorFor(language);
  const sign = sort.direction === "asc" ? 1 : -1;
  const named = (a: SortFields, b: SortFields) =>
    collator.compare(a.name, b.name) || byText(a.added, b.added);
  return (a, b) => {
    switch (sort.key) {
      case "name":
        return sign * collator.compare(a.name, b.name) || byText(a.added, b.added);
      case "kind":
        return sign * collator.compare(a.kind, b.kind) || named(a, b);
      case "added":
        return sign * byText(a.added, b.added) || named(a, b);
      case "modified":
        return sign * byText(a.modified ?? a.added, b.modified ?? b.added) || named(a, b);
    }
  };
}

/** The items in `sort`'s order; the same array, untouched, when nothing was chosen. */
export function sorted<T>(
  items: readonly T[],
  sort: Sort | null,
  fields: (item: T) => SortFields,
  language: Language,
): readonly T[] {
  if (!sort) return items;
  const compare = comparer(sort, language);
  // The fields are read once each, not on every comparison.
  return items
    .map((item) => ({ item, fields: fields(item) }))
    .sort((a, b) => compare(a.fields, b.fields))
    .map((each) => each.item);
}
