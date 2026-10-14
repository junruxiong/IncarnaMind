import { create } from "zustand";
import { SORT_KEYS, type Sort } from "./sort";

const KEY = "incarnamind.sort";

/** Where a choice is made: a Folder (by its id), the sidebar's top level, or a view of the Library. */
export type SortScope = `folder:${string}` | "sidebar" | `library:${string}`;

interface Saved {
  scopes: Record<string, Sort>;
  all: Sort | null;
}

const isSort = (value: unknown): value is Sort => {
  const sort = value as Partial<Sort> | null;
  return (
    typeof sort === "object" &&
    sort !== null &&
    SORT_KEYS.some((key) => key === sort.key) &&
    (sort.direction === "asc" || sort.direction === "desc") &&
    typeof sort.foldersFirst === "boolean"
  );
};

function read(): Saved {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? "null") as Partial<Saved> | null;
    const scopes = Object.fromEntries(
      Object.entries(value?.scopes ?? {}).filter(([, sort]) => isSort(sort)),
    );
    return { scopes, all: isSort(value?.all) ? value.all : null };
  } catch {
    return { scopes: {}, all: null };
  }
}

function save({ scopes, all }: Saved): void {
  try {
    localStorage.setItem(KEY, JSON.stringify({ scopes, all }));
  } catch {
    // Not remembered, then: the order starts as it was next time.
  }
}

interface SortChoices extends Saved {
  choose(scope: SortScope, sort: Sort): void;
  /** One choice for every Folder and the sidebar, the ones chosen apart forgotten. */
  chooseForAll(sort: Sort): void;
}

/**
 * How each Folder, the sidebar and each view of the Library is sorted,
 * remembered on this computer. Without a choice a place keeps its own order:
 * Minds newest changed first, Documents newest added first.
 */
export const useSort = create<SortChoices>()((set, get) => {
  const change = (next: Saved) => {
    set(next);
    save(next);
  };
  return {
    ...read(),
    choose: (scope, sort) => change({ scopes: { ...get().scopes, [scope]: sort }, all: get().all }),
    chooseForAll: (sort) =>
      change({
        scopes: Object.fromEntries(
          Object.entries(get().scopes).filter(([scope]) => scope.startsWith("library:")),
        ),
        all: sort,
      }),
  };
});

/** What a place is sorted by: its own choice, else the one for all Folders (not for the Library), else none. */
export const sortOf =
  (scope: SortScope) =>
  (state: Saved): Sort | null =>
    state.scopes[scope] ?? (scope.startsWith("library:") ? null : state.all);
