import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type { Document, LibraryGroup } from "../../../core/api";
import { LIBRARY_DELETED } from "../deletedItems";
import { useFindSources } from "../findSources";
import { useLanguage, useT } from "../i18n";
import {
  type FilterSelection,
  filterLibrary,
  formatFacet,
  isFiltering,
  type LibraryFacet,
  statusFacet,
  tagFacet,
  toggleFilterValue,
  yearFacet,
} from "../libraryFilters";
import { useLibraryFolds } from "../libraryFolds";
import {
  folderFacet,
  matchesSearch,
  NOT_IN_A_FOLDER,
  problemOf,
  problemText,
  RECENTLY_DELETED,
  searchWords,
  sectionsOf,
} from "../libraryPage";
import { extendThroughRows, useSelection, useSelectionOrder } from "../selection";
import { DEFAULT_SORT, sorted } from "../sort";
import { sortOf, useSort } from "../sortChoices";
import { documentFields, folderFields } from "../sortFields";
import { useAppStore } from "../store";
import { moveThroughRows } from "../treeKeys";
import { FindSourcesPanel } from "./FindSourcesPanel";
import { LibrarySelectionBar } from "./LibrarySelection";
import { DeletedSection } from "./library/DeletedSection";
import { DocumentMoveTo } from "./library/DocumentMoveTo";
import { FolderForm, StarterFolders } from "./library/FolderForm";
import { LibraryBar } from "./library/LibraryBar";
import { LibraryEmpty } from "./library/LibraryEmpty";
import { LibraryHeader } from "./library/LibraryHeader";
import { LibrarySection } from "./library/LibrarySection";
import { ProblemLine } from "./ProblemLine";
import { ReferenceImportPanel } from "./ReferenceImportPanel";

const NO_GROUPS: readonly LibraryGroup[] = [];

/** The Library's one order for every section, remembered on this computer. */
const SORT_SCOPE = "library:all" as const;

/** What `openLibrary` asked for that is a form at the top, not a section to show. */
const FORMS = ["all", "new", "starters"];

/**
 * The Library tab (#240): one page of every Document, grouped by Folder, as a
 * Notion page of toggles or Finder's list view. Its head holds the title,
 * Add Documents and the search; then the quiet bar (filters, sort, folding) or,
 * with two or more selected, the selection's bar; a problem in one line; one
 * section per Folder in the sidebar's order, then Not in a Folder; and
 * Recently deleted folded at the foot. A Folder opened from the sidebar brings
 * its section into view; the page still shows every Folder.
 */
export function LibraryPane() {
  const t = useT();
  const language = useLanguage();
  const documents = useAppStore((state) => state.documents);
  const snapshot = useAppStore((state) => state.library);
  const tags = useAppStore((state) => state.tags);
  const tagFilter = useAppStore(useShallow((state) => state.tagFilter));
  const request = useAppStore((state) => state.libraryFilter);
  const findingSources = useFindSources((state) => state.open);
  const [search, setSearch] = useState("");
  /** The format, year, status and Folder filters' choices; the Tags filter is the store's, shared with the sidebar. */
  const [facetFilter, setFacetFilter] = useState<FilterSelection>({});
  const field = useRef<HTMLInputElement>(null);

  const groups = snapshot?.groups ?? NO_GROUPS;
  const assignments = useMemo(
    () => new Map(snapshot?.assignments.map((item) => [item.documentId, item])),
    [snapshot?.assignments],
  );
  const folderOf = useCallback(
    (documentId: string) => assignments.get(documentId)?.groupId ?? null,
    [assignments],
  );
  const sidebarSort = useSort(sortOf("sidebar"));
  const folders = useMemo(
    () => sorted(groups, sidebarSort, folderFields, language),
    [groups, sidebarSort, language],
  );
  const facets = useMemo<LibraryFacet<Document>[]>(
    () => [tagFacet(tags), formatFacet, yearFacet, statusFacet, folderFacet(folders, folderOf)],
    [tags, folders, folderOf],
  );
  const selection = useMemo<FilterSelection>(
    () => (tagFilter.length ? { ...facetFilter, tag: tagFilter } : facetFilter),
    [facetFilter, tagFilter],
  );
  const words = useMemo(() => searchWords(search), [search]);
  const found = useMemo(() => {
    if (words.length === 0) return documents;
    const tagNames = new Map(tags.map((tag) => [tag.id, tag.name]));
    return documents.filter((doc) => matchesSearch(doc, words, tagNames));
  }, [documents, words, tags]);
  const { shown: filtered, options } = useMemo(
    () => filterLibrary<Document>(found, facets, selection),
    [found, facets, selection],
  );
  const chosenSort = useSort(sortOf(SORT_SCOPE));
  const ordered = useMemo(
    () => sorted(filtered, chosenSort, documentFields(t), language),
    [filtered, chosenSort, t, language],
  );
  const narrowing = words.length > 0 || isFiltering(selection);
  const sections = useMemo(
    () => sectionsOf(folders, ordered, folderOf, narrowing),
    [folders, ordered, folderOf, narrowing],
  );
  const folded = useLibraryFolds((state) => state.folded);
  const shownIds = useMemo(
    () =>
      sections.flatMap((section) =>
        folded.has(section.key) ? [] : section.documents.map((doc) => doc.id),
      ),
    [sections, folded],
  );
  // ⌘A and Shift-click reach every row in an open section, drawn or not.
  useSelectionOrder("library", () => shownIds.map((id) => ({ kind: "document", id })));
  // What is selected stays in view: a search, a filter or a fold that hides it, or a deletion, drops it.
  useEffect(() => {
    const held = useSelection.getState();
    if (held.surface !== "library") return;
    const visible = new Set(shownIds);
    held.keepOnly((ref) => visible.has(ref.id));
  }, [shownIds]);
  const selectingMany = useSelection(
    (state) => state.surface === "library" && state.items.length > 1,
  );

  const suggestedOf = useCallback(
    (documentId: string) => {
      const item = assignments.get(documentId);
      if (!item?.suggested || item.groupId === null) return null;
      return groups.find((group) => group.id === item.groupId)?.name ?? null;
    },
    [assignments, groups],
  );
  const notSortedOf = useCallback(
    (documentId: string) => {
      const item = assignments.get(documentId);
      return item?.status === "failed" ? (item.error?.message ?? "") : null;
    },
    [assignments],
  );
  const problem = useMemo(() => problemOf(documents), [documents]);

  // A Folder, Not in a Folder or Recently deleted asked for (from the sidebar, a menu or ⌘K): its section opens and comes into view.
  useEffect(() => {
    if (FORMS.includes(request)) return;
    const key =
      request === LIBRARY_DELETED
        ? RECENTLY_DELETED
        : request === "unsorted"
          ? NOT_IN_A_FOLDER
          : request;
    setSearch("");
    setFacetFilter({});
    useLibraryFolds.getState().reveal(key);
    useAppStore.setState({ libraryFilter: "all" });
  }, [request]);

  const sectionKeys = sections.map((section) => section.key);
  const allFolded = sectionKeys.length > 0 && sectionKeys.every((key) => folded.has(key));
  const clearNarrowing = () => {
    setSearch("");
    setFacetFilter({});
    useAppStore.getState().setTagFilter([]);
  };
  const closeForm = () => useAppStore.getState().openLibrary();

  return (
    <section
      data-testid="library"
      data-surface="library"
      role="tabpanel"
      id="mind-tabpanel"
      aria-labelledby="library-tab"
      className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-sheet text-ui text-ink"
    >
      <div className="@container/page min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1084px] pt-5 pr-8 pb-24 pl-11 @max-[680px]/page:pr-4 @max-[680px]/page:pl-9">
          <LibraryHeader
            count={documents.length}
            search={search}
            onSearch={setSearch}
            field={field}
          />
          {findingSources && <FindSourcesPanel />}
          <ReferenceImportPanel />
          {request === "new" && <FolderForm group="new" close={closeForm} className="mt-4" />}
          {request === "starters" && <StarterFolders close={closeForm} />}
          {!snapshot ? (
            <p role="status" className="mt-6 text-[13px] text-ink-meta">
              {t("library.loading")}
            </p>
          ) : documents.length === 0 ? (
            <LibraryEmpty />
          ) : (
            <>
              {selectingMany ? (
                <LibrarySelectionBar shownIds={shownIds} />
              ) : (
                <LibraryBar
                  facets={facets}
                  options={options}
                  selection={selection}
                  onToggle={(facetId, value) => {
                    if (facetId === "tag") useAppStore.getState().toggleTagFilter(value);
                    else setFacetFilter((current) => toggleFilterValue(current, facetId, value));
                  }}
                  onRemove={(facetId) => {
                    if (facetId === "tag") useAppStore.getState().setTagFilter([]);
                    else setFacetFilter(({ [facetId]: _removed, ...rest }) => rest);
                  }}
                  sort={chosenSort ?? DEFAULT_SORT}
                  sortScope={SORT_SCOPE}
                  allFolded={allFolded}
                  onFoldAll={() => useLibraryFolds.getState().setAll(sectionKeys, allFolded)}
                />
              )}
              {problem && (
                <ProblemLine
                  role="status"
                  testId="library-problem"
                  className="mt-1"
                  action={{
                    label: t("library.problem.show"),
                    testId: "library-problem-show",
                    onClick: () =>
                      setFacetFilter((current) => ({ ...current, status: problem.statuses })),
                  }}
                >
                  <span
                    aria-hidden="true"
                    className="mr-2 mb-px inline-block size-1.5 rounded-full bg-attention align-middle"
                  />
                  {problemText(problem, t)}
                </ProblemLine>
              )}
              {/* biome-ignore lint/a11y/useSemanticElements: sections of rows laid out as columns, not a table. */}
              <div
                role="grid"
                aria-multiselectable="true"
                aria-label={t("library.sections")}
                onKeyDown={(event) => {
                  if (!moveThroughRows(event)) extendThroughRows(event, "library");
                }}
              >
                {sections.map((section) => (
                  <LibrarySection
                    key={section.key}
                    section={section}
                    narrowing={narrowing}
                    suggestedOf={suggestedOf}
                    notSortedOf={notSortedOf}
                  />
                ))}
              </div>
              {narrowing && sections.length === 0 && (
                <ProblemLine
                  role="status"
                  testId="library-no-match"
                  className="mt-6"
                  action={{
                    label: t("library.noMatch.clear"),
                    testId: "library-clear-narrowing",
                    onClick: clearNarrowing,
                  }}
                >
                  {t("library.noMatch")}
                </ProblemLine>
              )}
            </>
          )}
          {!narrowing && <DeletedSection />}
        </div>
      </div>
      <DocumentMoveTo />
    </section>
  );
}
