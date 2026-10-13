import { memo, type Ref, useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type {
  Document,
  DocumentGroupAssignment,
  LibraryGroup,
  ReferenceField,
  ReferenceRecord,
} from "../../../core/api";
import { displayName, fileNameOf, sourceOf } from "../../../shared/documentNames";
import { libraryPresetKeys } from "../../../shared/libraryPresets";
import { missingDetails } from "../../../shared/references";
import { core } from "../core";
import { LIBRARY_DELETED } from "../deletedItems";
import { errorMessage as messageOf } from "../errors";
import { useFindSources } from "../findSources";
import { useLanguage, useT } from "../i18n";
import { bridgeFrom, type LibraryBridge, type LibraryView } from "../libraryBridges";
import {
  documentFacets,
  type FilterSelection,
  filterLibrary,
  isFiltering,
  type LibraryFacet,
  tagFacet,
  toggleFilterValue,
} from "../libraryFilters";
import { formatCount } from "../linkedFolders";
import { missingList } from "../referenceEditing";
import { useRequests } from "../requests";
import { extendThroughRows, useSelectable, useSelection, useSelectionOrder } from "../selection";
import { DEFAULT_SORT, sorted } from "../sort";
import { sortOf, useSort } from "../sortChoices";
import { documentFields } from "../sortFields";
import { useSortMenu } from "../sortMenu";
import { useAppStore } from "../store";
import { moveThroughRows, TREE_ROW } from "../treeKeys";
import { useRename } from "../useRename";
import { AddDocumentsMenu } from "./AddDocumentsMenu";
import { CiteAsPanel } from "./CiteAsPanel";
import { FindSourcesPanel } from "./FindSourcesPanel";
import { InlineName } from "./InlineName";
import { LibraryFilterBar } from "./LibraryFilterBar";
import { LibraryMoreMenu } from "./LibraryMoreMenu";
import { LibrarySelectBox, LibrarySelectionBar } from "./LibrarySelection";
import { DocumentLineIcon, SettingsLineIcon } from "./lineIcons";
import { ObjectMenu, useObjectMenu } from "./ObjectMenu";
import { RecentlyDeleted } from "./RecentlyDeleted";
import { ReferenceImportPanel } from "./ReferenceImportPanel";
import { SORT_KEY_LABELS } from "./SortMenu";
import { rowActionButtonClass } from "./sidebarRows";
import { DocumentTagChips } from "./TagChips";
import { useTabPlace } from "./tabs/tabItems";
import { buttonStyle, errorTextClass, fieldLabelClass, inputClass } from "./ui";

type Action = (work: () => Promise<unknown>) => Promise<boolean>;

const NO_GROUPS: readonly LibraryGroup[] = [];

/** Keeps the preview showing a Document, if one does. */
function keepPreviewOf(documentId: string): void {
  const { layout, changeLayout } = useAppStore.getState();
  const preview = layout?.groups
    .flatMap((group) => group.tabs)
    .find(
      (tab) => tab.preview && tab.item.kind === "document" && tab.item.documentId === documentId,
    );
  if (preview) void changeLayout({ type: "keep", tabId: preview.id });
}

/** The Library tab: the sheet of a Folder's Documents, or Recently deleted. */
export function LibraryPane() {
  const deleted = useAppStore((state) => state.libraryFilter === LIBRARY_DELETED);
  return deleted ? <RecentlyDeleted /> : <LibrarySheet />;
}

/** The sidebar chooses a folder; this sheet shows its documents and editable tags. */
function LibrarySheet() {
  const t = useT();
  const language = useLanguage();
  // Every Document: the Tags filter below is the sidebar's, and counts its options from all.
  const documents = useAppStore((state) => state.documents);
  const hasDocuments = useAppStore((state) => state.documents.length > 0);
  const snapshot = useAppStore((state) => state.library);
  const filter = useAppStore((state) => state.libraryFilter);
  const tags = useAppStore((state) => state.tags);
  const tagFilter = useAppStore(useShallow((state) => state.tagFilter));
  const findingSources = useFindSources((state) => state.open);
  const [search, setSearch] = useState("");
  /** The year, format and status filters' choices (see `LibraryFilterBar`). */
  const [facetFilter, setFacetFilter] = useState<FilterSelection>({});
  /** The filters, the Tags one last; its choice is the store's Tag filter, shared with the sidebar. */
  const facets = useMemo<LibraryFacet<Document>[]>(
    () => [...documentFacets, tagFacet(tags)],
    [tags],
  );
  const selection = useMemo<FilterSelection>(
    () => (tagFilter.length ? { ...facetFilter, tag: tagFilter } : facetFilter),
    [facetFilter, tagFilter],
  );
  const [limit, setLimit] = useState(100);
  const [editing, setEditing] = useState<LibraryGroup | "new" | null>(null);
  const [starters, setStarters] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setLimit(100);
    setSearch("");
    setFacetFilter({});
    setEditing(filter === "new" ? "new" : null);
    // Another Folder: what was selected in this one isn't in view any more.
    useSelection.getState().clear("library");
  }, [filter]);
  // The same function from one drawing to the next, so the rows needn't be drawn again for it.
  const act: Action = useCallback(async (work) => {
    setError(null);
    setBusy(true);
    try {
      await work();
      await useAppStore.getState().refreshLibrary();
      return true;
    } catch (failure) {
      setError(messageOf(failure));
      return false;
    } finally {
      setBusy(false);
    }
  }, []);
  const groups = snapshot?.groups ?? NO_GROUPS;
  const assignments = useMemo(
    () => new Map(snapshot?.assignments.map((item) => [item.documentId, item])),
    [snapshot?.assignments],
  );
  const selected = groups.find((group) => group.id === filter);
  // The sheet's Documents: its Folder's, found by the search; then those the filters keep.
  const inView = useMemo(() => {
    const tagNames = new Map(tags.map((tag) => [tag.id, tag.name]));
    const folder = groups.find((group) => group.id === filter);
    const wanted = search.toLocaleLowerCase();
    // Without a search, every Document's words match: they aren't put together then.
    const found = (doc: Document) =>
      wanted === "" ||
      [doc.name, doc.title ?? "", ...doc.tags.map((link) => tagNames.get(link.tagId) ?? "")]
        .join(" ")
        .toLocaleLowerCase()
        .includes(wanted);
    return documents.filter((doc) => {
      const groupId = assignments.get(doc.id)?.groupId ?? null;
      return (
        ((!folder && filter !== "unsorted") ||
          (filter === "unsorted" ? groupId === null : groupId === folder?.id)) &&
        found(doc)
      );
    });
  }, [documents, assignments, groups, tags, filter, search]);
  const { shown: filtered, options } = useMemo(
    () => filterLibrary<Document>(inView, facets, selection),
    [inView, facets, selection],
  );
  const sortScope = `library:${filter}` as const;
  const chosenSort = useSort(sortOf(sortScope));
  const sort = chosenSort ?? DEFAULT_SORT;
  const shown = useMemo(
    () => sorted(filtered, chosenSort, documentFields(t), language),
    [filtered, chosenSort, t, language],
  );
  /** A column's header sorts by it, and a second click reverses it. */
  const sortBy = (key: "name") =>
    useSort.getState().choose(sortScope, {
      ...sort,
      key,
      direction: sort.key === key && sort.direction === "asc" ? "desc" : "asc",
    });
  const sortLabel = `${t(SORT_KEY_LABELS[sort.key])} ${sort.direction === "asc" ? "↑" : "↓"}`;
  const shownIds = useMemo(() => shown.map((doc) => doc.id), [shown]);
  // ⌘A and Shift-click reach every row the filters show, drawn or not.
  useSelectionOrder("library", () => shownIds.map((id) => ({ kind: "document", id })));
  // What is selected stays in view: a search or filter that hides it, or a deleted Document, drops it.
  useEffect(() => {
    const held = useSelection.getState();
    if (held.surface !== "library") return;
    const visible = new Set(shownIds);
    held.keepOnly((ref) => visible.has(ref.id));
  }, [shownIds]);
  const filtering = isFiltering(selection);
  const view: LibraryView = selected
    ? { kind: "folder", id: selected.id, name: selected.name }
    : filter === "unsorted"
      ? { kind: "unsorted", name: t("library.unsorted") }
      : { kind: "all", name: t("library.all") };
  // With anything narrowing the list, a Question from here searches exactly what it shows.
  const bridge = bridgeFrom(view, shownIds, filtering || search.trim() !== "");
  const { pending, waiting, failed } = useMemo(() => {
    const counts = { pending: 0, waiting: 0, failed: 0 };
    for (const item of snapshot?.assignments ?? []) {
      if (item.status === "pending" || item.status === "classifying") counts.pending++;
      else if (item.status === "waiting") counts.waiting++;
      else if (item.status === "failed") counts.failed++;
    }
    return counts;
  }, [snapshot?.assignments]);
  const needsOrganization = shown.filter((doc) => {
    const item = assignments.get(doc.id);
    return item?.status !== "classified" || doc.tagging !== "tagged";
  });
  const editFolder = useRequests((state) => state.editFolder);
  useEffect(() => {
    if (editFolder && selected?.id === editFolder) {
      setEditing(selected);
      useRequests.getState().clear("editFolder");
    }
  }, [editFolder, selected]);
  const organize = () => {
    if (!snapshot?.settings.classifier) {
      useAppStore.getState().openSettings("models");
      return;
    }
    void act(() =>
      core.classifyDocuments(
        (needsOrganization.length ? needsOrganization : shown).map((doc) => doc.id),
      ),
    );
  };
  const renameFolder = useAppStore((state) => state.renameFolder);
  const renameHeading = useRename<HTMLButtonElement>("library-heading");
  const heading = selected?.name ?? t(filter === "unsorted" ? "library.unsorted" : "library.all");
  const closeForm = () => {
    setEditing(null);
    if (filter === "new") useAppStore.getState().openLibrary();
  };
  return (
    <section
      data-testid="library"
      data-surface="library"
      role="tabpanel"
      id="mind-tabpanel"
      aria-labelledby="library-tab"
      className="library-pane flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-sheet text-ui text-ink"
    >
      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-[1040px]">
          {/* The search comes first (#119). */}
          <div className="mb-5 flex flex-wrap items-center gap-3">
            <input
              aria-label={t("library.search")}
              placeholder={t("library.search")}
              type="search"
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setLimit(100);
              }}
              className={`${inputClass} mt-0 min-w-32 flex-1`}
            />
            <AddDocumentsMenu />
            <LibraryMoreMenu />
          </div>
          {findingSources && <FindSourcesPanel />}
          {/* The name takes the whole width and wraps by word, two lines at most (#213); the actions sit below it, never beside. */}
          <div className="mb-2">
            <h1
              data-testid="library-heading"
              title={heading}
              className="line-clamp-2 font-serif text-heading font-semibold [overflow-wrap:break-word]"
            >
              {selected && renameHeading.renaming ? (
                <InlineName
                  value={selected.name}
                  label={t("folder.renameLabel", { name: selected.name })}
                  testId="library-heading-rename"
                  className="h-9 w-full min-w-0 rounded-sm border border-accent bg-sheet px-1.5 font-serif text-heading font-semibold text-ink outline-1 outline-accent select-text"
                  {...renameHeading.field((name) => renameFolder(selected.id, name))}
                />
              ) : selected ? (
                <button
                  {...renameHeading.nameProps}
                  type="button"
                  className="cursor-text text-left [overflow-wrap:break-word]"
                >
                  {heading}
                </button>
              ) : (
                heading
              )}
            </h1>
            <p
              title={selected?.description || t("library.intro")}
              className="mt-1 line-clamp-2 text-[13px] text-ink-meta"
            >
              {selected?.description || t("library.intro")}
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {bridge && <WritingBridges bridge={bridge} />}
              <button
                type="button"
                className={buttonStyle("primary")}
                disabled={busy || pending > 0 || groups.length === 0 || shown.length === 0}
                onClick={organize}
              >
                {t(pending ? "library.working" : "library.classify")}
              </button>
            </div>
          </div>
          <div className="mb-5 flex flex-wrap items-center gap-1">
            <button
              type="button"
              className={buttonStyle("ghost", "sm")}
              onClick={() => {
                setEditing("new");
                setStarters(false);
              }}
            >
              {t("library.newGroup")}
            </button>
            <button
              type="button"
              className={buttonStyle("ghost", "sm")}
              onClick={() => {
                setStarters(!starters);
                setEditing(null);
              }}
            >
              {t("library.chooseStarters")}
            </button>
            {selected && (
              <>
                <button
                  type="button"
                  className={buttonStyle("ghost", "sm")}
                  onClick={() => setEditing(selected)}
                >
                  {t("library.edit")}
                </button>
                <button
                  type="button"
                  disabled={busy}
                  className={buttonStyle("ghost", "sm")}
                  onClick={() =>
                    void act(async () => {
                      await useAppStore.getState().deleteFolder(selected.id);
                      useAppStore.getState().openLibrary();
                    })
                  }
                >
                  {t("library.delete")}
                </button>
              </>
            )}
            <button
              type="button"
              className={`${buttonStyle("ghost", "sm")} ml-auto`}
              aria-label={t("library.settingsTitle")}
              title={t("library.settingsTitle")}
              onClick={() => useAppStore.getState().openSettings("models")}
            >
              <SettingsLineIcon className="size-4" />
            </button>
          </div>
          <ReferenceImportPanel />
          {error && (
            <p role="alert" className={`${errorTextClass} mb-4`}>
              {error}
            </p>
          )}
          {editing && (
            <GroupForm
              key={editing === "new" ? "new" : editing.id}
              group={editing}
              busy={busy}
              act={act}
              close={closeForm}
            />
          )}
          {(starters || groups.length === 0) && !editing && snapshot && (
            <StarterGroups busy={busy} act={act} close={() => setStarters(false)} />
          )}
          {(inView.length > 0 || filtering) && (
            <LibraryFilterBar
              facets={facets}
              options={options}
              selection={selection}
              onToggle={(facetId, value) => {
                if (facetId === "tag") useAppStore.getState().toggleTagFilter(value);
                else setFacetFilter((current) => toggleFilterValue(current, facetId, value));
                setLimit(100);
              }}
              onClear={() => {
                setFacetFilter({});
                useAppStore.getState().setTagFilter([]);
                setLimit(100);
              }}
            />
          )}
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-[13px] text-ink-meta">
            <p role="status" data-testid="library-count">
              {!snapshot
                ? t("library.loading")
                : pending
                  ? t("library.progress", { count: pending })
                  : filtering
                    ? t("library.countFiltered", {
                        count: formatCount(shown.length, language),
                        total: formatCount(inView.length, language),
                      })
                    : failed
                      ? t("library.failed", { count: failed })
                      : waiting
                        ? t("library.waiting", { count: waiting })
                        : t(shown.length === 1 ? "library.count.one" : "library.count", {
                            count: shown.length,
                          })}
            </p>
            <span className="flex flex-wrap items-center gap-x-4 gap-y-1">
              <button
                type="button"
                data-testid="library-sort"
                aria-label={t("sort.bar.label", {
                  by: t(SORT_KEY_LABELS[sort.key]),
                  direction: t(sort.direction === "asc" ? "sort.ascending" : "sort.descending"),
                })}
                className="hover:text-ink hover:underline focus-visible:outline-2 focus-visible:outline-accent"
                onClick={(event) =>
                  useSortMenu.getState().show({
                    scope: sortScope,
                    withFolders: false,
                    anchor: event.currentTarget,
                  })
                }
              >
                {t("sort.bar", { by: sortLabel })}
              </button>
              <button
                type="button"
                data-testid="library-recently-deleted"
                className="hover:text-ink hover:underline focus-visible:outline-2 focus-visible:outline-accent"
                onClick={() => useAppStore.getState().openLibrary(LIBRARY_DELETED)}
              >
                {t("deleted.link")}
              </button>
              <button
                type="button"
                className="hover:text-ink hover:underline focus-visible:outline-2 focus-visible:outline-accent"
                onClick={() => useAppStore.getState().openTagsDialog()}
              >
                {t("library.manageTags")}
              </button>
            </span>
          </div>
          {shown.length === 0 ? (
            <p className="border-t border-rule py-10 text-ink-meta">
              {t(hasDocuments ? "library.noMatches" : "library.empty")}
            </p>
          ) : (
            <>
              <LibrarySelectionBar shownIds={shownIds} />
              <div className="library-columns library-column-labels border-y border-rule py-2 text-label font-semibold text-ink-meta">
                <button
                  type="button"
                  data-testid="library-sort-name"
                  title={t("sort.header", { by: t("sort.name") })}
                  aria-label={t("sort.header", { by: t("sort.name") })}
                  onClick={() => sortBy("name")}
                  className="w-fit text-left font-semibold hover:text-ink focus-visible:outline-2 focus-visible:outline-accent"
                >
                  {t("library.document")}
                  {sort.key === "name" && (sort.direction === "asc" ? " ↑" : " ↓")}
                </button>
                <span>{t("library.folder")}</span>
                <span>{t("tags.title")}</span>
              </div>
              {/* biome-ignore lint/a11y/useSemanticElements: a list of rows laid out as columns, not a table. */}
              <ul
                // biome-ignore lint/a11y/noNoninteractiveElementToInteractiveRole: the list is one grid whose rows select (aria-multiselectable).
                role="grid"
                aria-multiselectable="true"
                aria-label={t("library.list")}
                onKeyDown={(event) => {
                  if (!moveThroughRows(event)) extendThroughRows(event, "library");
                }}
              >
                {shown.slice(0, limit).map((doc) => (
                  <LibraryRow
                    key={doc.id}
                    doc={doc}
                    item={assignments.get(doc.id)}
                    groups={groups}
                    busy={busy}
                    act={act}
                  />
                ))}
              </ul>
            </>
          )}
          {shown.length > limit && (
            <button
              type="button"
              className={`${buttonStyle("secondary")} mt-4`}
              onClick={() => setLimit(limit + 100)}
            >
              {t("library.showMore")}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

/**
 * A Document's row in the Library: its checkbox, its name (which opens it in
 * the viewer) and where Organize is with it, its Folder to choose, and its
 * Tags. Memoised (#156): while Organize works through thousands of
 * Documents, only the rows whose Document or assignment changed are drawn again.
 */
const LibraryRow = memo(function LibraryRow({
  doc,
  item,
  groups,
  busy,
  act,
}: {
  doc: Document;
  item: DocumentGroupAssignment | undefined;
  groups: readonly LibraryGroup[];
  busy: boolean;
  act: Action;
}) {
  const t = useT();
  const place = useTabPlace();
  const name = displayName(doc);
  const fileName = fileNameOf(doc.path);
  const source = sourceOf(doc);
  /** "Cite as" open under the row, and the field to start in. */
  const [citing, setCiting] = useState<{ focus: ReferenceField | null } | null>(null);
  const citeToggle = useRef<HTMLButtonElement>(null);
  const citePanelId = useId();
  const objectMenu = useObjectMenu({ kind: "document", id: doc.id });
  const select = useSelectable({ kind: "document", id: doc.id }, "library", () =>
    useAppStore
      .getState()
      .openDocument(
        { documentId: doc.id },
        { as: "library", ...(place && { from: place.groupId }) },
      ),
  );
  // A double-click keeps the preview it opened, then renames.
  const rename = useRename<HTMLButtonElement>(`library-document:${doc.id}`, () =>
    keepPreviewOf(doc.id),
  );
  const renameDocument = useAppStore((state) => state.renameDocument);
  const status =
    item?.status === "classifying" || item?.status === "pending"
      ? "library.working"
      : item?.status === "waiting" || item?.status === "failed"
        ? "library.needsAttention"
        : item?.source === "user"
          ? "library.manual"
          : item?.status === "classified"
            ? "library.saved"
            : "library.notClassified";
  return (
    // biome-ignore lint/a11y/useSemanticElements: the row is laid out as columns, not as a table row.
    // biome-ignore lint/a11y/useFocusableInteractive: the name's button takes the focus, and the arrow keys move between them.
    <li
      data-testid="library-document"
      data-document-id={doc.id}
      // biome-ignore lint/a11y/noNoninteractiveElementToInteractiveRole: a row of the grid, which says whether it is selected.
      role="row"
      {...select.rowProps}
      {...objectMenu.rowProps}
      className={`group/row library-columns relative border-b border-rule py-3 ${
        select.selected ? "bg-accent-wash" : ""
      }`}
    >
      <LibrarySelectBox document={doc} />
      <div className="library-document-name min-w-0">
        <div className="flex min-w-0 items-center gap-2">
          {rename.renaming ? (
            <InlineName
              value={doc.name}
              label={t("documents.renameLabel", { name: doc.name })}
              {...rename.field((next) => renameDocument(doc.id, next))}
            />
          ) : (
            <button
              {...rename.nameProps}
              type="button"
              {...TREE_ROW}
              data-row-main=""
              title={name === source ? name : `${name}\n${source}`}
              className="flex min-w-0 items-center gap-2 text-left hover:underline focus-visible:outline-2 focus-visible:outline-accent"
              // A click opens a preview to the Library's right; ⌘-click and Shift-click select (`useSelectable`).
              onClick={select.onClick}
            >
              <DocumentLineIcon kind={doc.kind} className="size-4 shrink-0 text-ink-meta" />
              <span data-testid="library-document-name" className="truncate font-semibold">
                {name}
              </span>
            </button>
          )}
          <ObjectMenu
            target={{ kind: "document", id: doc.id }}
            label={t("documents.more", { name: doc.name })}
            here={{ "document.rename": rename.start }}
            className={`${rowActionButtonClass} opacity-0 group-hover/row:opacity-100 group-has-[:focus-visible]/row:opacity-100 aria-expanded:opacity-100`}
          />
          {doc.reference && (
            <CiteAsToggle
              ref={citeToggle}
              document={doc}
              record={doc.reference}
              panelId={citePanelId}
              open={citing !== null}
              onToggle={(focus) => setCiting(citing === null ? { focus } : null)}
            />
          )}
        </div>
        <details className="mt-1 text-[12px] text-ink-meta">
          <summary className="w-fit cursor-pointer">{t(status)}</summary>
          <p className="mt-1 break-words">
            {doc.web ? (
              t("library.webAddress", { address: doc.web.address })
            ) : (
              <>
                {t("library.file", { name: fileName })} · {doc.kind.toUpperCase()} ·{" "}
                {t("library.originalsStay")}
              </>
            )}
          </p>
          {item?.model && (
            <p>
              {item.model.id} · {t(item.model.images ? "library.usedPages" : "library.usedText")}
              {item.model.reason === "slow"
                ? ` · ${t("library.route.slow")}`
                : item.model.reason === "memory"
                  ? ` · ${t("library.route.memory")}`
                  : item.model.reason === "unavailable"
                    ? ` · ${t("library.route.unavailable")}`
                    : ""}
            </p>
          )}
          {item?.error && <p className={`${errorTextClass} mt-1`}>{item.error.message}</p>}
        </details>
      </div>
      <select
        aria-label={t("library.groupFor", { name })}
        disabled={busy}
        value={item?.groupId ?? ""}
        className={`${inputClass} mt-0 min-w-0`}
        onChange={(event) =>
          void act(() =>
            useAppStore
              .getState()
              .moveToFolder({ documentIds: [doc.id] }, event.target.value || null),
          )
        }
      >
        <option value="">{t("library.unsorted")}</option>
        {groups.map((group) => (
          <option key={group.id} value={group.id}>
            {group.name}
          </option>
        ))}
      </select>
      <DocumentTagChips document={doc} />
      {citing && doc.reference && (
        <div id={citePanelId} className="col-span-full min-w-0">
          <CiteAsPanel
            document={doc}
            record={doc.reference}
            focusField={citing.focus}
            onClose={() => {
              setCiting(null);
              citeToggle.current?.focus();
            }}
          />
        </div>
      )}
    </li>
  );
});

/**
 * A Library row's way into "Cite as" (#286), after its name: "Needs details",
 * with the waiting dot, while the record is too thin to cite, which opens it
 * at the first missing field; otherwise "Cite as", shown while the row is
 * pointed at or focused.
 */
function CiteAsToggle({
  ref,
  document,
  record,
  panelId,
  open,
  onToggle,
}: {
  ref: Ref<HTMLButtonElement>;
  document: Document;
  record: ReferenceRecord;
  panelId: string;
  open: boolean;
  onToggle(focus: ReferenceField | null): void;
}) {
  const t = useT();
  const language = useLanguage();
  const missing = missingDetails(record);
  const list = missingList(language, record);
  const label = t("reference.citeAs.label", { name: displayName(document) });
  return (
    <button
      ref={ref}
      type="button"
      data-testid="cite-as-toggle"
      data-needs-details={missing.length > 0 ? "" : undefined}
      aria-expanded={open}
      aria-controls={open ? panelId : undefined}
      aria-label={
        list ? `${label}. ${t("reference.needsDetails.title", { missing: list })}` : label
      }
      title={list ? t("reference.needsDetails.title", { missing: list }) : undefined}
      onClick={() => onToggle(missing[0] ?? null)}
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-sm text-[12px] leading-5 whitespace-nowrap text-ink-meta hover:text-ink hover:underline ${
        missing.length > 0 || open
          ? ""
          : "opacity-0 group-focus-within/row:opacity-100 group-hover/row:opacity-100"
      }`}
    >
      {missing.length > 0 && (
        <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-attention" />
      )}
      {t(missing.length > 0 ? "reference.needsDetails" : "reference.citeAs")}
    </button>
  );
}

/**
 * From the sheet into writing: "Ask about this Folder" adds a Question that
 * searches only the Folder to the most recent Mind, and "Start a Mind from
 * this Folder" makes a Mind named after it, starting with that Question.
 * With the list narrowed, they say how many Documents the Question searches:
 * exactly those shown.
 */
function WritingBridges({ bridge }: { bridge: LibraryBridge }) {
  const t = useT();
  const language = useLanguage();
  const count = formatCount(bridge.count, language);
  const [ask, start, askHint, startHint] =
    bridge.kind === "folder"
      ? [
          t("library.ask.folder"),
          t("library.start.folder"),
          t("library.ask.folderHint"),
          t("library.start.folderHint"),
        ]
      : [
          t(bridge.count === 1 ? "library.ask.document" : "library.ask.documents", { count }),
          t(bridge.count === 1 ? "library.start.document" : "library.start.documents", { count }),
          t(bridge.count === 1 ? "library.ask.documentHint" : "library.ask.documentsHint", {
            count,
          }),
          t(bridge.count === 1 ? "library.start.documentHint" : "library.start.documentsHint", {
            count,
          }),
        ];
  return (
    <>
      <button
        type="button"
        data-testid="library-ask"
        data-scope={bridge.kind}
        title={askHint}
        className={buttonStyle("secondary")}
        onClick={() => void useAppStore.getState().askAbout(bridge)}
      >
        {ask}
      </button>
      <button
        type="button"
        data-testid="library-start-mind"
        data-scope={bridge.kind}
        title={startHint}
        className={buttonStyle("secondary")}
        onClick={() => void useAppStore.getState().startMindFrom(bridge)}
      >
        {start}
      </button>
    </>
  );
}

function GroupForm({
  group,
  busy,
  act,
  close,
}: {
  group: LibraryGroup | "new";
  busy: boolean;
  act: Action;
  close(): void;
}) {
  const t = useT();
  const nameInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    nameInput.current?.focus();
  }, []);
  const [name, setName] = useState(group === "new" ? "" : group.name);
  const [description, setDescription] = useState(group === "new" ? "" : group.description);
  const [countsOwnFiles, setCountsOwnFiles] = useState(
    group === "new" ? false : group.countsOwnFiles,
  );
  // Settings counts IncarnaMind's files everywhere: there's nothing for a folder to switch.
  const leftOut = useAppStore((state) => state.settings?.user.leaveOutOwnFiles ?? true);
  return (
    <form
      className="mb-5 max-w-xl border-y border-rule py-4"
      onSubmit={(event) => {
        event.preventDefault();
        void act(async () => {
          const input = { name, description, countsOwnFiles };
          if (group === "new") await core.createLibraryGroup(input);
          else await useAppStore.getState().editFolder(group.id, input);
          close();
        });
      }}
    >
      <label className={fieldLabelClass}>
        {t("library.name")}
        <input
          ref={nameInput}
          required
          maxLength={100}
          value={name}
          onChange={(event) => setName(event.target.value)}
          className={inputClass}
        />
      </label>
      <label className={`${fieldLabelClass} mt-3`}>
        {t("library.description")}
        <textarea
          rows={2}
          maxLength={500}
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder={t("library.descriptionHint")}
          className={inputClass}
        />
      </label>
      {/* Checks leave the files IncarnaMind wrote out of their evidence; a folder can count them (#80). */}
      <label className="mt-3 flex cursor-pointer items-start gap-3">
        <input
          type="checkbox"
          data-testid="folder-counts-own-files"
          checked={leftOut ? countsOwnFiles : true}
          disabled={!leftOut}
          onChange={(event) => setCountsOwnFiles(event.target.checked)}
          className="mt-1 size-4 shrink-0 accent-ink"
        />
        <span>
          <span className="text-ink">{t("library.countsOwnFiles")}</span>
          <span className="block text-[13px] text-ink-meta">
            {t(leftOut ? "library.countsOwnFiles.hint" : "library.countsOwnFiles.everywhere")}
          </span>
        </span>
      </label>
      <div className="mt-3 flex gap-2">
        <button type="submit" disabled={busy || !name.trim()} className={buttonStyle("primary")}>
          {t("library.save")}
        </button>
        <button type="button" className={buttonStyle("ghost")} onClick={close}>
          {t("library.cancel")}
        </button>
      </div>
    </form>
  );
}

function StarterGroups({ busy, act, close }: { busy: boolean; act: Action; close(): void }) {
  const t = useT();
  const [chosen, setChosen] = useState<string[]>([...libraryPresetKeys]);
  return (
    <section
      aria-label={t("library.chooseStarters")}
      className="mb-5 max-w-xl border-y border-rule py-4"
    >
      <h3 className="mb-1 font-semibold">{t("library.chooseStarters")}</h3>
      <p className="mb-3 text-[13px] text-ink-meta">{t("library.startersHint")}</p>
      {libraryPresetKeys.map((key) => (
        <label key={key} className="flex cursor-pointer items-start gap-3 py-2">
          <input
            type="checkbox"
            checked={chosen.includes(key)}
            onChange={(event) =>
              setChosen(
                event.target.checked ? [...chosen, key] : chosen.filter((item) => item !== key),
              )
            }
            className="mt-1 size-4 accent-ink"
          />
          <span>
            <span className="font-semibold">{t(`library.preset.${key}`)}</span>
            <span className="block text-[13px] text-ink-meta">
              {t(`library.preset.${key}.description`)}
            </span>
          </span>
        </label>
      ))}
      <button
        type="button"
        disabled={busy || chosen.length === 0}
        className={`${buttonStyle("primary")} mt-3`}
        onClick={() =>
          void act(async () => {
            await core.addLibraryStarterGroups(chosen);
            close();
          })
        }
      >
        {t("library.addSelected")}
      </button>
    </section>
  );
}
