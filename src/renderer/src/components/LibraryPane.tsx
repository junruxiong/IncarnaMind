import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type { Document, DocumentGroupAssignment, LibraryGroup } from "../../../core/api";
import { displayName, fileNameOf } from "../../../shared/documentNames";
import { libraryPresetKeys } from "../../../shared/libraryPresets";
import { core } from "../core";
import { errorMessage as messageOf } from "../errors";
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
import { useAppStore } from "../store";
import { LibraryFilterBar } from "./LibraryFilterBar";
import { LibrarySelectBox, LibrarySelectionBar } from "./LibrarySelection";
import { DocumentLineIcon, SettingsLineIcon } from "./lineIcons";
import { DocumentTagChips } from "./TagChips";
import { buttonStyle, errorTextClass, fieldLabelClass, inputClass } from "./ui";

type Action = (work: () => Promise<unknown>) => Promise<boolean>;

const NO_GROUPS: readonly LibraryGroup[] = [];

/** The sidebar chooses a folder; this sheet shows its documents and editable tags. */
export function LibraryPane() {
  const t = useT();
  const language = useLanguage();
  // Every Document: the Tags filter below is the sidebar's, and counts its options from all.
  const documents = useAppStore((state) => state.documents);
  const hasDocuments = useAppStore((state) => state.documents.length > 0);
  const snapshot = useAppStore((state) => state.library);
  const filter = useAppStore((state) => state.libraryFilter);
  const tags = useAppStore((state) => state.tags);
  const tagFilter = useAppStore(useShallow((state) => state.tagFilter));
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
  const [deleting, setDeleting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setLimit(100);
    setSearch("");
    setFacetFilter({});
    setDeleting(false);
    setEditing(filter === "new" ? "new" : null);
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
  const { shown, options } = useMemo(
    () => filterLibrary<Document>(inView, facets, selection),
    [inView, facets, selection],
  );
  const shownIds = useMemo(() => shown.map((doc) => doc.id), [shown]);
  // A row's Shift-click reads the rows in view when clicked, so a change to them redraws no row.
  const shownRef = useRef(shownIds);
  useLayoutEffect(() => {
    shownRef.current = shownIds;
  });
  const shownNow = useCallback(() => shownRef.current, []);
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
  const heading = selected?.name ?? t(filter === "unsorted" ? "library.unsorted" : "library.all");
  const closeForm = () => {
    setEditing(null);
    if (filter === "new") useAppStore.getState().openLibrary();
  };
  return (
    <section
      data-testid="library"
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
            <button
              type="button"
              className={buttonStyle("secondary")}
              onClick={() => void useAppStore.getState().pickDocuments()}
            >
              {t("documents.add")}
            </button>
          </div>
          {/* The name takes the whole width and wraps by word, two lines at most (#213); the actions sit below it, never beside. */}
          <div className="mb-2">
            <h1
              data-testid="library-heading"
              title={heading}
              className="line-clamp-2 font-serif text-heading font-semibold [overflow-wrap:break-word]"
            >
              {heading}
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
                  className={buttonStyle("ghost", "sm")}
                  onClick={() => setDeleting(!deleting)}
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
          {error && (
            <p role="alert" className={`${errorTextClass} mb-4`}>
              {error}
            </p>
          )}
          {deleting && selected && (
            <div className="mb-5 border-y border-rule py-4">
              <p className="mb-3 text-ink-secondary">{t("library.deleteNotice")}</p>
              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={busy}
                  className={buttonStyle("danger")}
                  onClick={() =>
                    void act(async () => {
                      await core.deleteLibraryGroup(selected.id);
                      useAppStore.getState().openLibrary();
                    })
                  }
                >
                  {t("library.delete")}
                </button>
                <button
                  type="button"
                  className={buttonStyle("ghost")}
                  onClick={() => setDeleting(false)}
                >
                  {t("library.cancel")}
                </button>
              </div>
            </div>
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
            <button
              type="button"
              className="hover:text-ink hover:underline focus-visible:outline-2 focus-visible:outline-accent"
              onClick={() => useAppStore.getState().openTagsDialog()}
            >
              {t("library.manageTags")}
            </button>
          </div>
          {shown.length === 0 ? (
            <p className="border-t border-rule py-10 text-ink-meta">
              {t(hasDocuments ? "library.noMatches" : "library.empty")}
            </p>
          ) : (
            <>
              <LibrarySelectionBar shownIds={shownIds} />
              <div
                aria-hidden="true"
                className="library-columns library-column-labels border-y border-rule py-2 text-label font-semibold text-ink-meta"
              >
                <span>{t("library.document")}</span>
                <span>{t("library.folder")}</span>
                <span>{t("tags.title")}</span>
              </div>
              <ul>
                {shown.slice(0, limit).map((doc) => (
                  <LibraryRow
                    key={doc.id}
                    doc={doc}
                    item={assignments.get(doc.id)}
                    groups={groups}
                    busy={busy}
                    act={act}
                    shownIds={shownNow}
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
  shownIds,
}: {
  doc: Document;
  item: DocumentGroupAssignment | undefined;
  groups: readonly LibraryGroup[];
  busy: boolean;
  act: Action;
  /** The rows in view, in order, read when a Shift-click selects a range. */
  shownIds(): readonly string[];
}) {
  const t = useT();
  const name = displayName(doc);
  const fileName = fileNameOf(doc.path);
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
    <li
      data-testid="library-document"
      data-document-id={doc.id}
      className="group/row library-columns relative border-b border-rule py-3"
    >
      <LibrarySelectBox document={doc} shownIds={shownIds} />
      <div className="library-document-name min-w-0">
        <button
          type="button"
          title={name === fileName ? name : `${name}\n${fileName}`}
          className="flex max-w-full items-center gap-2 text-left hover:underline focus-visible:outline-2 focus-visible:outline-accent"
          onClick={() => useAppStore.getState().openDocument({ documentId: doc.id })}
        >
          <DocumentLineIcon kind={doc.kind} className="size-4 shrink-0 text-ink-meta" />
          <span data-testid="library-document-name" className="truncate font-semibold">
            {name}
          </span>
        </button>
        <details className="mt-1 text-[12px] text-ink-meta">
          <summary className="w-fit cursor-pointer">{t(status)}</summary>
          <p className="mt-1 break-words">
            {t("library.file", { name: fileName })} · {doc.kind.toUpperCase()} ·{" "}
            {t("library.originalsStay")}
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
          void act(() => core.assignDocumentGroup(doc.id, event.target.value || null))
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
    </li>
  );
});

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
  return (
    <form
      className="mb-5 max-w-xl border-y border-rule py-4"
      onSubmit={(event) => {
        event.preventDefault();
        void act(async () => {
          const input = { name, description };
          if (group === "new") await core.createLibraryGroup(input);
          else await core.updateLibraryGroup(group.id, input);
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
