import {
  type ChangeEvent,
  memo,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { Document, DocumentStatus, Tag } from "../../../core/api";
import { displayName, sourceOf } from "../../../shared/documentNames";
import type { MessageKey } from "../../../shared/i18n";
import { startDocumentDrag } from "../documentDrag";
import { documentStatusText } from "../documentStatus";
import { useLanguage, useT } from "../i18n";
import { currentTabOf } from "../layoutSlice";
import { fileStatusLabel, formatCount } from "../linkedFolders";
import { useRequests } from "../requests";
import { scanningStatus } from "../scanStatus";
import {
  extendThroughRows,
  rowsIn,
  selectedOn,
  useSelectable,
  useSelectionOrder,
} from "../selection";
import { useSortMenu } from "../sortMenu";
import { selectVisibleDocuments, useAppStore } from "../store";
import { chipsOf, tagIndex } from "../tagEditing";
import { moveThroughRows, TREE_ROW } from "../treeKeys";
import { useRename } from "../useRename";
import { ActiveTagFilter, DocumentTagMenu, TagFilterMenu } from "./DocumentTags";
import { FolderTree } from "./FolderTree";
import { InlineName } from "./InlineName";
import { LibraryFolders } from "./LibraryFolders";
import { LinkFolderDialog } from "./LinkFolderDialog";
import { DocumentLineIcon, MoreLineIcon } from "./lineIcons";
import { ObjectMenu, useObjectMenu } from "./ObjectMenu";
import { SidebarSearch } from "./SidebarSearch";
import { SidebarTags } from "./SidebarTags";
import {
  clearsActions,
  rowActionButtonClass,
  rowActionsClass,
  rowButtonClass,
  rowClass,
  rowIconClass,
  rowPadding,
} from "./sidebarRows";
import { TagMarks, TagNamesTip, tagNamesTipListProps } from "./TagMarks";

/** What the file picker offers. The core decides what it takes. */
const ACCEPTED_FILES = ".pdf,.docx,.pptx,.xlsx,.csv,.txt,.md,.markdown";

/** A Document's status as its row's end shows it: nothing once it's ready. */
const shortStatusMessages: Record<DocumentStatus, MessageKey | null> = {
  queued: "documents.statusShort.queued",
  extracting: "documents.statusShort.extracting",
  "waiting-for-model": "documents.statusShort.waiting",
  embedding: "documents.statusShort.embedding",
  ready: null,
  failed: "documents.statusShort.failed",
  "no-text": "documents.statusShort.noText",
};

/** The ways to browse the Documents, in the switcher's order. */
type BrowseView = "folders" | "tags" | "sources";

const browseLabels: Record<BrowseView, MessageKey> = {
  folders: "library.groups",
  tags: "tags.title",
  sources: "library.sources",
};

/** Still being processed: the row is muted until it's done. */
const isProcessing = (status: DocumentStatus) =>
  status === "queued" ||
  status === "extracting" ||
  status === "waiting-for-model" ||
  status === "embedding";

/**
 * The sidebar's tree: "All Documents", then a switcher, "Folders | Tags |
 * Source locations" (with the Tag filter and "+" for a new Folder), over one
 * of three views. Folders, the first, is the User's projects: each Folder
 * with its Minds and Documents, then Not in a Folder (see `LibraryFolders`).
 * Tags lists the Tags (see `SidebarTags`); Source locations, each Linked
 * folder with its folders as on disk, then "Other Documents", the files
 * added on their own (see `FolderTree`). With nothing to browse but Minds,
 * there is no switcher. Each Document is one row: its name, its Tags' colours (see
 * `TagMarks`) and, while it's processed, if it failed, or if its file is
 * missing or can't be reached, its status at the end. Its Tags are in its
 * Tags menu, and the Tags dialog; its "More" menu renames it, opens its file
 * or shows it in its folder, and deletes it. Clicking a Document opens it in
 * the viewer. Documents and Linked folders are added from the header's "+"
 * menu, or dropped anywhere on the window (see `FileDrop`); with none yet,
 * the tree says so.
 */
export function DocumentsSection() {
  const t = useT();
  const language = useLanguage();
  const hasTags = useAppStore((state) => state.tags.length > 0);
  const documentCount = useAppStore((state) => state.documents.length);
  const hasAnyDocuments = documentCount > 0;
  const [chosenView, setView] = useState<BrowseView>("folders");
  const libraryOpen = useAppStore((state) => currentTabOf(state.layout)?.item.kind === "library");
  // The same array until the Documents or the Tag filter change (see `selectVisibleDocuments`).
  const documents = useAppStore(selectVisibleDocuments);
  const filtering = useAppStore((state) => state.tagFilter.length > 0);
  const hasFolders = useAppStore(
    (state) => state.folders.length > 0 || state.linkedFolders.length > 0,
  );
  const addDocuments = useAppStore((state) => state.addDocuments);
  const picker = useRef<HTMLInputElement>(null);
  const tree = useRef<HTMLDivElement>(null);
  // ⌘A and Shift-click reach the rows in view: every Mind and Document the tree draws.
  useSelectionOrder("sidebar", () => rowsIn(tree.current));
  const tags = useAppStore((state) => state.tags);
  // Built once per change to the Tags, so each row only looks its own up.
  const tagsById = useMemo(() => tagIndex(tags), [tags]);

  // Folders always: the Minds are there. Tags and the tree on disk once there is something to
  // browse (a Document or a Linked folder, so the first Document moves nothing).
  const somethingToBrowse = hasAnyDocuments || hasFolders;
  const views: BrowseView[] = [
    "folders",
    ...(hasTags && somethingToBrowse ? (["tags"] as const) : []),
    ...(somethingToBrowse ? (["sources"] as const) : []),
  ];
  const view = views.includes(chosenView) ? chosenView : "folders";
  // A Folder just made is named in its row, which only the Folders view has.
  const naming = useRequests((state) => state.nameFolder !== null);
  useEffect(() => {
    if (naming) setView("folders");
  }, [naming]);
  /**
   * Each Document's row as last made, by depth, while the Tags stay the same
   * (#156). A list of thousands drawn again because one Document changed
   * hands React the same rows for the rest, which it passes over without
   * comparing them; only the row whose Document changed is made and drawn again.
   */
  const renderDocument = useMemo(() => {
    const made = new WeakMap<Document, Map<number, ReactNode>>();
    return (item: Document, depth: number): ReactNode => {
      let atDepth = made.get(item);
      if (!atDepth) {
        atDepth = new Map();
        made.set(item, atDepth);
      }
      let row = atDepth.get(depth);
      if (row === undefined) {
        row = <DocumentRow key={item.id} item={item} depth={depth} tagsById={tagsById} />;
        atDepth.set(depth, row);
      }
      return row;
    };
  }, [tagsById]);

  const addPicked = (event: ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(event.target.files ?? []);
    event.target.value = ""; // so picking the same file again still counts as a change
    if (picked.length > 0) void addDocuments(picked);
  };

  // A Linked folder shows even before its first Document, unless a filter hides it.
  const empty = documents.length === 0 && (filtering || !hasFolders);

  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: ↑ and ↓ move between the rows' buttons inside it.
    <div
      data-testid="documents-section"
      {...tagNamesTipListProps}
      onKeyDown={(event) => {
        if (!moveThroughRows(event)) extendThroughRows(event, "sidebar");
      }}
    >
      {/* Files given straight to the page arrive here, as from the dialog: the smoke tests add them so. */}
      <input
        ref={picker}
        type="file"
        multiple
        accept={ACCEPTED_FILES}
        data-testid="add-documents-input"
        className="hidden"
        onChange={addPicked}
      />
      <SidebarSearch />
      <div className={rowClass(libraryOpen)}>
        <button
          type="button"
          {...TREE_ROW}
          data-testid="open-library"
          // ⌘-click (Ctrl-click) opens it to the side, as a Mind's row does.
          onClick={(event) =>
            useAppStore.getState().openLibrary(undefined, { side: event.metaKey || event.ctrlKey })
          }
          className={`${rowButtonClass} ${clearsActions.menu}`}
        >
          <DocumentLineIcon kind="text" className="size-4 shrink-0" />
          <span className="min-w-0 flex-1 truncate">{t("library.title")}</span>
          <span data-testid="library-row-count" className="text-[12px] tabular-nums text-ink-meta">
            {formatCount(documentCount, language)}
          </span>
        </button>
        <div className={rowActionsClass}>
          <button
            type="button"
            data-testid="sidebar-sort"
            aria-label={t("sort.sidebar")}
            title={t("sort.sidebar")}
            onClick={(event) =>
              useSortMenu
                .getState()
                .show({ scope: "sidebar", withFolders: true, anchor: event.currentTarget })
            }
            className={rowActionButtonClass}
          >
            <MoreLineIcon className="size-4" />
          </button>
        </div>
      </div>
      {views.length > 1 && (
        <div className="mx-2 my-2 flex items-center gap-2 border-b border-rule pb-1">
          <fieldset
            className="flex min-w-0 items-center gap-3 text-[12px] text-ink-meta"
            data-testid="browse-views"
            aria-label={t("library.browse")}
          >
            {views.map((each) => (
              <button
                key={each}
                type="button"
                aria-pressed={view === each}
                title={t(browseLabels[each])}
                className={`min-w-0 truncate ${view === each ? "font-semibold text-ink" : "hover:text-ink"}`}
                onClick={() => setView(each)}
              >
                {t(browseLabels[each])}
              </button>
            ))}
          </fieldset>
          <span className="ml-auto flex shrink-0 items-center">
            <TagFilterMenu />
          </span>
        </div>
      )}
      {/* Under the switcher, so it never moves; in the Tags view, the chosen Tags say it. */}
      {view !== "tags" && <ActiveTagFilter />}
      <div ref={tree} role="tree" aria-multiselectable="true" aria-label={t("sidebar.tree")}>
        {view === "folders" ? (
          <LibraryFolders documents={documents} renderDocument={renderDocument} />
        ) : view === "tags" ? (
          <SidebarTags renderDocument={renderDocument} />
        ) : (
          <FolderTree documents={documents} filtering={filtering} renderDocument={renderDocument} />
        )}
      </div>
      {/* The Tags view lists every Tag, whatever the filter. */}
      {empty &&
        view !== "tags" &&
        (filtering ? (
          <p className="px-2 py-1 text-[13px] leading-5 text-ink-meta">{t("tags.filter.empty")}</p>
        ) : (
          // Adding is the header's "+" menu's: this only says where.
          <p
            data-testid="documents-empty"
            className="px-2 pt-2 pb-1 text-[13px] leading-5 text-ink-meta"
          >
            {t("documents.none")}
          </p>
        ))}
      <LinkFolderDialog />
      <TagNamesTip />
    </div>
  );
}

/** The Documents a drag from this row takes: all those selected, if it is one of them. */
function draggedWith(documentId: string): string[] {
  const chosen = selectedOn("sidebar").flatMap((each) =>
    each.kind === "document" ? [each.id] : [],
  );
  return chosen.includes(documentId) ? chosen : [documentId];
}

/**
 * A Document's row: its icon and name, its Tags' colours right after the
 * name (their names in its tooltip and description), and its status at the
 * end while it's processed, if it failed, or if its file is missing or can't
 * be reached (then muted too; clicking still opens what IncarnaMind kept of
 * it). The name gives way first; the row stays one 28px line. Pointed at, it
 * offers its Tags and a menu for the rest. It is in the Folder its file is
 * in, so there is no moving it here. Memoised: a sidebar of thousands of
 * Documents draws again only the rows whose Document changed.
 */
const DocumentRow = memo(function DocumentRow({
  item,
  depth,
  tagsById,
}: {
  item: Document;
  depth: number;
  tagsById: ReadonlyMap<string, Tag>;
}) {
  const t = useT();
  const rename = useRename<HTMLButtonElement>(`sidebar-document:${item.id}`);
  const renameDocument = useAppStore((state) => state.renameDocument);
  const shownName = displayName(item);
  const source = sourceOf(item);
  const chips = chipsOf(item.tags, tagsById);
  const tagNames =
    chips.length > 0
      ? t("tags.marks", { names: chips.map(({ tag }) => tag.name).join(", ") })
      : undefined;
  const openDocument = useAppStore((state) => state.openDocument);
  const isOpen = useAppStore((state) => {
    const shown = currentTabOf(state.layout)?.item;
    return shown?.kind === "document" && shown.documentId === item.id;
  });
  const objectMenu = useObjectMenu({ kind: "document", id: item.id });
  const select = useSelectable({ kind: "document", id: item.id }, "sidebar", () =>
    openDocument({ documentId: item.id }),
  );
  const lit = isOpen || select.selected;
  const muted = !lit && (isProcessing(item.status) || item.fileStatus !== "available");

  return (
    // biome-ignore lint/a11y/useFocusableInteractive: the row's own button takes the focus, and the arrow keys move between them.
    <li
      data-testid="document-list-item"
      data-document-id={item.id}
      data-folder-id={item.folderId ?? ""}
      data-depth={depth}
      data-status={item.status}
      data-file-status={item.fileStatus}
      data-tagging={item.tagging}
      role="treeitem"
      {...select.rowProps}
      {...(rename.renaming ? {} : objectMenu.rowProps)}
      className={rowClass(lit, muted ? "muted" : "item")}
    >
      {rename.renaming ? (
        <div
          className="flex h-full w-full min-w-0 items-center gap-2 pr-1"
          style={rowPadding(depth)}
        >
          <DocumentLineIcon kind={item.kind} className={rowIconClass(false)} />
          <InlineName
            value={item.name}
            label={t("documents.renameLabel", { name: item.name })}
            {...rename.field((name) => renameDocument(item.id, name))}
          />
        </div>
      ) : (
        // Opens the Document in the viewer, replacing whatever it showed.
        <button
          {...rename.nameProps}
          type="button"
          {...TREE_ROW}
          data-testid="open-document"
          aria-current={isOpen ? "true" : undefined}
          aria-description={tagNames}
          title={[shownName === source ? shownName : `${shownName}\n${source}`, tagNames]
            .filter(Boolean)
            .join("\n")}
          // A click opens it, in a preview beside the Mind or in the group already showing a Document;
          // ⌘-click and Shift-click select (`useSelectable`).
          onClick={select.onClick}
          draggable
          onDragStart={(event) => startDocumentDrag(event, draggedWith(item.id))}
          className={`${rowButtonClass} ${clearsActions.tagsAndMenu}`}
          style={rowPadding(depth)}
        >
          <DocumentLineIcon kind={item.kind} className={rowIconClass(lit)} />
          {/* Takes what the status leaves; in it, the name gives way before the colours. */}
          <span className="flex min-w-0 flex-1 items-center gap-2">
            <span data-testid="row-text" className="min-w-0 truncate">
              {shownName}
            </span>
            <TagMarks chips={chips} />
          </span>
          <DocumentStatusLabel item={item} />
        </button>
      )}
      {!rename.renaming && (
        <div className={rowActionsClass}>
          <DocumentTagMenu item={item} buttonClassName={rowActionButtonClass} />
          <ObjectMenu
            target={{ kind: "document", id: item.id }}
            label={t("documents.more", { name: item.name })}
            here={{ "document.rename": rename.start }}
          />
        </div>
      )}
    </li>
  );
});

/**
 * The status at a row's end: "Missing" or "Unavailable" when its file isn't
 * where it was (that comes first: it's what the User can act on), "38%"
 * while it's embedded, "Failed" if it failed, nothing once it's ready. Its
 * full words (and a failure's message) are its tooltip, and what a screen
 * reader reads.
 */
export function DocumentStatusLabel({ item }: { item: Document }) {
  const t = useT();
  const language = useLanguage();
  const fileState = fileStatusLabel(item.fileStatus);
  if (fileState) {
    return (
      <span
        data-testid="document-status"
        data-file-status={item.fileStatus}
        title={t(fileState.full)}
        className="min-w-0 shrink truncate text-label font-semibold text-ink-meta"
      >
        <span aria-hidden="true">{t(fileState.short)}</span>
        <span className="sr-only">{t(fileState.full)}</span>
      </span>
    );
  }
  // Its scanned pages being read, or some that couldn't be (#397).
  const scanning = item.status === "extracting" && item.scan ? scanningStatus(item.scan, t) : null;
  const unread = item.status === "failed" && item.failure?.reason === "pages-unread";
  const full = documentStatusText(item, t, language, scanning?.full ?? null);
  const percent = Math.floor((item.progress ?? 0) * 100);
  const shortKey = unread ? "documents.statusShort.pagesUnread" : shortStatusMessages[item.status];
  const short = shortKey && (scanning?.short ?? t(shortKey, { percent }));
  const title = item.failure?.message ? `${full}\n${item.failure.message}` : full;
  if (!short) {
    return (
      <span data-testid="document-status" className="sr-only">
        {full}
      </span>
    );
  }
  return (
    <span
      data-testid="document-status"
      title={title}
      className="flex shrink-0 items-center gap-1.5 text-label font-semibold text-ink-meta"
    >
      {item.status === "failed" && !unread && (
        <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-attention" />
      )}
      <span aria-hidden="true">{short}</span>
      <span className="sr-only">{full}</span>
    </span>
  );
}
