import { type KeyboardEvent, type ReactNode, useEffect, useMemo } from "react";
import { create } from "zustand";
import type { Document, LibraryGroup, Mind } from "../../../core/api";
import { useFolderDrop } from "../folderDrag";
import { useT } from "../i18n";
import { currentMindIdOf, currentTabOf } from "../layoutSlice";
import { useAppStore } from "../store";
import { TREE_ROW } from "../treeKeys";
import { useRename } from "../useRename";
import { InlineName } from "./InlineName";
import { ChevronDownLineIcon, ChevronRightLineIcon, FolderLineIcon } from "./lineIcons";
import { MindRow } from "./MindRow";
import { ObjectMenu, useObjectMenu } from "./ObjectMenu";
import { rowActionsClass, rowButtonClass, rowClass, rowIconClass, rowPadding } from "./sidebarRows";

/** How many Documents a Folder lists at first; "Show N more Documents" lists more, in place. */
export const FIRST_DOCUMENTS = 5;
/** How many more each "Show more" lists. */
const MORE_DOCUMENTS = 50;

/** Not in a Folder's key, among the folds and the drop targets. */
const NOT_IN_A_FOLDER = "not-in-a-folder";

const NO_DOCUMENTS: readonly Document[] = [];
const NO_MINDS: readonly Mind[] = [];

const FOLDS_KEY = "incarnamind.sidebar.folders";

interface Folds {
  /** The Folders shown open; the rest are closed. Not in a Folder is open unless in here. */
  open: ReadonlySet<string>;
  /** Not in a Folder folded away. */
  looseFolded: boolean;
  /** How many Documents each lists, past the first few, by key. Forgotten at a restart. */
  shown: Readonly<Record<string, number>>;
  toggle(key: string): void;
  /** Opens a Folder (e.g. the open Mind's), if it isn't. */
  reveal(folderId: string): void;
  showMore(key: string, count: number): void;
}

function readFolds(): { open: string[]; looseFolded: boolean } {
  try {
    const value = JSON.parse(localStorage.getItem(FOLDS_KEY) ?? "null") as {
      open?: unknown;
      looseFolded?: unknown;
    } | null;
    return {
      open: Array.isArray(value?.open) ? value.open.filter((id) => typeof id === "string") : [],
      looseFolded: value?.looseFolded === true,
    };
  } catch {
    return { open: [], looseFolded: false };
  }
}

function saveFolds(folds: Pick<Folds, "open" | "looseFolded">): void {
  try {
    localStorage.setItem(
      FOLDS_KEY,
      JSON.stringify({ open: [...folds.open], looseFolded: folds.looseFolded }),
    );
  } catch {
    // Not remembered, then: they start closed next time.
  }
}

/** Which Folders are open in the sidebar, remembered on this computer. */
const useFolds = create<Folds>()((set, get) => {
  const initial = readFolds();
  const change = (next: Partial<Folds>) => {
    set(next);
    saveFolds(get());
  };
  return {
    open: new Set(initial.open),
    looseFolded: initial.looseFolded,
    shown: {},
    toggle(key) {
      if (key === NOT_IN_A_FOLDER) {
        change({ looseFolded: !get().looseFolded });
        return;
      }
      const open = new Set(get().open);
      if (!open.delete(key)) open.add(key);
      change({ open });
    },
    reveal(folderId) {
      if (get().open.has(folderId)) return;
      change({ open: new Set([...get().open, folderId]) });
    },
    showMore(key, count) {
      set((state) => ({ shown: { ...state.shown, [key]: (state.shown[key] ?? 0) + count } }));
    },
  };
});

/**
 * The sidebar's Folders view: one tree of the User's projects (DESIGN.md,
 * "One tree"). Each Folder lists its Minds, then its Documents (the first
 * few, then "Show N more Documents"), and opens and closes in place: the
 * main area stays as it is. After the Folders, "Not in a Folder" holds every
 * Mind and Document in none; a Mind there searches every Document, which it
 * says. Minds drag onto a Folder, or onto Not in a Folder to leave theirs,
 * and the target lights up; a Folder renames in place (double-click, Enter
 * or F2, as in Finder) and has its menu (⋯ or a right-click): Open in the
 * Library, New Mind here, Rename, Delete…. → and ← open and close a Folder
 * (↑ and ↓ move through the rows, see `moveThroughRows`).
 */
export function LibraryFolders({
  documents,
  renderDocument,
}: {
  /** The Documents to show, already filtered by Tag. */
  documents: readonly Document[];
  renderDocument(document: Document, depth: number): ReactNode;
}) {
  const t = useT();
  const groups = useAppStore((state) => state.library?.groups) ?? [];
  const assignments = useAppStore((state) => state.library?.assignments);
  const minds = useAppStore((state) => state.minds);
  const openMindFolder = useAppStore(
    (state) =>
      state.minds.find((mind) => mind.id === currentMindIdOf(state.layout))?.folderId ?? null,
  );
  const openMindId = useAppStore((state) => currentMindIdOf(state.layout));

  // The open Mind's Folder opens, so its row is in sight: when it opens, or moves there.
  // biome-ignore lint/correctness/useExhaustiveDependencies: another Mind of the same Folder opened reveals it again.
  useEffect(() => {
    if (openMindFolder) useFolds.getState().reveal(openMindFolder);
  }, [openMindId, openMindFolder]);

  // Each Folder's Documents and Minds (those in none under null), in one pass, again only when they change.
  const documentsByFolder = useMemo(() => {
    const folderOf = new Map(assignments?.map((item) => [item.documentId, item.groupId]));
    const grouped = new Map<string | null, Document[]>();
    for (const doc of documents) {
      const key = folderOf.get(doc.id) ?? null;
      const list = grouped.get(key);
      if (list) list.push(doc);
      else grouped.set(key, [doc]);
    }
    return grouped;
  }, [documents, assignments]);
  const mindsByFolder = useMemo(() => {
    const live = new Set(groups.map((group) => group.id));
    const grouped = new Map<string | null, Mind[]>();
    for (const mind of minds) {
      // A Mind whose Folder this window doesn't know (yet) shows outside any.
      const key = mind.folderId !== null && live.has(mind.folderId) ? mind.folderId : null;
      const list = grouped.get(key);
      if (list) list.push(mind);
      else grouped.set(key, [mind]);
    }
    return grouped;
  }, [minds, groups]);

  const looseMinds = mindsByFolder.get(null) ?? NO_MINDS;
  const looseDocuments = documentsByFolder.get(null) ?? NO_DOCUMENTS;
  // Shown once there is a Folder to be out of, or something in it.
  const showLoose = groups.length > 0 || looseMinds.length > 0 || looseDocuments.length > 0;

  return (
    <nav aria-label={t("sidebar.tree")} data-testid="library-folders">
      {groups.length > 0 && (
        <ul>
          {groups.map((folder) => (
            <FolderGroup
              key={folder.id}
              folder={folder}
              minds={mindsByFolder.get(folder.id) ?? NO_MINDS}
              documents={documentsByFolder.get(folder.id) ?? NO_DOCUMENTS}
              renderDocument={renderDocument}
            />
          ))}
        </ul>
      )}
      {showLoose && (
        <NotInAFolder
          minds={looseMinds}
          documents={looseDocuments}
          renderDocument={renderDocument}
        />
      )}
    </nav>
  );
}

/** A Folder's or Not in a Folder's contents: its Minds, then its first Documents and "Show N more". */
function Contents({
  listKey,
  depth,
  minds,
  documents,
  renderDocument,
}: {
  listKey: string;
  depth: number;
  minds: readonly Mind[];
  documents: readonly Document[];
  renderDocument(document: Document, depth: number): ReactNode;
}) {
  const t = useT();
  const extra = useFolds((state) => state.shown[listKey] ?? 0);
  const listed = FIRST_DOCUMENTS + extra;
  const more = Math.min(MORE_DOCUMENTS, documents.length - listed);
  if (minds.length === 0 && documents.length === 0) return null;
  return (
    <ul>
      {minds.map((mind) => (
        <MindRow key={mind.id} mind={mind} depth={depth} />
      ))}
      {documents.slice(0, listed).map((doc) => renderDocument(doc, depth))}
      {more > 0 && (
        <li className="flex h-7 items-center">
          <button
            type="button"
            {...TREE_ROW}
            data-testid="show-more-documents"
            onClick={() => useFolds.getState().showMore(listKey, MORE_DOCUMENTS)}
            // Its text on the rows' text edge at this depth.
            style={{ marginLeft: rowPadding(depth).paddingLeft + 24 }}
            className="rounded-sm text-[12px] leading-4 text-ink-meta hover:text-ink hover:underline"
          >
            {more === 1 ? t("sidebar.showMore.one") : t("sidebar.showMore", { count: more })}
          </button>
        </li>
      )}
    </ul>
  );
}

/** A drop target lit while something is dragged over it. */
const dropLit = "bg-accent-wash shadow-[inset_0_0_0_1px_var(--color-accent)]";

/**
 * A Folder's row and, while open, what is in it. Clicking (or Enter, or
 * Space) opens and closes it in place; double-click or F2 renames it.
 */
function FolderGroup({
  folder,
  minds,
  documents,
  renderDocument,
}: {
  folder: LibraryGroup;
  minds: readonly Mind[];
  documents: readonly Document[];
  renderDocument(document: Document, depth: number): ReactNode;
}) {
  const expanded = useFolds((state) => state.open.has(folder.id));
  const selected = useAppStore(
    (state) =>
      currentTabOf(state.layout)?.item.kind === "library" && state.libraryFilter === folder.id,
  );
  const drop = useFolderDrop(folder.id);
  // The first click of a double-click folds or unfolds it: back as it was before renaming.
  const rename = useRename<HTMLButtonElement>(`folder:${folder.id}`, () =>
    useFolds.getState().toggle(folder.id),
  );
  const count = documents.length;
  const t = useT();
  const objectMenu = useObjectMenu({ kind: "folder", id: folder.id });

  const onKeyDown = (event: KeyboardEvent) => {
    // Enter, a click, Space or the arrows open and close it.
    if (event.key === "ArrowRight" && !expanded) {
      event.preventDefault();
      useFolds.getState().toggle(folder.id);
    } else if (event.key === "ArrowLeft" && expanded) {
      event.preventDefault();
      useFolds.getState().toggle(folder.id);
    }
  };

  return (
    // The whole Folder takes a drop: its row, or anything listed in it.
    <li
      data-testid="library-folder"
      data-folder-id={folder.id}
      data-count={count}
      data-minds={minds.length}
      data-expanded={expanded}
      {...drop.handlers}
      data-drop-target={drop.active ? "true" : undefined}
      className={`rounded-md ${drop.active ? dropLit : ""}`}
    >
      {/* biome-ignore lint/a11y/noStaticElementInteractions: a right-click shortcut to the row's menu, whose ⋯ button is the way in by keyboard. */}
      <div {...(rename.renaming ? {} : objectMenu.rowProps)} className={rowClass(selected)}>
        {rename.renaming ? (
          <span className="flex h-full w-full min-w-0 items-center gap-2 pr-1 pl-2">
            <FolderLineIcon className={rowIconClass(selected)} />
            <InlineName
              value={folder.name}
              label={t("folder.renameLabel", { name: folder.name })}
              testId="folder-rename-field"
              {...rename.field((name) => useAppStore.getState().renameFolder(folder.id, name))}
            />
          </span>
        ) : (
          <button
            {...rename.nameProps}
            type="button"
            {...TREE_ROW}
            data-testid="folder-row"
            aria-expanded={expanded}
            aria-current={selected ? "page" : undefined}
            title={folder.description ? `${folder.name}\n${folder.description}` : folder.name}
            onClick={(event) => {
              // The second click of a double-click renames instead.
              if (event.detail <= 1) useFolds.getState().toggle(folder.id);
            }}
            onKeyDown={(event) => {
              rename.nameProps.onKeyDown(event);
              onKeyDown(event);
            }}
            className={rowButtonClass}
          >
            <FolderLineIcon className={rowIconClass(selected)} />
            <span data-testid="row-text" className="min-w-0 flex-1 truncate">
              {folder.name}
            </span>
            <span data-testid="browse-count" className="text-[12px] tabular-nums text-ink-meta">
              {count}
            </span>
          </button>
        )}
        {!rename.renaming && (
          <div className={rowActionsClass}>
            <ObjectMenu
              target={{ kind: "folder", id: folder.id }}
              label={t("folder.more", { name: folder.name })}
              here={{ "folder.rename": rename.start }}
            />
          </div>
        )}
      </div>
      {expanded && (
        <Contents
          listKey={folder.id}
          depth={1}
          minds={minds}
          documents={documents}
          renderDocument={renderDocument}
        />
      )}
    </li>
  );
}

/**
 * Everything in no Folder, under one label: its Minds, which search every
 * Document (so the label says), then its Documents. Dropping a Mind or a
 * Document on it takes it out of its Folder.
 */
function NotInAFolder({
  minds,
  documents,
  renderDocument,
}: {
  minds: readonly Mind[];
  documents: readonly Document[];
  renderDocument(document: Document, depth: number): ReactNode;
}) {
  const t = useT();
  const folded = useFolds((state) => state.looseFolded);
  const drop = useFolderDrop(null);
  const Chevron = folded ? ChevronRightLineIcon : ChevronDownLineIcon;
  return (
    <section
      aria-labelledby="not-in-a-folder-heading"
      data-testid="not-in-a-folder"
      data-count={documents.length}
      data-minds={minds.length}
      {...drop.handlers}
      data-drop-target={drop.active ? "true" : undefined}
      className={`mt-3 rounded-md ${drop.active ? dropLit : ""}`}
    >
      <div className="flex h-7 items-end gap-2 pr-2 pb-1 pl-2 text-label">
        <h2 id="not-in-a-folder-heading" className="min-w-0 font-semibold text-ink-meta">
          <button
            type="button"
            {...TREE_ROW}
            data-testid="not-in-a-folder-toggle"
            aria-expanded={!folded}
            onClick={() => useFolds.getState().toggle(NOT_IN_A_FOLDER)}
            className="flex max-w-full items-center gap-1 rounded-sm hover:text-ink-secondary"
          >
            <span className="truncate">{t("library.unsorted")}</span>
            <Chevron className="size-3 shrink-0" />
          </button>
        </h2>
        <span
          data-testid="not-in-a-folder-hint"
          title={t("sidebar.notInFolder.title")}
          className="ml-auto min-w-0 truncate font-normal text-ink-meta"
        >
          {t("sidebar.notInFolder.hint")}
        </span>
      </div>
      {!folded && (
        <Contents
          listKey={NOT_IN_A_FOLDER}
          depth={0}
          minds={minds}
          documents={documents}
          renderDocument={renderDocument}
        />
      )}
    </section>
  );
}
