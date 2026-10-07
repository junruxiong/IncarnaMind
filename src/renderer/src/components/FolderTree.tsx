import { type DragEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { create } from "zustand";
import type { Document, Folder } from "../../../core/api";
import {
  buildFolderTree,
  endSidebarDrag,
  type FolderNode,
  isSameOrInside,
  type SidebarDrag,
  sidebarDragOf,
  startSidebarDrag,
} from "../folders";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import {
  ChevronDownLineIcon,
  ChevronRightLineIcon,
  FolderLineIcon,
  FolderPlusLineIcon,
  PencilLineIcon,
  TrashLineIcon,
} from "./lineIcons";
import {
  rowActionButtonClass,
  rowActionsClass,
  rowButtonClass,
  rowClass,
  rowIconClass,
  rowPadding,
} from "./sidebarRows";
import {
  buttonClass,
  dangerButtonClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
} from "./ui";
import { useModal } from "./useModal";

/** Where a new Folder is being named: in a Folder (its id), at the top level (null), or nowhere. */
export type NewFolderPlace = string | null | undefined;

interface FolderTreeState {
  /** Folders the User folded. Every other Folder shows its contents. */
  collapsed: ReadonlySet<string>;
  toggle(folderId: string): void;
  /** Unfolds a Folder and every Folder above it, e.g. after something is moved into it. */
  reveal(folderId: string | null): void;
}

/** Which Folders are folded. Kept while the window is open. */
export const useFolderTree = create<FolderTreeState>()((set) => ({
  collapsed: new Set(),
  toggle: (folderId) =>
    set((state) => {
      const collapsed = new Set(state.collapsed);
      if (!collapsed.delete(folderId)) collapsed.add(folderId);
      return { collapsed };
    }),
  reveal: (folderId) =>
    set((state) => {
      if (folderId === null) return state;
      const parents = new Map(
        useAppStore.getState().folders.map((folder) => [folder.id, folder.parentId]),
      );
      const collapsed = new Set(state.collapsed);
      const seen = new Set<string>();
      let current: string | null | undefined = folderId;
      while (current && !seen.has(current)) {
        seen.add(current);
        collapsed.delete(current);
        current = parents.get(current);
      }
      return { collapsed };
    }),
}));

interface FolderTreeProps {
  /** The Documents to show, already filtered. */
  documents: readonly Document[];
  /**
   * While filtering (by Tag), Folders with none of `documents` below them are
   * left out, and the rest show their contents.
   */
  filtering: boolean;
  /** Shows an input for naming a new Folder there. */
  newFolderIn: NewFolderPlace;
  onNewFolder(parentId: string | null): void;
  onNewFolderDone(): void;
  /** A Document's row, at a depth: 0 at the top level. */
  renderDocument(item: Document, depth: number): ReactNode;
}

/**
 * The Documents section's tree: Folders, each with its sub-Folders and then
 * its Documents one step deeper, and then the Documents in no Folder.
 * Clicking a Folder folds or unfolds it. Documents and Folders are moved by
 * dropping them on a Folder, or on the "Documents" label for the top level.
 */
export function FolderTree(props: FolderTreeProps) {
  const { documents, filtering, newFolderIn, onNewFolder, onNewFolderDone, renderDocument } = props;
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const collapsed = useFolderTree((state) => state.collapsed);
  const reveal = useFolderTree((state) => state.reveal);
  const tree = useMemo(() => buildFolderTree(folders), [folders]);
  const [deleting, setDeleting] = useState<Folder | null>(null);

  // A new sub-Folder is named inside its parent, so unfold the parent.
  useEffect(() => {
    if (typeof newFolderIn === "string") reveal(newFolderIn);
  }, [newFolderIn, reveal]);

  /** Each Folder's Documents; a Document whose Folder isn't listed shows at the top level. */
  const byFolder = useMemo(() => {
    const known = new Set(folders.map((folder) => folder.id));
    const grouped = new Map<string | null, Document[]>();
    for (const item of documents) {
      const folderId = item.folderId !== null && known.has(item.folderId) ? item.folderId : null;
      const list = grouped.get(folderId) ?? [];
      list.push(item);
      grouped.set(folderId, list);
    }
    return grouped;
  }, [documents, folders]);

  /** While filtering: whether a Folder has a shown Document anywhere below it. */
  const hasDocuments = (node: FolderNode): boolean =>
    (byFolder.get(node.folder.id)?.length ?? 0) > 0 || node.children.some(hasDocuments);

  const renderLevel = (
    nodes: readonly FolderNode[],
    parentId: string | null,
    depth: number,
  ): ReactNode => {
    const shown = filtering ? nodes.filter(hasDocuments) : nodes;
    const items = byFolder.get(parentId) ?? [];
    return (
      <>
        {shown.map((node) => (
          <FolderItem
            key={node.folder.id}
            node={node}
            expanded={filtering || !collapsed.has(node.folder.id)}
            onNewFolder={() => onNewFolder(node.folder.id)}
            onDelete={() => setDeleting(node.folder)}
          >
            {renderLevel(node.children, node.folder.id, depth + 1)}
          </FolderItem>
        ))}
        {newFolderIn === parentId && (
          <li>
            <NewFolderInput parentId={parentId} depth={depth} onDone={onNewFolderDone} />
          </li>
        )}
        {items.map((item) => renderDocument(item, depth))}
      </>
    );
  };

  return (
    <>
      <ul aria-label={t("folders.label")} data-testid="document-tree">
        {renderLevel(tree, null, 0)}
      </ul>
      <DeleteFolderDialog target={deleting} onClose={() => setDeleting(null)} />
    </>
  );
}

/**
 * Makes an element a drop target for Documents and Folders dragged within
 * the sidebar. `folderId` is where they go: a Folder, or null for the top
 * level. A Folder can't be dropped into itself or below itself. What lands is
 * shown: its Folder unfolds.
 */
export function useDropTarget(folderId: string | null) {
  const folders = useAppStore((state) => state.folders);
  const moveDocument = useAppStore((state) => state.moveDocument);
  const moveFolder = useAppStore((state) => state.moveFolder);
  const reveal = useFolderTree((state) => state.reveal);
  const [over, setOver] = useState(false);

  const accepts = (drag: SidebarDrag) =>
    drag.kind === "document" ||
    (folderId === null
      ? folders.some((folder) => folder.id === drag.id && folder.parentId !== null)
      : !isSameOrInside(folders, folderId, drag.id));

  return {
    over,
    handlers: {
      onDragOver(event: DragEvent) {
        const drag = sidebarDragOf(event);
        if (!drag || !accepts(drag)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        setOver(true);
      },
      onDragLeave(event: DragEvent) {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOver(false);
      },
      onDrop(event: DragEvent) {
        const drag = sidebarDragOf(event);
        setOver(false);
        if (!drag || !accepts(drag)) return;
        event.preventDefault();
        reveal(folderId);
        if (drag.kind === "document") void moveDocument(drag.id, folderId);
        else void moveFolder(drag.id, folderId);
      },
    },
  };
}

interface FolderItemProps {
  node: FolderNode;
  expanded: boolean;
  onNewFolder(): void;
  onDelete(): void;
  /** What is inside: sub-Folders, then Documents, shown while unfolded. */
  children: ReactNode;
}

/** A Folder's row (icon, name, a chevron at its end), and its contents one step deeper. */
function FolderItem({ node, expanded, onNewFolder, onDelete, children }: FolderItemProps) {
  const { folder, depth } = node;
  const t = useT();
  const renameFolder = useAppStore((state) => state.renameFolder);
  const toggle = useFolderTree((state) => state.toggle);
  const drop = useDropTarget(folder.id);
  const [renaming, setRenaming] = useState(false);
  const Chevron = expanded ? ChevronDownLineIcon : ChevronRightLineIcon;

  return (
    <li>
      <div
        data-testid="folder-item"
        data-folder-id={folder.id}
        data-depth={depth}
        {...drop.handlers}
        className={rowClass(false, "item", drop.over)}
      >
        {renaming ? (
          <div
            className="flex h-full w-full min-w-0 items-center gap-2 pr-1"
            style={rowPadding(depth)}
          >
            <FolderLineIcon className={rowIconClass(false)} />
            <FolderNameInput
              initial={folder.name}
              label={t("folders.renameLabel", { name: folder.name })}
              onSubmit={(name) => {
                if (name !== folder.name) void renameFolder(folder.id, name);
              }}
              onDone={() => setRenaming(false)}
            />
          </div>
        ) : (
          <button
            type="button"
            data-testid="folder-toggle"
            aria-expanded={expanded}
            title={folder.name}
            onClick={() => toggle(folder.id)}
            draggable
            onDragStart={(event) => startSidebarDrag(event, { kind: "folder", id: folder.id })}
            onDragEnd={endSidebarDrag}
            className={rowButtonClass}
            style={rowPadding(depth)}
          >
            <FolderLineIcon className={rowIconClass(false)} />
            <span data-testid="row-text" className="min-w-0 flex-1 truncate">
              {folder.name}
            </span>
            <Chevron className="size-3.5 shrink-0 text-ink-meta" />
          </button>
        )}
        {!renaming && (
          <div className={rowActionsClass}>
            <button
              type="button"
              data-testid="new-subfolder"
              aria-label={t("folders.newInside", { name: folder.name })}
              title={t("folders.newInside", { name: folder.name })}
              onClick={onNewFolder}
              className={rowActionButtonClass}
            >
              <FolderPlusLineIcon className="size-[15px]" />
            </button>
            <button
              type="button"
              aria-label={t("folders.rename", { name: folder.name })}
              title={t("folders.rename", { name: folder.name })}
              onClick={() => setRenaming(true)}
              className={rowActionButtonClass}
            >
              <PencilLineIcon className="size-[15px]" />
            </button>
            <button
              type="button"
              data-testid="delete-folder"
              aria-label={t("folders.delete", { name: folder.name })}
              title={t("folders.delete", { name: folder.name })}
              onClick={onDelete}
              className={rowActionButtonClass}
            >
              <TrashLineIcon className="size-[15px]" />
            </button>
          </div>
        )}
      </div>
      {expanded && <ul>{children}</ul>}
    </li>
  );
}

function NewFolderInput(props: { parentId: string | null; depth: number; onDone(): void }) {
  const { parentId, depth, onDone } = props;
  const t = useT();
  const createFolder = useAppStore((state) => state.createFolder);
  return (
    <div className="flex h-7 items-center gap-2 pr-1" style={rowPadding(depth)}>
      <FolderLineIcon className={rowIconClass(false)} />
      <FolderNameInput
        initial=""
        label={t("folders.nameLabel")}
        onSubmit={(name) => void createFolder(name, parentId)}
        onDone={onDone}
      />
    </div>
  );
}

/** A name being typed in a row. */
export const rowInputClass =
  "h-6 w-full min-w-0 rounded-sm border border-accent bg-sheet px-1.5 text-ui text-ink outline-1 outline-accent";

/** Enter or leaving the field submits a non-empty name; Esc cancels. */
function FolderNameInput(props: {
  initial: string;
  label: string;
  onSubmit(name: string): void;
  onDone(): void;
}) {
  const { initial, label, onSubmit, onDone } = props;
  const [value, setValue] = useState(initial);
  const field = useRef<HTMLInputElement>(null);
  const finished = useRef(false);

  useEffect(() => {
    field.current?.focus();
    field.current?.select();
  }, []);

  const finish = (save: boolean) => {
    if (finished.current) return;
    finished.current = true;
    const name = value.trim();
    if (save && name) onSubmit(name);
    onDone();
  };

  return (
    <input
      ref={field}
      value={value}
      data-testid="folder-name-input"
      aria-label={label}
      onChange={(event) => setValue(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === "Escape") {
          event.preventDefault();
          finish(event.key === "Enter");
        }
      }}
      onBlur={() => finish(true)}
      className={rowInputClass}
    />
  );
}

/** Asks before deleting a Folder, and says its Documents are kept. */
function DeleteFolderDialog({ target, onClose }: { target: Folder | null; onClose(): void }) {
  const t = useT();
  const dialog = useModal(target !== null);
  const deleteFolder = useAppStore((state) => state.deleteFolder);

  const confirm = () => {
    if (target) void deleteFolder(target.id);
    onClose();
  };

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      aria-labelledby="delete-folder-title"
      className={`${dialogClass} w-[26rem]`}
    >
      <div className={dialogBodyClass}>
        <h2 id="delete-folder-title" className={dialogTitleClass}>
          {t("folders.delete.title")}
        </h2>
        <p className={dialogTextClass}>{t("folders.delete.body", { name: target?.name ?? "" })}</p>
        <div className={dialogActionsClass}>
          <button type="button" onClick={onClose} className={buttonClass}>
            {t("folders.delete.cancel")}
          </button>
          <button
            type="button"
            data-testid="confirm-delete-folder"
            onClick={confirm}
            className={dangerButtonClass}
          >
            {t("folders.delete.confirm")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
