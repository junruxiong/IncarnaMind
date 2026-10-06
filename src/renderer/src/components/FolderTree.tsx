import { type DragEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import type { Folder } from "../../../core/api";
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
  AllDocumentsIcon,
  ChevronIcon,
  FolderIcon,
  FolderPlusIcon,
  PencilIcon,
  TrashIcon,
} from "./icons";

/** Where a new Folder is being named: in a Folder (its id), at the top level (null), or nowhere. */
export type NewFolderPlace = string | null | undefined;

const INDENT_PX = 12;
/** Left padding of a row at `depth`. Rows start with a chevron column, so icons line up. */
const indent = (depth: number) => ({ paddingLeft: 4 + depth * INDENT_PX });
/** The chevron column's width plus the gap after it. */
const CHEVRON_PX = 18;
const actionButton = "rounded-[6px] p-[3px] text-gray-500 hover:bg-gray-200 hover:text-gray-700";

interface FolderTreeProps {
  /** Shows an input for naming a new Folder there. */
  newFolderIn: NewFolderPlace;
  onNewFolder(parentId: string | null): void;
  onNewFolderDone(): void;
}

/**
 * The Documents section's Folder tree: "All Documents", then the Folders,
 * nested. Clicking one shows only its Documents (sub-Folders included).
 * Documents, and Folders, are moved by dropping them on a Folder; dropping on
 * "All Documents" unfiles a Document or moves a Folder to the top level.
 */
export function FolderTree({ newFolderIn, onNewFolder, onNewFolderDone }: FolderTreeProps) {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const folderFilter = useAppStore((state) => state.folderFilter);
  const filterByFolder = useAppStore((state) => state.filterByFolder);
  const tree = useMemo(() => buildFolderTree(folders), [folders]);
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set());
  const [deleting, setDeleting] = useState<Folder | null>(null);

  // A new sub-Folder is named inside its parent, so open the parent.
  useEffect(() => {
    if (typeof newFolderIn === "string") setExpanded((open) => new Set(open).add(newFolderIn));
  }, [newFolderIn]);

  const toggle = (id: string) =>
    setExpanded((open) => {
      const next = new Set(open);
      if (!next.delete(id)) next.add(id);
      return next;
    });

  if (folders.length === 0 && newFolderIn === undefined) return null;

  const renderNodes = (
    nodes: readonly FolderNode[],
    parentId: string | null,
    depth: number,
  ): ReactNode => (
    <ul>
      {nodes.map((node) => (
        <FolderItem
          key={node.folder.id}
          node={node}
          expanded={expanded.has(node.folder.id)}
          selected={folderFilter === node.folder.id}
          onToggle={() => toggle(node.folder.id)}
          onSelect={() => void filterByFolder(node.folder.id)}
          onNewFolder={() => onNewFolder(node.folder.id)}
          onDelete={() => setDeleting(node.folder)}
        >
          {renderNodes(node.children, node.folder.id, depth + 1)}
        </FolderItem>
      ))}
      {newFolderIn === parentId && (
        <li>
          <NewFolderInput parentId={parentId} depth={depth} onDone={onNewFolderDone} />
        </li>
      )}
    </ul>
  );

  return (
    <nav aria-label={t("folders.label")} className="mx-3 mb-1 border-b border-gray-200 pb-1">
      <AllDocumentsItem
        selected={folderFilter === null}
        onSelect={() => void filterByFolder(null)}
      />
      {renderNodes(tree, null, 0)}
      <DeleteFolderDialog target={deleting} onClose={() => setDeleting(null)} />
    </nav>
  );
}

/**
 * Makes a row a drop target for Documents and Folders dragged within the
 * sidebar. `folderId` is where they go: a Folder, or null for "All Documents".
 * A Folder can't be dropped into itself or below itself.
 */
function useDropTarget(folderId: string | null) {
  const folders = useAppStore((state) => state.folders);
  const moveDocument = useAppStore((state) => state.moveDocument);
  const moveFolder = useAppStore((state) => state.moveFolder);
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
        if (drag.kind === "document") void moveDocument(drag.id, folderId);
        else void moveFolder(drag.id, folderId);
      },
    },
  };
}

const rowClass = (selected: boolean, over: boolean) =>
  `group my-[1px] flex items-center gap-[2px] rounded-[9px] py-[3px] pr-1 text-sm ${
    over ? "bg-sky-100 ring-1 ring-sky-400" : selected ? "bg-gray-200" : "hover:bg-gray-100"
  }`;

function AllDocumentsItem({ selected, onSelect }: { selected: boolean; onSelect(): void }) {
  const t = useT();
  const drop = useDropTarget(null);
  return (
    <div data-testid="all-documents" className={rowClass(selected, drop.over)} style={indent(0)}>
      <button
        type="button"
        aria-current={selected ? "true" : undefined}
        onClick={onSelect}
        {...drop.handlers}
        className="flex min-w-0 flex-1 items-center gap-[6px] py-[2px] text-left"
        style={{ paddingLeft: CHEVRON_PX }}
      >
        <AllDocumentsIcon className="size-4 shrink-0" />
        <span className="truncate text-gray-700">{t("folders.all")}</span>
      </button>
    </div>
  );
}

interface FolderItemProps {
  node: FolderNode;
  expanded: boolean;
  selected: boolean;
  onToggle(): void;
  onSelect(): void;
  onNewFolder(): void;
  onDelete(): void;
  /** The sub-Folders' list, shown while expanded. */
  children: ReactNode;
}

function FolderItem(props: FolderItemProps) {
  const { node, expanded, selected, onToggle, onSelect, onNewFolder, onDelete, children } = props;
  const { folder, depth } = node;
  const t = useT();
  const renameFolder = useAppStore((state) => state.renameFolder);
  const drop = useDropTarget(folder.id);
  const [renaming, setRenaming] = useState(false);
  const hasChildren = node.children.length > 0;

  return (
    <li>
      <div
        data-testid="folder-item"
        data-folder-id={folder.id}
        className={rowClass(selected, drop.over)}
        style={indent(depth)}
      >
        {hasChildren ? (
          <button
            type="button"
            aria-expanded={expanded}
            aria-label={t(expanded ? "folders.collapse" : "folders.expand", { name: folder.name })}
            onClick={onToggle}
            className="shrink-0 rounded-[6px] p-[2px] text-gray-400 hover:bg-gray-200 hover:text-gray-700"
          >
            <ChevronIcon className={`size-3 transition-transform ${expanded ? "rotate-90" : ""}`} />
          </button>
        ) : (
          <span className="size-4 shrink-0" />
        )}
        {renaming ? (
          <div className="flex min-w-0 flex-1 items-center gap-[6px]">
            <FolderIcon className="size-4 shrink-0" />
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
            data-testid="folder-filter"
            aria-current={selected ? "true" : undefined}
            onClick={onSelect}
            draggable
            onDragStart={(event) => startSidebarDrag(event, { kind: "folder", id: folder.id })}
            onDragEnd={endSidebarDrag}
            {...drop.handlers}
            className="flex min-w-0 flex-1 items-center gap-[6px] py-[2px] text-left"
          >
            <FolderIcon className="size-4 shrink-0" />
            <span className="truncate text-gray-700" title={folder.name}>
              {folder.name}
            </span>
          </button>
        )}
        {!renaming && (
          <div className="flex shrink-0 items-center opacity-0 group-focus-within:opacity-100 group-hover:opacity-100">
            <button
              type="button"
              data-testid="new-subfolder"
              aria-label={t("folders.newInside", { name: folder.name })}
              title={t("folders.newInside", { name: folder.name })}
              onClick={onNewFolder}
              className={actionButton}
            >
              <FolderPlusIcon className="size-[14px]" />
            </button>
            <button
              type="button"
              aria-label={t("folders.rename", { name: folder.name })}
              title={t("folders.rename", { name: folder.name })}
              onClick={() => setRenaming(true)}
              className={actionButton}
            >
              <PencilIcon className="size-[14px]" />
            </button>
            <button
              type="button"
              data-testid="delete-folder"
              aria-label={t("folders.delete", { name: folder.name })}
              title={t("folders.delete", { name: folder.name })}
              onClick={onDelete}
              className={actionButton}
            >
              <TrashIcon className="size-[14px]" />
            </button>
          </div>
        )}
      </div>
      {expanded && children}
    </li>
  );
}

function NewFolderInput(props: { parentId: string | null; depth: number; onDone(): void }) {
  const { parentId, depth, onDone } = props;
  const t = useT();
  const createFolder = useAppStore((state) => state.createFolder);
  return (
    <div
      className="my-[1px] flex items-center gap-[6px] py-[3px] pr-1"
      style={{ paddingLeft: indent(depth).paddingLeft + CHEVRON_PX }}
    >
      <FolderIcon className="size-4 shrink-0" />
      <FolderNameInput
        initial=""
        label={t("folders.nameLabel")}
        onSubmit={(name) => void createFolder(name, parentId)}
        onDone={onDone}
      />
    </div>
  );
}

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
      className="w-full min-w-0 rounded-[6px] border border-gray-300 bg-white px-1 text-sm text-gray-700 outline-none focus:border-gray-400"
    />
  );
}

/** Asks before deleting a Folder, and says its Documents are kept. A native modal <dialog>. */
function DeleteFolderDialog({ target, onClose }: { target: Folder | null; onClose(): void }) {
  const t = useT();
  const dialog = useRef<HTMLDialogElement>(null);
  const deleteFolder = useAppStore((state) => state.deleteFolder);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (target && !element.open) element.showModal();
    if (!target && element.open) element.close();
  }, [target]);

  const confirm = () => {
    if (target) void deleteFolder(target.id);
    onClose();
  };

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      aria-labelledby="delete-folder-title"
      className="m-auto w-96 rounded-[9px] bg-white p-4 text-gray-800 shadow-custom-focus backdrop:bg-black/20"
    >
      <h2 id="delete-folder-title" className="text-lg font-semibold">
        {t("folders.delete.title")}
      </h2>
      <p className="mt-2 text-sm break-words text-gray-600">
        {t("folders.delete.body", { name: target?.name ?? "" })}
      </p>
      <div className="mt-4 flex justify-end gap-2">
        <button
          type="button"
          onClick={onClose}
          className="rounded-[9px] border border-gray-300 px-4 py-2 text-sm hover:bg-gray-100"
        >
          {t("folders.delete.cancel")}
        </button>
        <button
          type="button"
          data-testid="confirm-delete-folder"
          onClick={confirm}
          className="rounded-[9px] bg-red-600 px-4 py-2 text-sm text-white hover:bg-red-700"
        >
          {t("folders.delete.confirm")}
        </button>
      </div>
    </dialog>
  );
}
