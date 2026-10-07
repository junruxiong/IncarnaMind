import { type ComponentType, type ReactNode, type SVGProps, useMemo } from "react";
import { create } from "zustand";
import type { Document } from "../../../core/api";
import { buildFolderTree, type FolderNode } from "../folders";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import {
  ChevronDownLineIcon,
  ChevronRightLineIcon,
  FolderLineIcon,
  LooseDocumentsLineIcon,
} from "./lineIcons";
import { rowButtonClass, rowClass, rowIconClass, rowPadding } from "./sidebarRows";

/** The "Other Documents" group's key among the folded. */
const OTHER_DOCUMENTS = "other-documents";

interface FolderTreeState {
  /**
   * The Folders (and "Other Documents") the User folded. Everything else
   * shows its contents, so every Document is in sight at first. (A folder
   * of one-file folders, such as Zotero's storage, starts flat.)
   */
  collapsed: ReadonlySet<string>;
  toggle(key: string): void;
}

/** What is folded. Kept while the window is open. */
export const useFolderTree = create<FolderTreeState>()((set) => ({
  collapsed: new Set(),
  toggle: (key) =>
    set((state) => {
      const collapsed = new Set(state.collapsed);
      if (!collapsed.delete(key)) collapsed.add(key);
      return { collapsed };
    }),
}));

interface FolderTreeProps {
  /** The Documents to show, already filtered. */
  documents: readonly Document[];
  /**
   * While filtering (by Tag), Folders and groups with none of `documents`
   * below them are left out, and the rest show their contents.
   */
  filtering: boolean;
  /** A Document's row, at a depth: 0 at the top level. */
  renderDocument(item: Document, depth: number): ReactNode;
}

/**
 * The Documents section's tree (ADR-0010): each Linked folder with its
 * Folders as they are on disk, and then "Other Documents", the files added
 * on their own. A Document is one step deeper than its Folder. A Linked
 * folder shown flat lists all its Documents right under it. Folders follow
 * the disk, so there is nothing to create, rename or move here; clicking one
 * folds or unfolds it. Without Linked folders, the Documents are the tree.
 */
export function FolderTree({ documents, filtering, renderDocument }: FolderTreeProps) {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const linkedFolders = useAppStore((state) => state.linkedFolders);
  const collapsed = useFolderTree((state) => state.collapsed);
  const tree = useMemo(() => buildFolderTree(folders), [folders]);

  /** The Linked folders shown flat, each with its own Folder's id. */
  const flatRoots = useMemo(
    () =>
      new Map(
        linkedFolders
          .filter((linked) => linked.layout === "flat")
          .map((linked) => [linked.id, linked.folderId]),
      ),
    [linkedFolders],
  );

  /** Each Folder's Documents (a flat Linked folder's own has all of its), and the Other Documents under null. */
  const groups = useMemo(() => {
    const known = new Set(folders.map((folder) => folder.id));
    const grouped = new Map<string | null, Document[]>();
    for (const item of documents) {
      const flatRoot =
        item.linkedFolderId !== null ? flatRoots.get(item.linkedFolderId) : undefined;
      const key =
        flatRoot !== undefined && known.has(flatRoot)
          ? flatRoot
          : item.folderId !== null && known.has(item.folderId)
            ? item.folderId
            : null;
      const list = grouped.get(key) ?? [];
      list.push(item);
      grouped.set(key, list);
    }
    return grouped;
  }, [documents, folders, flatRoots]);

  /** A Folder's sub-Folders, none in a Linked folder shown flat. */
  const subfoldersOf = (node: FolderNode) =>
    flatRoots.has(node.folder.linkedFolderId) ? [] : node.children;

  /** While filtering: whether a Folder has a shown Document anywhere below it. */
  const hasDocuments = (node: FolderNode): boolean =>
    (groups.get(node.folder.id)?.length ?? 0) > 0 || subfoldersOf(node).some(hasDocuments);

  /** While filtering, everything shown shows its contents. */
  const isExpanded = (key: string) => filtering || !collapsed.has(key);

  const renderLevel = (nodes: readonly FolderNode[]): ReactNode =>
    (filtering ? nodes.filter(hasDocuments) : nodes).map((node) => {
      const { folder, depth } = node;
      const root = folder.parentId === null;
      return (
        <GroupItem
          key={folder.id}
          testId="folder-item"
          groupKey={folder.id}
          name={folder.name}
          title={root ? linkedFolders.find((l) => l.id === folder.linkedFolderId)?.path : undefined}
          icon={FolderLineIcon}
          depth={depth}
          expanded={isExpanded(folder.id)}
          data={{ "data-folder-id": folder.id, "data-linked-folder-id": folder.linkedFolderId }}
        >
          {renderLevel(subfoldersOf(node))}
          {(groups.get(folder.id) ?? []).map((item) => renderDocument(item, depth + 1))}
        </GroupItem>
      );
    });

  const others = groups.get(null) ?? [];
  // Without Linked folders there is nothing to set the Other Documents apart from.
  if (tree.length === 0) {
    return others.length === 0 ? null : (
      <ul data-testid="document-tree">{others.map((item) => renderDocument(item, 0))}</ul>
    );
  }
  return (
    <ul aria-label={t("folders.label")} data-testid="document-tree">
      {renderLevel(tree)}
      {others.length > 0 && (
        <GroupItem
          testId="other-documents"
          groupKey={OTHER_DOCUMENTS}
          name={t("documents.other")}
          icon={LooseDocumentsLineIcon}
          depth={0}
          expanded={isExpanded(OTHER_DOCUMENTS)}
        >
          {others.map((item) => renderDocument(item, 1))}
        </GroupItem>
      )}
    </ul>
  );
}

interface GroupItemProps {
  testId: string;
  /** What `useFolderTree` knows it by. */
  groupKey: string;
  name: string;
  /** Its tooltip, if not its name: a Linked folder's path. */
  title?: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  depth: number;
  expanded: boolean;
  data?: Record<`data-${string}`, string>;
  /** What is inside: sub-Folders, then Documents, shown while unfolded. */
  children: ReactNode;
}

/**
 * A Folder's row, or the "Other Documents" group's (icon, name, a chevron at
 * its end), and its contents one step deeper.
 */
function GroupItem(props: GroupItemProps) {
  const { testId, groupKey, name, title, icon: Icon, depth, expanded, data, children } = props;
  const toggle = useFolderTree((state) => state.toggle);
  const Chevron = expanded ? ChevronDownLineIcon : ChevronRightLineIcon;

  return (
    <li>
      <div data-testid={testId} data-depth={depth} {...data} className={rowClass(false, "item")}>
        <button
          type="button"
          data-testid="folder-toggle"
          aria-expanded={expanded}
          title={title ?? name}
          onClick={() => toggle(groupKey)}
          className={rowButtonClass}
          style={rowPadding(depth)}
        >
          <Icon className={rowIconClass(false)} />
          <span data-testid="row-text" className="min-w-0 flex-1 truncate">
            {name}
          </span>
          <Chevron className="size-3.5 shrink-0 text-ink-meta" />
        </button>
      </div>
      {expanded && <ul>{children}</ul>}
    </li>
  );
}
