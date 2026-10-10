import { type ComponentType, type ReactNode, type SVGProps, useMemo, useState } from "react";
import { create } from "zustand";
import type { Document, LinkedFolder } from "../../../core/api";
import { buildFolderTree, type FolderNode } from "../folders";
import { useLanguage, useT } from "../i18n";
import {
  indexingShare,
  type LinkedFolderRowState,
  linkedFolderRowState,
  rowStateLabel,
} from "../linkedFolders";
import { selectTagFilterKey, useAppStore } from "../store";
import { TREE_ROW } from "../treeKeys";
import { ExampleChip } from "./GettingStarted";
import { LinkedFolderMenu, UnlinkFolderDialog } from "./LinkedFolderMenu";
import {
  ChevronDownLineIcon,
  ChevronRightLineIcon,
  FolderLineIcon,
  LooseDocumentsLineIcon,
} from "./lineIcons";
import {
  INDENT_PX,
  openRowMenu,
  rowActionButtonClass,
  rowActionsClass,
  rowButtonClass,
  rowClass,
  rowIconClass,
  rowPadding,
} from "./sidebarRows";

/** The "Other Documents" group's key among the folded. */
const OTHER_DOCUMENTS = "other-documents";

interface FolderTreeState {
  /**
   * The Folders (and "Other Documents") the User folded. Everything else
   * shows its contents, so every Document is in sight at first. (A big
   * folder of one-file folders, such as Zotero's storage, starts flat.)
   */
  collapsed: ReadonlySet<string>;
  /**
   * What the User folded while filtering by a Tag: kept apart, so a filtered
   * view starts with everything in it unfolded, folds what the User folds
   * there, and leaves the unfiltered tree as it was. Forgotten for another Tag.
   */
  filtered: { tagId: string; collapsed: ReadonlySet<string> } | null;
  /** Folds or unfolds a Folder or group, in the tree as it shows now: filtered by `tagId`, or not. */
  toggle(key: string, tagId: string | null): void;
}

const toggled = (set: ReadonlySet<string>, key: string): ReadonlySet<string> => {
  const next = new Set(set);
  if (!next.delete(key)) next.add(key);
  return next;
};

/** What is folded. Kept while the window is open. */
export const useFolderTree = create<FolderTreeState>()((set) => ({
  collapsed: new Set(),
  filtered: null,
  toggle: (key, tagId) =>
    set((state) => {
      if (tagId === null) return { collapsed: toggled(state.collapsed, key) };
      const before = state.filtered?.tagId === tagId ? state.filtered.collapsed : new Set<string>();
      return { filtered: { tagId, collapsed: toggled(before, key) } };
    }),
}));

interface FolderTreeProps {
  /** The Documents to show, already filtered. */
  documents: readonly Document[];
  /**
   * While filtering (by Tag), Folders and groups with none of `documents`
   * below them are left out, and the rest show their contents until the
   * User folds them in the filtered view (see `FolderTreeState.filtered`).
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
 *
 * A Linked folder's own row says how it is (indexing, paused, unavailable,
 * online-only files skipped, nothing in it) at its end, on its one line, and
 * offers its menu (see `LinkedFolderMenu`).
 */
export function FolderTree({ documents, filtering, renderDocument }: FolderTreeProps) {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const linkedFolders = useAppStore((state) => state.linkedFolders);
  const tagFilter = useAppStore(selectTagFilterKey);
  const collapsed = useFolderTree((state) => state.collapsed);
  const filteredFolds = useFolderTree((state) => state.filtered);
  const tree = useMemo(() => buildFolderTree(folders), [folders]);
  const [unlinking, setUnlinking] = useState<{ linked: LinkedFolder; name: string } | null>(null);

  const linkedById = useMemo(
    () => new Map(linkedFolders.map((linked) => [linked.id, linked])),
    [linkedFolders],
  );

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

  /** While filtering, what is shown shows its contents unless folded in this filtered view. */
  const isExpanded = (key: string) =>
    filtering && tagFilter !== null
      ? !(filteredFolds?.tagId === tagFilter && filteredFolds.collapsed.has(key))
      : !collapsed.has(key);

  const renderLevel = (nodes: readonly FolderNode[]): ReactNode =>
    (filtering ? nodes.filter(hasDocuments) : nodes).map((node) => {
      const { folder, depth } = node;
      const subfolders = subfoldersOf(node);
      const own = groups.get(folder.id) ?? [];
      const hasContents = subfolders.length > 0 || own.length > 0;
      const linked = folder.parentId === null ? linkedById.get(folder.linkedFolderId) : undefined;
      const contents = (
        <>
          {renderLevel(subfolders)}
          {own.map((item) => renderDocument(item, depth + 1))}
        </>
      );
      if (linked) {
        return (
          <LinkedFolderItem
            key={folder.id}
            linked={linked}
            folderId={folder.id}
            name={folder.name}
            depth={depth}
            hasContents={hasContents}
            expanded={isExpanded(folder.id)}
            onUnlink={() => setUnlinking({ linked, name: folder.name })}
          >
            {contents}
          </LinkedFolderItem>
        );
      }
      return (
        <GroupItem
          key={folder.id}
          testId="folder-item"
          groupKey={folder.id}
          name={folder.name}
          icon={FolderLineIcon}
          depth={depth}
          expanded={isExpanded(folder.id)}
          expandable={hasContents}
          data={{ "data-folder-id": folder.id, "data-linked-folder-id": folder.linkedFolderId }}
        >
          {contents}
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
    <>
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
            expandable
          >
            {others.map((item) => renderDocument(item, 1))}
          </GroupItem>
        )}
      </ul>
      <UnlinkFolderDialog target={unlinking} onClose={() => setUnlinking(null)} />
    </>
  );
}

/**
 * A Linked folder's own row: its Folder's row, with its state at the end
 * (see `linkedFolderRowState`) and, while it is indexed, a thin bar under
 * its name. Muted while it can't be reached. Its menu sits with its actions.
 */
function LinkedFolderItem({
  linked,
  folderId,
  name,
  depth,
  hasContents,
  expanded,
  onUnlink,
  children,
}: {
  linked: LinkedFolder;
  folderId: string;
  name: string;
  depth: number;
  hasContents: boolean;
  expanded: boolean;
  onUnlink(): void;
  children: ReactNode;
}) {
  const t = useT();
  const language = useLanguage();
  const state = linkedFolderRowState(linked, hasContents);
  const label = rowStateLabel(state, t, language);
  const isExample = useAppStore((current) => current.examples?.linkedFolderId === linked.id);
  return (
    <GroupItem
      testId="folder-item"
      groupKey={folderId}
      name={name}
      title={linked.path}
      icon={FolderLineIcon}
      depth={depth}
      expanded={expanded}
      expandable={hasContents}
      muted={state.kind === "unavailable"}
      status={label && { ...label, kind: state.kind }}
      progress={indexingShare(state)}
      paused={state.kind === "paused"}
      chip={isExample && <ExampleChip />}
      actions={
        <LinkedFolderMenu
          linked={linked}
          name={name}
          buttonClassName={rowActionButtonClass}
          onUnlink={onUnlink}
        />
      }
      data={{
        "data-folder-id": folderId,
        "data-linked-folder-id": linked.id,
        "data-root": "true",
        "data-state": state.kind,
        "data-status": linked.status,
        "data-layout": linked.layout,
      }}
    >
      {children}
    </GroupItem>
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
  /** Whether there is anything inside to show: without, there is no chevron to unfold. */
  expandable: boolean;
  /** Its name in the meta ink, e.g. a Linked folder that can't be reached. */
  muted?: boolean;
  /** What its row's end says, before the chevron: short, shorter for a narrow sidebar, and in full. */
  status?: {
    kind: LinkedFolderRowState["kind"];
    short: string;
    compact: string;
    full: string;
  } | null;
  /** A thin bar under the row, this far along (0 to 1). Null: none. */
  progress?: number | null;
  /** The bar is held: greyed. */
  paused?: boolean;
  /** Its actions, over its right end while pointed at (see `rowActionsClass`). */
  actions?: ReactNode;
  /** After its name, unless a status takes the room: the examples' Linked folder says "Example". */
  chip?: ReactNode;
  data?: Record<`data-${string}`, string>;
  /** What is inside: sub-Folders, then Documents, shown while unfolded. */
  children: ReactNode;
}

/**
 * A Folder's row, or the "Other Documents" group's (icon, name, anything its
 * end says, a chevron), and its contents one step deeper. Everything stays on
 * its one 28px line: the name gives way first, then the status.
 */
function GroupItem(props: GroupItemProps) {
  const { testId, groupKey, name, title, icon: Icon, depth, expanded, expandable } = props;
  const { muted = false, status, progress = null, paused = false, actions, chip, data } = props;
  const children = props.children;
  const toggle = useFolderTree((state) => state.toggle);
  const tagFilter = useAppStore(selectTagFilterKey);
  const open = expandable && expanded;
  const Chevron = open ? ChevronDownLineIcon : ChevronRightLineIcon;
  const inner = (
    <>
      <Icon className={rowIconClass(false)} />
      <span data-testid="row-text" className="min-w-12 flex-1 truncate">
        {name}
      </span>
      {!status && chip}
      {status && (
        <span
          data-testid="folder-status"
          data-state={status.kind}
          title={status.full}
          className="min-w-0 shrink truncate text-label font-semibold text-ink-meta"
        >
          <span aria-hidden="true" className="@max-[13rem]:hidden">
            {status.short}
          </span>
          <span aria-hidden="true" className="hidden @max-[13rem]:inline">
            {status.compact}
          </span>
          <span className="sr-only">{status.full}</span>
        </span>
      )}
      {expandable && <Chevron className="size-3.5 shrink-0 text-ink-meta" />}
    </>
  );

  return (
    <li>
      {/* biome-ignore lint/a11y/noStaticElementInteractions: a right-click shortcut to the row's menu, whose ⋯ button is the way in by keyboard. */}
      <div
        data-testid={testId}
        data-depth={depth}
        {...data}
        onContextMenu={actions ? (event) => openRowMenu(event, "linked-folder-menu") : undefined}
        className={`${rowClass(false, muted ? "muted" : "item")} @container`}
      >
        {expandable ? (
          <button
            type="button"
            {...TREE_ROW}
            data-testid="folder-toggle"
            aria-expanded={open}
            title={title ?? name}
            onClick={() => toggle(groupKey, tagFilter)}
            className={rowButtonClass}
            style={rowPadding(depth)}
          >
            {inner}
          </button>
        ) : (
          <div title={title ?? name} className={rowButtonClass} style={rowPadding(depth)}>
            {inner}
          </div>
        )}
        {progress !== null && (
          <span
            aria-hidden="true"
            data-testid="folder-progress"
            className="pointer-events-none absolute right-2 bottom-0.5 h-0.5 overflow-hidden rounded-full bg-rule"
            // Under the name: from the row's text edge to its end.
            style={{ left: 32 + depth * INDENT_PX }}
          >
            <span
              className={`block h-full rounded-full transition-[width] duration-180 ease-in-out motion-reduce:transition-none ${
                paused ? "bg-ink-placeholder" : "bg-accent"
              }`}
              style={{ width: `${Math.round(progress * 1000) / 10}%` }}
            />
          </span>
        )}
        {actions && <div className={rowActionsClass}>{actions}</div>}
      </div>
      {open && <ul>{children}</ul>}
    </li>
  );
}
