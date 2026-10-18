import {
  type ComponentType,
  type ReactNode,
  type SVGProps,
  useCallback,
  useEffect,
  useMemo,
} from "react";
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
import type { ObjectRef } from "../objects";
import { useRequests } from "../requests";
import { sorted } from "../sort";
import { useSort } from "../sortChoices";
import { documentFields, folderFields } from "../sortFields";
import { allExamples, useAppStore } from "../store";
import { foldKeys, useTreeItem } from "../treeKeys";
import { ExampleChip } from "./GettingStarted";
import { ChevronDownLineIcon, ChevronRightLineIcon, FolderLineIcon } from "./lineIcons";
import { ObjectMenu, useObjectMenu } from "./ObjectMenu";
import {
  clearsActions,
  INDENT_PX,
  rowActionsClass,
  rowButtonClass,
  rowClass,
  rowIconClass,
  rowPadding,
} from "./sidebarRows";

/** The sort key of the files added on their own, which a Linked folder's tree leaves out. */
const OTHER_DOCUMENTS = "other-documents";

/** As long as the wash's animation lasts (`[data-just-linked]` in styles.css). */
const LINKED_WASH_MS = 1600;

interface FolderTreeState {
  /**
   * The folders inside a Linked folder the User folded. Everything else
   * shows its contents, so every Document is in sight once its Linked folder
   * is unfolded. (A big folder of one-file folders, such as Zotero's
   * storage, starts flat.)
   */
  collapsed: ReadonlySet<string>;
  /** The Linked folders the User unfolded: each starts folded, its row and state in sight. */
  opened: ReadonlySet<string>;
  toggle(key: string): void;
  toggleOpened(key: string): void;
}

const toggled = (set: ReadonlySet<string>, key: string): ReadonlySet<string> => {
  const next = new Set(set);
  if (!next.delete(key)) next.add(key);
  return next;
};

/** What is folded. Kept while the window is open. */
export const useFolderTree = create<FolderTreeState>()((set) => ({
  collapsed: new Set(),
  opened: new Set(),
  toggle: (key) => set((state) => ({ collapsed: toggled(state.collapsed, key) })),
  toggleOpened: (key) => set((state) => ({ opened: toggled(state.opened, key) })),
}));

interface FolderTreeProps {
  documents: readonly Document[];
  /** A Document's row, at a depth: 0 at the top level. */
  renderDocument(item: Document, depth: number): ReactNode;
}

/**
 * The sidebar's Linked folders (ADR-0010): each Linked folder's row, with
 * its state, folded at first; unfolded, its folders as they are on disk. The
 * files added on their own are in the Folders and Not in a Folder only. A
 * Document is one step deeper than its Folder. A Linked folder shown flat lists all its
 * Documents right under it. Folders follow the disk, so there is nothing to
 * create, rename or move here; clicking one folds or unfolds it.
 *
 * A Linked folder's own row says how it is (indexing, paused, unavailable,
 * online-only files skipped, nothing in it) at its end, on its one line, and
 * offers its menu (see `ObjectMenu`).
 */
export function FolderTree({ documents, renderDocument }: FolderTreeProps) {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const linkedFolders = useAppStore((state) => state.linkedFolders);
  const collapsed = useFolderTree((state) => state.collapsed);
  const opened = useFolderTree((state) => state.opened);
  const tree = useMemo(() => buildFolderTree(folders), [folders]);
  const language = useLanguage();
  const choices = useSort((state) => state.scopes);
  const forAll = useSort((state) => state.all);
  const sortOf = useCallback(
    (key: string) => choices[`folder:${key}`] ?? forAll,
    [choices, forAll],
  );

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
    // Each in its Folder's order (the Other Documents', under its own key).
    for (const [key, list] of grouped) {
      grouped.set(key, [
        ...sorted(list, sortOf(key ?? OTHER_DOCUMENTS), documentFields(t), language),
      ]);
    }
    return grouped;
  }, [documents, folders, flatRoots, sortOf, language, t]);

  /** A Folder's sub-Folders, none in a Linked folder shown flat. */
  const subfoldersOf = (node: FolderNode) =>
    flatRoots.has(node.folder.linkedFolderId)
      ? []
      : sorted(
          node.children,
          sortOf(node.folder.id),
          (child) => folderFields(child.folder),
          language,
        );

  const renderLevel = (nodes: readonly FolderNode[]): ReactNode =>
    nodes.map((node) => {
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
            expanded={opened.has(folder.id)}
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
          expanded={!collapsed.has(folder.id)}
          expandable={hasContents}
          data={{ "data-folder-id": folder.id, "data-linked-folder-id": folder.linkedFolderId }}
        >
          {contents}
        </GroupItem>
      );
    });

  if (tree.length === 0) return null;
  return (
    <ul role="none" data-testid="document-tree">
      {renderLevel(
        sorted(tree, choices.sidebar ?? forAll, (node) => folderFields(node.folder), language),
      )}
    </ul>
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
  children,
}: {
  linked: LinkedFolder;
  folderId: string;
  name: string;
  depth: number;
  hasContents: boolean;
  expanded: boolean;
  children: ReactNode;
}) {
  const t = useT();
  const language = useLanguage();
  const state = linkedFolderRowState(linked, hasContents);
  const label = rowStateLabel(state, t, language);
  const isExample = useAppStore((current) =>
    allExamples(current).some((each) => each.linkedFolderId === linked.id),
  );
  const justLinked = useRequests((current) => current.showLinked === linked.id);
  // Its row, once it shows: in sight, washed for a moment, then as any other (#763).
  useEffect(() => {
    if (!justLinked) return;
    document
      .querySelector(`[data-linked-folder-id="${linked.id}"][data-root]`)
      ?.scrollIntoView({ block: "nearest" });
    const timer = setTimeout(() => useRequests.getState().clear("showLinked"), LINKED_WASH_MS);
    return () => clearTimeout(timer);
  }, [justLinked, linked.id]);
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
      object={{ kind: "linkedFolder", id: linked.id }}
      onToggle={() => useFolderTree.getState().toggleOpened(folderId)}
      actions={
        <ObjectMenu
          target={{ kind: "linkedFolder", id: linked.id }}
          label={t("linkedFolders.more", { name })}
          tabIndex={-1}
        />
      }
      data={{
        "data-folder-id": folderId,
        "data-linked-folder-id": linked.id,
        "data-root": "true",
        "data-state": state.kind,
        "data-status": linked.status,
        "data-layout": linked.layout,
        ...(justLinked && { "data-just-linked": "true" }),
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
  /** The object its `actions` menu is of, so a right-click on the row opens it. */
  object?: ObjectRef;
  /** After its name, unless a status takes the room: the examples' Linked folder says "Example". */
  chip?: ReactNode;
  /** Folds or unfolds it, where that isn't `useFolderTree`'s `toggle` (a Linked folder's row). */
  onToggle?(): void;
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
  const { muted = false, status, progress = null, paused = false, actions, object, chip } = props;
  const { onToggle } = props;
  const { data } = props;
  const children = props.children;
  const toggle = useFolderTree((state) => state.toggle);
  const objectMenu = useObjectMenu(object ?? { kind: "folder", id: groupKey });
  const open = expandable && expanded;
  const treeItem = useTreeItem({ depth, expanded: expandable ? open : undefined });
  const Chevron = open ? ChevronDownLineIcon : ChevronRightLineIcon;
  const buttonClass = actions ? `${rowButtonClass} ${clearsActions.menu}` : rowButtonClass;
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
    <li role="none">
      <div
        data-testid={testId}
        data-depth={depth}
        {...data}
        {...(actions ? objectMenu.rowProps : {})}
        className={`${rowClass(false, muted ? "muted" : "item")} @container`}
      >
        {expandable ? (
          <button
            type="button"
            {...treeItem}
            data-testid="folder-toggle"
            title={title ?? name}
            onClick={() => (onToggle ? onToggle() : toggle(groupKey))}
            onKeyDown={foldKeys(open, () => (onToggle ? onToggle() : toggle(groupKey)))}
            className={buttonClass}
            style={rowPadding(depth)}
          >
            {inner}
          </button>
        ) : (
          <div
            {...treeItem}
            tabIndex={-1}
            title={title ?? name}
            className={buttonClass}
            style={rowPadding(depth)}
          >
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
      {open && <ul role="none">{children}</ul>}
    </li>
  );
}
