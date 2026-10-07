import { type ReactNode, useMemo, useState } from "react";
import { buildFolderTree, type FolderNode } from "../folders";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { AllDocumentsIcon, ChevronIcon, FolderIcon } from "./icons";

const INDENT_PX = 12;
/** Left padding of a row at `depth`. Rows start with a chevron column, so icons line up. */
const indent = (depth: number) => ({ paddingLeft: 4 + depth * INDENT_PX });
/** The chevron column's width plus the gap after it. */
const CHEVRON_PX = 18;

/**
 * The Documents section's Folder tree: "All Documents", then the Linked
 * folders' Folders, nested as on disk. Clicking one shows only its Documents
 * (sub-Folders included). Folders follow the disk, so there is nothing to
 * create, rename, move or delete here.
 */
export function FolderTree() {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const folderFilter = useAppStore((state) => state.folderFilter);
  const filterByFolder = useAppStore((state) => state.filterByFolder);
  const tree = useMemo(() => buildFolderTree(folders), [folders]);
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set());

  const toggle = (id: string) =>
    setExpanded((open) => {
      const next = new Set(open);
      if (!next.delete(id)) next.add(id);
      return next;
    });

  if (folders.length === 0) return null;

  const renderNodes = (nodes: readonly FolderNode[]): ReactNode => (
    <ul>
      {nodes.map((node) => (
        <FolderItem
          key={node.folder.id}
          node={node}
          expanded={expanded.has(node.folder.id)}
          selected={folderFilter === node.folder.id}
          onToggle={() => toggle(node.folder.id)}
          onSelect={() => void filterByFolder(node.folder.id)}
        >
          {renderNodes(node.children)}
        </FolderItem>
      ))}
    </ul>
  );

  return (
    <nav aria-label={t("folders.label")} className="mx-3 mb-1 border-b border-gray-200 pb-1">
      <AllDocumentsItem
        selected={folderFilter === null}
        onSelect={() => void filterByFolder(null)}
      />
      {renderNodes(tree)}
    </nav>
  );
}

const rowClass = (selected: boolean) =>
  `group my-[1px] flex items-center gap-[2px] rounded-[9px] py-[3px] pr-1 text-sm ${
    selected ? "bg-gray-200" : "hover:bg-gray-100"
  }`;

function AllDocumentsItem({ selected, onSelect }: { selected: boolean; onSelect(): void }) {
  const t = useT();
  return (
    <div data-testid="all-documents" className={rowClass(selected)} style={indent(0)}>
      <button
        type="button"
        aria-current={selected ? "true" : undefined}
        onClick={onSelect}
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
  /** The sub-Folders' list, shown while expanded. */
  children: ReactNode;
}

function FolderItem(props: FolderItemProps) {
  const { node, expanded, selected, onToggle, onSelect, children } = props;
  const { folder, depth } = node;
  const t = useT();
  const hasChildren = node.children.length > 0;

  return (
    <li>
      <div
        data-testid="folder-item"
        data-folder-id={folder.id}
        className={rowClass(selected)}
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
        <button
          type="button"
          data-testid="folder-filter"
          aria-current={selected ? "true" : undefined}
          onClick={onSelect}
          className="flex min-w-0 flex-1 items-center gap-[6px] py-[2px] text-left"
        >
          <FolderIcon className="size-4 shrink-0" />
          <span className="truncate text-gray-700" title={folder.name}>
            {folder.name}
          </span>
        </button>
      </div>
      {expanded && children}
    </li>
  );
}
