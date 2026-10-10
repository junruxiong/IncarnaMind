import { type ReactNode, useMemo, useState } from "react";
import type { Document } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { ChevronDownLineIcon, ChevronRightLineIcon, FolderLineIcon } from "./lineIcons";
import { rowButtonClass, rowClass } from "./sidebarRows";

/** How many Documents a Folder or Tag lists under it in the sidebar; the Library shows them all. */
const LISTED = 50;

const NONE: readonly Document[] = [];

/** These folders are index entries. Linked source locations remain a separate view. */
export function LibraryFolders({
  documents,
  renderDocument,
}: {
  documents: readonly Document[];
  renderDocument(document: Document, depth: number): ReactNode;
}) {
  const t = useT();
  const groups = useAppStore((state) => state.library?.groups);
  const assignments = useAppStore((state) => state.library?.assignments);
  const active = useAppStore((state) => (state.libraryOpen ? state.libraryFilter : null));
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  // Each Folder's Documents (Unsorted's under "unsorted"), in one pass, again only when they change.
  const byFolder = useMemo(() => {
    const folderOf = new Map(assignments?.map((item) => [item.documentId, item.groupId]));
    const grouped = new Map<string, Document[]>();
    for (const doc of documents) {
      const key = folderOf.get(doc.id) ?? "unsorted";
      const list = grouped.get(key);
      if (list) list.push(doc);
      else grouped.set(key, [doc]);
    }
    return grouped;
  }, [documents, assignments]);
  const folders = [...(groups ?? []), { id: "unsorted", name: t("library.unsorted") }];
  return (
    <nav aria-label={t("library.groups")} data-testid="library-folders">
      <ul>
        {folders.map((folder) => (
          <BrowseGroup
            key={folder.id}
            testId="library-folder"
            data={{ "data-folder-id": folder.id }}
            icon={<FolderLineIcon className="size-4 shrink-0" />}
            name={folder.name}
            documents={byFolder.get(folder.id) ?? NONE}
            selected={active === folder.id}
            expanded={!collapsed.has(folder.id)}
            onOpen={() => useAppStore.getState().openLibrary(folder.id)}
            onToggle={() =>
              setCollapsed((previous) => {
                const next = new Set(previous);
                if (!next.delete(folder.id)) next.add(folder.id);
                return next;
              })
            }
            renderDocument={renderDocument}
          />
        ))}
      </ul>
    </nav>
  );
}

/**
 * A Folder's or a Tag's row in the sidebar: its mark (a Folder's icon, a
 * Tag's colour), its name, which opens the Library on it, how many Documents
 * it holds and, if any, a chevron that lists them under it (the first 50,
 * then "Show more" opens the Library on the rest). With none, no chevron.
 */
export function BrowseGroup({
  testId,
  data,
  icon,
  name,
  title = name,
  documents,
  selected,
  selectedAs = "current",
  expanded,
  onOpen,
  onToggle,
  renderDocument,
}: {
  testId: string;
  data?: Record<`data-${string}`, string>;
  /** In the icon column: 16px. */
  icon: ReactNode;
  name: string;
  /** Its name's tooltip, if not its name: a Tag's description. */
  title?: string;
  documents: readonly Document[];
  /**
   * Chosen: a Folder the Library shows (`aria-current`), or a Tag the Tag
   * filter has (`aria-pressed`: choosing it again stops filtering by it).
   */
  selected: boolean;
  selectedAs?: "current" | "pressed";
  expanded: boolean;
  onOpen(): void;
  onToggle(): void;
  renderDocument(document: Document, depth: number): ReactNode;
}) {
  const t = useT();
  const count = documents.length;
  const Chevron = expanded ? ChevronDownLineIcon : ChevronRightLineIcon;
  return (
    <li data-testid={testId} data-count={count} {...data}>
      <div className={rowClass(selected)}>
        <button
          type="button"
          aria-current={selectedAs === "current" && selected ? "page" : undefined}
          aria-pressed={selectedAs === "pressed" ? selected : undefined}
          className={rowButtonClass}
          onClick={onOpen}
        >
          {icon}
          <span data-testid="row-text" className="min-w-0 flex-1 truncate" title={title}>
            {name}
          </span>
          <span data-testid="browse-count" className="pr-2 text-[12px] tabular-nums text-ink-meta">
            {count}
          </span>
        </button>
        {count > 0 ? (
          <button
            type="button"
            aria-label={t(expanded ? "library.collapse" : "library.expand", { name })}
            aria-expanded={expanded}
            className="inline-flex size-6 shrink-0 items-center justify-center rounded-sm text-ink-meta hover:text-ink focus-visible:outline-offset-0"
            onClick={onToggle}
          >
            <Chevron className="size-3" />
          </button>
        ) : (
          // Where the chevron would be, so every count lines up.
          <span aria-hidden="true" className="size-6 shrink-0" />
        )}
      </div>
      {expanded && count > 0 && (
        <ul>
          {documents.slice(0, LISTED).map((doc) => renderDocument(doc, 1))}
          {count > LISTED && (
            <li>
              <button
                type="button"
                className="ml-8 text-[12px] text-ink-meta hover:underline"
                onClick={onOpen}
              >
                {t("library.showMore")}
              </button>
            </li>
          )}
        </ul>
      )}
    </li>
  );
}
