import { type ReactNode, useState } from "react";
import type { Document } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { ChevronDownLineIcon, ChevronRightLineIcon, FolderLineIcon } from "./lineIcons";
import { rowButtonClass, rowClass } from "./sidebarRows";

/** These folders are index entries. Linked source locations remain a separate view. */
export function LibraryFolders({
  documents,
  renderDocument,
}: {
  documents: readonly Document[];
  renderDocument(document: Document, depth: number): ReactNode;
}) {
  const t = useT();
  const library = useAppStore((state) => state.library);
  const active = useAppStore((state) => (state.libraryOpen ? state.libraryFilter : null));
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const assignments = new Map(library?.assignments.map((item) => [item.documentId, item.groupId]));
  const folders = [...(library?.groups ?? []), { id: "unsorted", name: t("library.unsorted") }];
  return (
    <nav aria-label={t("library.groups")} data-testid="library-folders">
      <ul>
        {folders.map((folder) => {
          const contents = documents.filter(
            (doc) => (assignments.get(doc.id) ?? "unsorted") === folder.id,
          );
          const folded = collapsed.has(folder.id);
          const Chevron = folded ? ChevronRightLineIcon : ChevronDownLineIcon;
          return (
            <li key={folder.id} data-testid="library-folder" data-folder-id={folder.id}>
              <div className={rowClass(active === folder.id)}>
                <button
                  type="button"
                  aria-current={active === folder.id ? "page" : undefined}
                  className={rowButtonClass}
                  onClick={() => useAppStore.getState().openLibrary(folder.id)}
                >
                  <FolderLineIcon className="size-4 shrink-0" />
                  <span className="min-w-0 flex-1 truncate" title={folder.name}>
                    {folder.name}
                  </span>
                  <span className="pr-2 text-[12px] tabular-nums text-ink-meta">
                    {contents.length}
                  </span>
                </button>
                <button
                  type="button"
                  aria-label={t(folded ? "library.expand" : "library.collapse", {
                    name: folder.name,
                  })}
                  aria-expanded={!folded}
                  className="inline-flex size-6 shrink-0 items-center justify-center rounded-sm text-ink-meta hover:text-ink focus-visible:outline-offset-0"
                  onClick={() =>
                    setCollapsed((previous) => {
                      const next = new Set(previous);
                      if (!next.delete(folder.id)) next.add(folder.id);
                      return next;
                    })
                  }
                >
                  <Chevron className="size-3" />
                </button>
              </div>
              {!folded && (
                <ul>
                  {contents.slice(0, 50).map((doc) => renderDocument(doc, 1))}
                  {contents.length > 50 && (
                    <li>
                      <button
                        type="button"
                        className="ml-8 text-[12px] text-ink-meta hover:underline"
                        onClick={() => useAppStore.getState().openLibrary(folder.id)}
                      >
                        {t("library.showMore")}
                      </button>
                    </li>
                  )}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
