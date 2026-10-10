import type { ReactNode } from "react";
import type { Document } from "../../../core/api";
import { useT } from "../i18n";
import { TREE_ROW } from "../treeKeys";
import { ChevronDownLineIcon, ChevronRightLineIcon } from "./lineIcons";
import { rowButtonClass, rowClass } from "./sidebarRows";

/** How many Documents a Tag lists under it in the sidebar; the Library shows them all. */
const LISTED = 50;

/**
 * A Tag's row in the sidebar's Tags view: its mark (its colour), its name,
 * which opens the Library on it, how many Documents it holds and, if any, a
 * chevron that lists them under it (the first 50, then "Show more" opens the
 * Library on the rest). With none, no chevron.
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
          {...TREE_ROW}
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
