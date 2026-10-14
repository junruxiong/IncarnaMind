import type { ReactNode } from "react";
import type { Document } from "../../../core/api";
import { useT } from "../i18n";
import type { ObjectRef } from "../objects";
import { foldKeys, useTreeItem } from "../treeKeys";
import { useRename } from "../useRename";
import { InlineName } from "./InlineName";
import { ChevronDownLineIcon, ChevronRightLineIcon } from "./lineIcons";
import { ObjectMenu, useObjectMenu } from "./ObjectMenu";
import { clearsActions, rowActionsClass, rowButtonClass, rowClass } from "./sidebarRows";

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
  object,
  documents,
  selected,
  expanded,
  onOpen,
  onToggle,
  rename,
  renderDocument,
}: {
  testId: string;
  data?: Record<`data-${string}`, string>;
  /** In the icon column: 16px. */
  icon: ReactNode;
  name: string;
  /** Its name's tooltip, if not its name: a Tag's description. */
  title?: string;
  /** What its menu is of (a Tag), if it has one: a right-click on the row, or its ⋯, opens it. */
  object?: ObjectRef;
  documents: readonly Document[];
  /** Chosen: a Tag the Tag filter has (choosing it again stops filtering by it). */
  selected: boolean;
  expanded: boolean;
  onOpen(): void;
  onToggle(): void;
  /** Renames it in place, if it can be: what the field is called, and the name to keep. */
  rename?: { label: string; commit(name: string): void | Promise<void> };
  renderDocument(document: Document, depth: number): ReactNode;
}) {
  const t = useT();
  const count = documents.length;
  const Chevron = expanded ? ChevronDownLineIcon : ChevronRightLineIcon;
  const objectMenu = useObjectMenu(object ?? { kind: "tag", id: "" });
  // The first click of a double-click chose or cleared it: back as it was before renaming.
  const naming = useRename<HTMLButtonElement>(`${testId}:${object?.id ?? name}`, onOpen);
  const renaming = naming.renaming && rename !== undefined;
  const treeItem = useTreeItem({ depth: 0, expanded: count > 0 ? expanded : undefined, selected });
  const showMoreItem = useTreeItem({ depth: 1 });
  const fold = foldKeys(expanded, onToggle);
  return (
    <li role="none" data-testid={testId} data-count={count} {...data}>
      <div className={rowClass(selected)} {...(object ? objectMenu.rowProps : {})}>
        {renaming ? (
          <span className={`${rowButtonClass} pr-1`}>
            {icon}
            <InlineName value={name} label={rename.label} {...naming.field(rename.commit)} />
          </span>
        ) : (
          <button
            type="button"
            {...(rename ? naming.nameProps : {})}
            {...treeItem}
            className={
              object ? `${rowButtonClass} ${clearsActions.menuBeforeChevron}` : rowButtonClass
            }
            onClick={onOpen}
            onKeyDown={(event) => {
              if (rename) naming.nameProps.onKeyDown(event);
              if (count > 0) fold(event);
            }}
          >
            {icon}
            <span data-testid="row-text" className="min-w-0 flex-1 truncate" title={title}>
              {name}
            </span>
            <span
              data-testid="browse-count"
              className="pr-2 text-[12px] tabular-nums text-ink-meta"
            >
              {count}
            </span>
          </button>
        )}
        {count > 0 ? (
          <button
            type="button"
            aria-label={t(expanded ? "library.collapse" : "library.expand", { name })}
            aria-expanded={expanded}
            tabIndex={-1}
            className="inline-flex size-6 shrink-0 items-center justify-center rounded-sm text-ink-meta hover:text-ink focus-visible:outline-offset-0"
            onClick={onToggle}
          >
            <Chevron className="size-3" />
          </button>
        ) : (
          // Where the chevron would be, so every count lines up.
          <span aria-hidden="true" className="size-6 shrink-0" />
        )}
        {object && !renaming && (
          <div className={rowActionsClass} style={{ right: 29 }}>
            <ObjectMenu
              target={object}
              label={t("tags.menu.of", { name })}
              tabIndex={-1}
              here={rename ? { "tag.rename": naming.start } : undefined}
            />
          </div>
        )}
      </div>
      {expanded && count > 0 && (
        <ul role="none">
          {documents.slice(0, LISTED).map((doc) => renderDocument(doc, 1))}
          {count > LISTED && (
            <li role="none">
              <button
                type="button"
                {...showMoreItem}
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
