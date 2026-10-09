import { useEffect, useRef } from "react";
import type { Document } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { TagLineIcon } from "./lineIcons";
import { pickerLabel, TagPicker, tagPopoverClass, useTagPopover } from "./TagPicker";
import { buttonStyle } from "./ui";

/** The Document a Shift-click selects from: the last one clicked. */
let anchor: string | null = null;

const boxClass =
  "size-4 shrink-0 cursor-pointer accent-ink outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

/**
 * A Library row's checkbox, in the margin left of its name: shown when the
 * row is pointed at or focused, and on every row once any is selected.
 * Shift-click selects every row between it and the last one clicked.
 */
export function LibrarySelectBox({
  document,
  shownIds,
}: {
  document: Pick<Document, "id" | "name">;
  /** The rows in view, in order, for Shift-click ranges. */
  shownIds: readonly string[];
}) {
  const t = useT();
  const checked = useAppStore((state) => state.selectedDocuments.has(document.id));
  const selecting = useAppStore((state) => state.selectedDocuments.size > 0);
  return (
    <input
      type="checkbox"
      data-testid="library-select"
      aria-label={t("library.select", { name: document.name })}
      checked={checked}
      onChange={(event) => {
        const range =
          (event.nativeEvent as MouseEvent).shiftKey && anchor !== null
            ? rangeBetween(shownIds, anchor, document.id)
            : [document.id];
        useAppStore.getState().selectDocuments(range, event.target.checked);
        anchor = document.id;
      }}
      className={`${boxClass} absolute top-3.5 -left-[21px] ${
        checked || selecting
          ? ""
          : "opacity-0 group-hover/row:opacity-100 focus-visible:opacity-100"
      }`}
    />
  );
}

/** The ids from one to the other, both included, in the rows' order. */
function rangeBetween(ids: readonly string[], from: string, to: string): string[] {
  const start = ids.indexOf(from);
  const end = ids.indexOf(to);
  if (start === -1 || end === -1) return [to];
  return ids.slice(Math.min(start, end), Math.max(start, end) + 1);
}

/**
 * Above the rows while any in view is selected: select all or none, how
 * many, "Tags…" to add or remove Tags on all of them at once (the Tag
 * picker), and clearing the selection. Esc clears it too.
 */
export function LibrarySelectionBar({ shownIds }: { shownIds: readonly string[] }) {
  const t = useT();
  const selected = useAppStore((state) => state.selectedDocuments);
  const documents = useAppStore((state) => state.documents);
  const popover = useTagPopover();
  const all = useRef<HTMLInputElement>(null);
  const inView = shownIds.filter((id) => selected.has(id));
  const every = inView.length > 0 && inView.length === shownIds.length;
  useEffect(() => {
    if (all.current) all.current.indeterminate = inView.length > 0 && !every;
  });
  // The selection keeps only Documents that still exist.
  useEffect(() => {
    const gone = [...selected].filter((id) => !documents.some((each) => each.id === id));
    if (gone.length) useAppStore.getState().selectDocuments(gone, false);
  }, [documents, selected]);
  if (inView.length === 0) return null;
  const chosen = documents.filter((each) => inView.includes(each.id));
  const label = pickerLabel(t, chosen);
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: Esc anywhere in the bar clears the selection.
    <div
      data-testid="library-selection"
      onKeyDown={(event) => {
        if (event.key === "Escape" && !event.defaultPrevented && !popover.open) {
          event.preventDefault();
          useAppStore.getState().clearSelection();
        }
      }}
      className="sticky top-0 z-10 mb-2 flex h-10 items-center gap-3 rounded-md bg-frame px-2 text-[13px]"
    >
      <input
        ref={all}
        type="checkbox"
        data-testid="library-select-all"
        aria-label={t("library.selectAll", { count: shownIds.length })}
        checked={every}
        onChange={() => useAppStore.getState().selectDocuments(shownIds, !every)}
        className={boxClass}
      />
      <span role="status" className="font-semibold text-ink">
        {t("library.selected", { count: inView.length })}
      </span>
      <button
        {...popover.buttonProps}
        type="button"
        data-testid="library-selection-tags"
        className={buttonStyle("secondary", "sm")}
      >
        <TagLineIcon className="size-3.5" />
        {t("library.selected.tags")}
      </button>
      <div
        {...popover.popoverProps}
        role="dialog"
        aria-label={label}
        data-testid="selection-tags-popover"
        className={tagPopoverClass}
      >
        {popover.open && (
          <TagPicker documentIds={inView} label={label} onReposition={popover.reposition} />
        )}
      </div>
      <button
        type="button"
        data-testid="library-selection-clear"
        onClick={() => useAppStore.getState().clearSelection()}
        className={buttonStyle("ghost", "sm")}
      >
        {t("library.selected.clear")}
      </button>
    </div>
  );
}
