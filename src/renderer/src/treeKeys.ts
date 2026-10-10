import type { KeyboardEvent } from "react";

/*
 * The keys of the sidebar's tree (docs/agents/interaction.md, rules 1, 2 and
 * 7), as Finder has them: Enter or F2 renames, ⇧⌘M (Ctrl+Shift+M) is
 * "Move to…", and the arrow keys move through the rows.
 */

const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

/** "Move to…", as this system writes it, for menus. */
export const MOVE_TO_SHORTCUT = isMac ? "⇧⌘M" : "Ctrl+Shift+M";

/** Rename, as this system writes it, for menus: Return in Finder, F2 in Explorer. */
export const RENAME_SHORTCUT = isMac ? "↩" : "F2";

/** Whether a key is "Move to…". */
export const isMoveToKey = (event: KeyboardEvent) =>
  event.code === "KeyM" &&
  event.shiftKey &&
  !event.altKey &&
  (isMac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey);

/** Marks a row's main button as a stop for the arrow keys (`moveThroughRows`). */
export const TREE_ROW = { "data-tree-row": "" } as const;

/**
 * ↑ and ↓ (and Home, End) on a row move to the row above or below, through
 * Folders, Minds and Documents alike. True if the key was used.
 */
export function moveThroughRows(event: KeyboardEvent<HTMLElement>): boolean {
  if (event.altKey || event.metaKey || event.ctrlKey || event.shiftKey) return false;
  if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return false;
  const target = event.target as HTMLElement;
  if (!target.matches("[data-tree-row]")) return false;
  const rows = Array.from(event.currentTarget.querySelectorAll<HTMLElement>("[data-tree-row]"));
  const at = rows.indexOf(target);
  const next =
    event.key === "Home"
      ? rows[0]
      : event.key === "End"
        ? rows.at(-1)
        : rows[at + (event.key === "ArrowDown" ? 1 : -1)];
  if (!next) return false;
  event.preventDefault();
  next.focus();
  return true;
}
