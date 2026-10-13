import type { KeyboardEvent } from "react";
import { pressed } from "./commands/keys";

/*
 * The keys of the sidebar's tree (docs/agents/interaction.md, rules 1, 2 and
 * 7): Enter opens a row, F2 renames it, ⇧⌘M (Ctrl+Shift+M) is "Move to…",
 * and the arrow keys move through the rows.
 */

const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

/** Whether a key is "Move to…". */
export const isMoveToKey = (event: KeyboardEvent) =>
  event.code === "KeyM" &&
  event.shiftKey &&
  !event.altKey &&
  (isMac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey);

/** A Library row's "Tags…", besides T alone. */
export const TAGS_KEYS = "CmdOrCtrl+Alt+T";

/** Whether a key is a Library row's "Tags…": T, or ⌥⌘T (Ctrl+Alt+T). */
export const isTagsKey = (event: KeyboardEvent) =>
  !event.nativeEvent.isComposing &&
  (pressed("T", event, isMac) || pressed(TAGS_KEYS, event, isMac));

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
