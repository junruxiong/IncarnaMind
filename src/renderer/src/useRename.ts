import { type KeyboardEvent, useCallback, useEffect, useId, useRef, useState } from "react";

/** Two clicks this close in time and place are a double-click, whichever row is under the second. */
const DOUBLE_CLICK_MS = 500;
const DOUBLE_CLICK_PX = 4;

interface Starter {
  start(): void;
  /** Settles what the first click did before renaming: a Folder folds back, a preview is kept. */
  settleClick?(): void;
}

/** Each row on screen, and the latest row for what it stands for: the first click may have remounted it. */
const starters = new Map<string, Starter>();
const latest = new Map<string, Starter>();
let lastClick: { key: string; what: string; at: number; x: number; y: number } | null = null;
let listening = 0;

/**
 * Sees every click before the page does, so a second click that lands on
 * something else (the list moved) still renames the row the first one hit.
 */
function countDoubleClick(event: MouseEvent): void {
  // A key (Enter on a focused row) clicks too, at no place: it never starts a rename.
  if (event.detail === 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
    lastClick = null;
    return;
  }
  const now = Date.now();
  const before = lastClick;
  const row = event.target instanceof Element ? event.target.closest("[data-renamable]") : null;
  const key = row?.getAttribute("data-renamable");
  const what = row?.getAttribute("data-renamable-what");
  lastClick = key && what ? { key, what, at: now, x: event.clientX, y: event.clientY } : null;
  if (
    !before ||
    event.detail < 2 ||
    now - before.at > DOUBLE_CLICK_MS ||
    Math.hypot(event.clientX - before.x, event.clientY - before.y) > DOUBLE_CLICK_PX
  ) {
    return;
  }
  const first = starters.get(before.key) ?? latest.get(before.what);
  lastClick = null;
  if (!first) return;
  // The second click is the rename's, not a second open.
  event.preventDefault();
  event.stopPropagation();
  first.settleClick?.();
  first.start();
}

/**
 * Renaming a name where it is shown: a double-click, F2 on the row, or the
 * menu's Rename (`start`) start it, and `field` goes to `InlineName`.
 *
 * A click opens, so a double-click opens first and then renames. Opening can
 * move the list under the pointer, so the second click is judged by time and
 * place, not by what it lands on: it renames the row the first click hit.
 * `nameProps` goes on the row's button, which takes the focus back afterwards;
 * `what` says what the row stands for ("document:ID"), to find it again if
 * opening drew it anew.
 */
export function useRename<E extends HTMLElement>(what: string, settleClick?: () => void) {
  const [renaming, setRenaming] = useState(false);
  const main = useRef<E>(null);
  // One per row on screen: the same Document can be in the sidebar and the Library at once.
  const key = useId();
  const settle = useRef(settleClick);
  settle.current = settleClick;

  const start = useCallback(() => setRenaming(true), []);
  const stop = useCallback(() => {
    setRenaming(false);
    // Back on its row, so the keyboard carries on from there, unless it has gone elsewhere since.
    requestAnimationFrame(() => {
      if (document.activeElement === document.body) main.current?.focus();
    });
  }, []);

  useEffect(() => {
    const starter = { start, settleClick: () => settle.current?.() };
    starters.set(key, starter);
    latest.set(what, starter);
    if (listening++ === 0) document.addEventListener("click", countDoubleClick, true);
    return () => {
      starters.delete(key);
      if (latest.get(what) === starter) latest.delete(what);
      if (--listening === 0) document.removeEventListener("click", countDoubleClick, true);
    };
  }, [key, what, start]);

  return {
    renaming,
    start,
    stop,
    /** For the row's button. */
    nameProps: {
      ref: main,
      "data-renamable": key,
      "data-renamable-what": what,
      onKeyDown(event: KeyboardEvent<E>) {
        if (event.key !== "F2" || event.nativeEvent.isComposing) return;
        event.preventDefault();
        start();
      },
    },
    /** For `InlineName`: leaves the field, and saves a name by `commit`. */
    field: (commit: (name: string) => void | Promise<void>) => ({
      onCommit(name: string) {
        stop();
        return commit(name);
      },
      onCancel: stop,
    }),
  };
}
