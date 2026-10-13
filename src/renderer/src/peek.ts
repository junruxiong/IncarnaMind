import { type KeyboardEvent, useMemo } from "react";
import { create } from "zustand";
import { neighbour } from "./peekPlace";
import { useSelection } from "./selection";
import { refOfKey, type SelectSurface } from "./selectionModel";
import { useAppStore } from "./store";

/*
 * Peek: Space on a Document's row shows a quick look at it beside the list
 * (see `Peek`), without opening the viewer. A row asks for it with
 * `usePeekKey`; `Peek`, mounted once, draws it and takes the keys while it is open.
 */

/** A row's own button, which takes the keys: its name. */
export const ROW_BUTTON = "[data-tree-row], [data-row-main]";

const documentRows = (scope: ParentNode) =>
  Array.from(scope.querySelectorAll<HTMLElement>('[data-select-ref^="document:"]'));

export interface PeekTarget {
  documentId: string;
  surface: SelectSurface;
  /** The row it is about: it shows beside it, and Open clicks it. */
  row: HTMLElement;
}

/** The sidebar or the Library, whichever holds the element. */
export const scopeOf = (element: Element): HTMLElement | null => element.closest("[data-surface]");

export const usePeek = create<{
  target: PeekTarget | null;
  show(target: PeekTarget): void;
  close(): void;
}>()((set) => ({
  target: null,
  show: (target) => set({ target }),
  close: () => set({ target: null }),
}));

/** Shows the Document's peek, or puts it away if it is showing it. */
export function togglePeek(target: PeekTarget): void {
  const { target: shown, show, close } = usePeek.getState();
  if (shown?.documentId === target.documentId && shown.surface === target.surface) close();
  else show(target);
}

const rowOfDocument = (documentId: string, surface: SelectSurface) =>
  document.querySelector<HTMLElement>(
    `[data-surface="${surface}"] [data-select-ref="document:${CSS.escape(documentId)}"]`,
  );

/**
 * The row the peek is about, found again if its list drew it anew (moving the
 * Document to another Folder does); null if the list no longer shows it.
 */
export function peekedRow(target: PeekTarget): HTMLElement | null {
  if (target.row.isConnected) return target.row;
  const row = rowOfDocument(target.documentId, target.surface);
  if (row) usePeek.getState().show({ ...target, row });
  return row;
}

/**
 * Peeks at a Document by its row in the sidebar or the Library, if the row is on
 * the page, with the focus on the row so that the keys of the peek reach it.
 */
export function peekAt(documentId: string, surface: SelectSurface): void {
  const row = rowOfDocument(documentId, surface);
  if (!row) return;
  togglePeek({ documentId, surface, row });
  row.querySelector<HTMLElement>(ROW_BUTTON)?.focus();
}

/** The key handler of a Document's row: Space on its name peeks at the Document. */
export function peekRowProps(documentId: string, surface: SelectSurface) {
  return {
    onKeyDownCapture(event: KeyboardEvent<HTMLElement>) {
      if (event.key !== " " || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.shiftKey || event.nativeEvent.isComposing) return;
      if (!(event.target instanceof Element) || !event.target.matches(ROW_BUTTON)) return;
      event.preventDefault();
      if (!event.repeat) togglePeek({ documentId, surface, row: event.currentTarget });
    },
  };
}

/**
 * For a Document's row: Space on its name peeks at it, and doesn't open the
 * viewer as a button would. `rowProps` go on the row's own element, the one
 * with `useSelectable`'s props; the other buttons in the row keep Space.
 */
export function usePeekKey(documentId: string, surface: SelectSurface) {
  return useMemo(() => ({ rowProps: peekRowProps(documentId, surface) }), [documentId, surface]);
}

/**
 * ↑ and ↓ while a peek is open: the focus goes to the next or previous
 * Document row of the list it is in, which the selection follows, and the
 * peek follows the focus. Folders and Minds between them are passed over.
 */
export function stepPeek(step: 1 | -1): void {
  const { target } = usePeek.getState();
  const row = target && peekedRow(target);
  const scope = row && scopeOf(row);
  if (!target || !row || !scope) return;
  const next = neighbour(documentRows(scope), row, step);
  const id = refOfKey(next?.dataset.selectRef ?? "")?.id;
  if (!next || !id) return;
  useSelection.getState().choose(target.surface, { kind: "document", id });
  next.querySelector<HTMLElement>(ROW_BUTTON)?.focus();
}

/** Opens the Document as a click on its row does (beside the Mind, or beside the Library), and puts the peek away. */
export function openPeeked(): void {
  const { target, close } = usePeek.getState();
  if (!target) return;
  const button = peekedRow(target)?.querySelector<HTMLElement>(ROW_BUTTON);
  close();
  if (button) button.click();
  else useAppStore.getState().openDocument({ documentId: target.documentId });
}

/** The focus back on the peeked Document's row. */
export function focusPeekedRow(): void {
  const { target } = usePeek.getState();
  const row = target && peekedRow(target);
  row?.querySelector<HTMLElement>(ROW_BUTTON)?.focus();
}
