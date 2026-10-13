/*
 * Where a peek goes and which row it moves to (see `Peek`). Pure, so the unit
 * tests import it: no store and no page.
 */

export const PEEK_WIDTH = 320;
/** The card's side padding (Tailwind's `px-3.5`): its page well takes the width between. */
export const PEEK_PADDING = 14;
const EDGE_PX = 8;
/** From the sidebar's edge: the card's 8px gap, then 8px inside the card. */
const BESIDE_PX = 16;

type Edges = Pick<DOMRect, "left" | "right" | "top" | "bottom">;

/**
 * Over the card's side, next to the row, in the sidebar; at the Library's top
 * right, over the end of its rows, in the Library. Kept inside the window.
 */
export function placePeek(
  surface: "sidebar" | "library",
  scope: Edges,
  row: Edges,
  size: { width: number; height: number },
  viewport: { width: number; height: number },
): { left: number; top: number } {
  const keep = (value: number, span: number, limit: number) =>
    Math.max(EDGE_PX, Math.min(value, limit - span - EDGE_PX));
  const left =
    surface === "sidebar"
      ? scope.right + BESIDE_PX
      : Math.max(scope.left + BESIDE_PX, scope.right - size.width - BESIDE_PX);
  const top = surface === "sidebar" ? row.top : scope.top + BESIDE_PX;
  return {
    left: keep(left, size.width, viewport.width),
    top: keep(top, size.height, viewport.height),
  };
}

/**
 * The largest file, in bytes, whose first page a peek draws. Every kind but a PDF
 * (which pdf.js streams a page at a time) is read and laid out whole, as the
 * viewer does, which a peek asks of a small file only: a large one shows its icon.
 */
export const DRAWN_UP_TO_BYTES = 2 * 1024 * 1024;

/** Whether a peek draws this Document's first page, or only shows its kind. */
export const peekDraws = (document: { kind: string; size: number }): boolean =>
  document.kind === "pdf" || document.size <= DRAWN_UP_TO_BYTES;

/** The row after (`1`) or before (`-1`) this one in a list, if there is one. */
export function neighbour<T>(rows: readonly T[], from: T, step: 1 | -1): T | undefined {
  const at = rows.indexOf(from);
  return at === -1 ? undefined : rows[at + step];
}
