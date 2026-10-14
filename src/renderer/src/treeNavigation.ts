import { startsWith } from "./matcher";

/** A row of the sidebar's tree as the keys see it, in the order shown. */
export interface TreeRow {
  /** 1 at the top; what a row holds is one level deeper. */
  level: number;
  /** Open or closed; undefined for a row with nothing to open. */
  expanded: boolean | undefined;
  /** The name as shown: a typed name is matched with its start. */
  name: string;
}

/** The row ← steps out to: the nearest one above that is a level up. */
export function parentOf(rows: readonly TreeRow[], at: number): number | null {
  const level = rows[at]?.level;
  if (level === undefined) return null;
  for (let index = at - 1; index >= 0; index--) {
    if ((rows[index]?.level ?? level) < level) return index;
  }
  return null;
}

/** The row → steps into: the one below, when this row is open and the next is a level deeper. */
export function firstChildOf(rows: readonly TreeRow[], at: number): number | null {
  const row = rows[at];
  const next = rows[at + 1];
  return row?.expanded === true && next !== undefined && next.level > row.level ? at + 1 : null;
}

/** How long the letters typed stay together: a pause this long starts a new name. */
export const TYPE_AHEAD_MS = 1000;

/** The letters typed so far: `key` joins them if it comes within `TYPE_AHEAD_MS` of the last, else it starts again. */
export const typedAfter = (typed: string, key: string, sinceLastKey: number): string =>
  sinceLastKey > TYPE_AHEAD_MS ? key : typed + key;

/**
 * The row a typed name jumps to, or null if none starts with it. One letter
 * looks from the row after this one, wrapping round, so pressing it again goes
 * on to the next; several look from this row, so a name still matching stays.
 * The same letter again and again (when no name starts with that many) goes
 * round the names that start with it.
 */
export function jumpTo(rows: readonly TreeRow[], at: number, typed: string): number | null {
  const letters = Array.from(typed);
  const first = letters[0];
  if (first === undefined) return null;
  const search = (from: number, wanted: string): number | null => {
    for (let step = 0; step < rows.length; step++) {
      const index = (from + step) % rows.length;
      if (startsWith(wanted, rows[index]?.name ?? "")) return index;
    }
    return null;
  };
  if (letters.length === 1) return search(at + 1, first);
  const whole = search(Math.max(at, 0), typed);
  if (whole !== null) return whole;
  return letters.every((letter) => letter === first) ? search(at + 1, first) : null;
}
