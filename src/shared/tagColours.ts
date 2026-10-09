/**
 * Tag colours (DESIGN.md, Tags; the bright, Finder-like palette the User chose
 * on 2026-10-09), in Finder's order. Each is a bright dot, for where a Tag has
 * no room for its name, and a pale fill with dark text (7:1 or better) for its
 * named chip. Stored with the Tag as its key (migration 28 moved the first
 * palette's keys here); the renderer maps keys to tokens (`--color-tag-*` in
 * styles.css).
 */
export const TAG_COLOURS = [
  "red",
  "orange",
  "yellow",
  "green",
  "teal",
  "blue",
  "purple",
  "gray",
] as const;

export type TagColour = (typeof TAG_COLOURS)[number];

export const isTagColour = (value: unknown): value is TagColour =>
  typeof value === "string" && (TAG_COLOURS as readonly string[]).includes(value);

/**
 * The colour a new Tag gets: the one fewest live Tags have, the earlier in
 * the palette on a tie, so Tags made one after another differ.
 */
export function nextTagColour(used: readonly string[]): TagColour {
  const counts = new Map<string, number>();
  for (const colour of used) counts.set(colour, (counts.get(colour) ?? 0) + 1);
  let best: TagColour = TAG_COLOURS[0];
  for (const colour of TAG_COLOURS)
    if ((counts.get(colour) ?? 0) < (counts.get(best) ?? 0)) best = colour;
  return best;
}
