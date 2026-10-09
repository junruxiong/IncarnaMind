/**
 * Tag colours (DESIGN.md, Tags; approved by the User on 2026-10-09): a small
 * fixed palette of muted colours, each a pale fill with dark text that reads
 * at AA (6.6:1 or better). None is green or amber, so a Tag never looks like
 * a Citation check or the amber "needs review" dot, and none is the accent
 * blue. Stored with the Tag as its key; the renderer maps keys to tokens
 * (`--color-tag-*` in styles.css).
 */
export const TAG_COLOURS = [
  "stone",
  "taupe",
  "brick",
  "rose",
  "orchid",
  "violet",
  "indigo",
  "petrol",
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
