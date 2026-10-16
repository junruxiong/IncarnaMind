import { isCjk } from "../../shared/text";

/** How many characters of its name a tab keeps in view before the tab strip scrolls instead of shrinking it (#905). */
export const TAB_LABEL_CHARACTERS = 10;

/** A Latin letter or digit is up to about this wide, in em, at the tab's size; a Chinese or Japanese character is one em. */
const LATIN_EM = 0.7;

/** Left over after the last character, so its edge isn't cut by the ellipsis. */
const EDGE_EM = 0.4;

/**
 * The room, in em, a tab's name needs to show its first `TAB_LABEL_CHARACTERS` characters, or all
 * of it if it is shorter.
 */
export function tabLabelEm(name: string): number {
  let room = 0;
  for (const character of [...name].slice(0, TAB_LABEL_CHARACTERS)) {
    room += isCjk(character) ? 1 : LATIN_EM;
  }
  return room === 0 ? 0 : Math.round((room + EDGE_EM) * 100) / 100;
}
