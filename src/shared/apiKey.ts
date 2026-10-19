const EDGES = /^[\s\p{Cc}\p{Cf}]+|[\s\p{Cc}\p{Cf}]+$/gu;
const NOT_KEY = /[^\x21-\x7e]/u;
const SEEN = /^[\p{L}\p{M}\p{N}\p{P}\p{S}]$/u;

/** What a pasted key has around it: spaces, line breaks, and characters that show nothing (zero-width, BOM). */
export function cleanApiKey(text: string): string {
  return text.replace(EDGES, "");
}

export interface StrayCharacter {
  /** Where it is in the cleaned key, in UTF-16 code units. */
  index: number;
  character: string;
}

/** The first character of a cleaned key that keys don't use: they are made of printable ASCII. */
export function strayKeyCharacter(key: string): StrayCharacter | null {
  const found = NOT_KEY.exec(key);
  return found ? { index: found.index, character: found[0] } : null;
}

/** The character as a line names it: itself when it can be seen, else its code ("U+200B"). */
export function nameCharacter(character: string): string {
  if (SEEN.test(character)) return character;
  const code = character.codePointAt(0) ?? 0;
  return `U+${code.toString(16).toUpperCase().padStart(4, "0")}`;
}

/** A key as typed or pasted: cleaned, and the stray character it still holds, if any. */
export function checkApiKey(text: string): { key: string; stray: StrayCharacter | null } {
  const key = cleanApiKey(text);
  return { key, stray: strayKeyCharacter(key) };
}
