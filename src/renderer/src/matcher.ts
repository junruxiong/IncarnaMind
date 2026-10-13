import type { TextRange } from "../../core/api";
import { SIMPLIFIED } from "../../shared/hanVariants";

/**
 * Matching what the User types against names: Minds, Documents, Folders, Tags
 * and commands. Case and accents don't count, and traditional Chinese reads
 * as simplified, as in search. Every word typed must be in the name; a name
 * that is the words typed, starts with them or has them at the start of its
 * words ranks above one that only holds them.
 */

/** How well a name matches, and where in it: a higher score is a better match. */
export interface NameMatch {
  score: number;
  ranges: TextRange[];
}

interface Folded {
  text: string;
  /** For each UTF-16 unit of `text`, where the character it came from starts and ends in the name. */
  starts: number[];
  ends: number[];
}

const MARKS = /\p{Mn}/gu;
const HAN = /\p{Script=Han}/u;
const WORD_CHARACTER = /[\p{L}\p{N}]/u;

/** Names are folded once, not at every key typed: a Library holds thousands. */
const folds = new Map<string, Folded>();
const MOST_FOLDS = 50_000;

function fold(name: string): Folded {
  const known = folds.get(name);
  if (known) return known;
  let text = "";
  const starts: number[] = [];
  const ends: number[] = [];
  let at = 0;
  for (const character of name) {
    const plain = (SIMPLIFIED.get(character) ?? character)
      .normalize("NFD")
      .replace(MARKS, "")
      .toLowerCase();
    text += plain;
    for (let unit = 0; unit < plain.length; unit++) {
      starts.push(at);
      ends.push(at + character.length);
    }
    at += character.length;
  }
  if (folds.size >= MOST_FOLDS) folds.clear();
  const folded = { text, starts, ends };
  folds.set(name, folded);
  return folded;
}

/** The words typed, folded as names are. */
export const queryWords = (query: string): string[] =>
  fold(query).text.split(/\s+/).filter(Boolean);

/** Whether a word can start at `at`: the name's start, after a space or a sign, or at a Chinese character. */
function startsWord(text: string, at: number): boolean {
  if (at === 0) return true;
  if (HAN.test(text[at] ?? "")) return true;
  return !WORD_CHARACTER.test(text[at - 1] ?? "");
}

/** Where `word` is in `text`: at the start of a word if it is there, else anywhere; -1 if not at all. */
function find(text: string, word: string): { at: number; atStart: boolean } {
  let first = -1;
  for (let at = text.indexOf(word); at !== -1; at = text.indexOf(word, at + 1)) {
    if (startsWord(text, at)) return { at, atStart: true };
    if (first === -1) first = at;
  }
  return { at: first, atStart: false };
}

/** Ranges in order, those that touch or overlap made one. */
export function joinRanges(ranges: TextRange[]): TextRange[] {
  ranges.sort((a, b) => a.start - b.start || a.end - b.end);
  const out: TextRange[] = [];
  for (const range of ranges) {
    const previous = out.at(-1);
    if (previous && range.start <= previous.end) previous.end = Math.max(previous.end, range.end);
    else out.push({ ...range });
  }
  return out;
}

/** How `name` matches the words typed, or null when one of them isn't in it. */
export function matchName(words: readonly string[], name: string): NameMatch | null {
  if (words.length === 0) return null;
  const folded = fold(name);
  const ranges: TextRange[] = [];
  let allAtStarts = true;
  for (const word of words) {
    const { at, atStart } = find(folded.text, word);
    if (at === -1) return null;
    allAtStarts &&= atStart;
    ranges.push({
      start: folded.starts[at] ?? 0,
      end: folded.ends[at + word.length - 1] ?? name.length,
    });
  }
  const whole = words.join(" ");
  const kind =
    folded.text.trim() === whole
      ? 4
      : folded.text.startsWith(words[0] ?? "")
        ? 3
        : allAtStarts
          ? 2
          : 1;
  // Of names that match alike, the shorter is the closer match.
  return { score: kind * 1000 - Math.min(folded.text.length, 999), ranges: joinRanges(ranges) };
}
