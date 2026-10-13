/**
 * Searching for the words a User types, in text indexed as Passages are
 * (./documents/keywords): normalised, traditional Chinese read as simplified,
 * split into words with `Intl.Segmenter`. A search looks for every word of
 * the query, the last letters of each still to type; this finds the same
 * words again in a piece of text, to mark them in a short excerpt.
 * Conversations, the words inside Minds and ⌘K's words in Documents share it.
 */
import type { TextRange } from "./api";
import { keywordTerms, keywordText } from "./documents/keywords";

const segmenter = new Intl.Segmenter("zh", { granularity: "word" });

/** How much of a text an excerpt shows before the first word found, and in all, in characters. */
const EXCERPT_BEFORE = 40;
const EXCERPT_LENGTH = 160;

/** Accents off, as the index's tokenizer takes them off (`remove_diacritics 2`). */
const withoutAccents = (word: string) =>
  word
    .normalize("NFD")
    .replace(/\p{Mn}/gu, "")
    .normalize("NFC");

/** The words a search looks for (see `keywordTerms`), each as its tokens, accents off. */
export function searchTerms(query: string): string[][] {
  return keywordTerms(query).map((term) => term.split(" ").map(withoutAccents));
}

/**
 * The FTS5 query for one term: its tokens as a phrase, the last one a prefix, so
 * "cool" finds "cooling" while the User types.
 */
export const termQuery = (term: readonly string[]) => `"${term.join(" ").replaceAll('"', '""')}" *`;

interface Token {
  word: string;
  start: number;
  end: number;
}

/** The words of `text`, folded as the index folds them, each with where it is, read as asked for. */
function* tokensOf(text: string): Generator<Token> {
  for (const segment of segmenter.segment(text)) {
    if (!segment.isWordLike) continue;
    const start = segment.index;
    const end = start + segment.segment.length;
    // Folding can split a word in two ("聯合國大會" reads as "联合国 大会"): both point at it.
    for (const word of keywordText(segment.segment).split(" ")) {
      if (word) yield { word: withoutAccents(word), start, end };
    }
  }
}

/**
 * Where the terms are in `text`, in order, overlapping ones joined. With
 * `within`, only those that end at most that many characters after the first
 * one found starts: the rest of the text isn't read, which an excerpt of a
 * long Passage needs (#240).
 */
export function findTerms(
  text: string,
  terms: readonly (readonly string[])[],
  within = Number.POSITIVE_INFINITY,
): TextRange[] {
  const tokens: Token[] = [];
  const found: TextRange[] = [];
  let first = Number.POSITIVE_INFINITY;
  for (const token of tokensOf(text)) {
    if (token.start > first + within) break;
    tokens.push(token);
    const at = tokens.length - 1;
    for (const term of terms) {
      const from = at - (term.length - 1);
      const matches =
        from >= 0 &&
        term.every((word, index) => {
          const seen = tokens[from + index]?.word ?? "";
          return index === term.length - 1 ? seen.startsWith(word) : seen === word;
        });
      const start = tokens[from]?.start;
      if (!matches || start === undefined) continue;
      found.push({ start, end: token.end });
      first = Math.min(first, start);
    }
  }
  found.sort((a, b) => a.start - b.start || a.end - b.end);
  const joined: TextRange[] = [];
  for (const range of found) {
    const previous = joined.at(-1);
    if (previous && range.start <= previous.end) previous.end = Math.max(previous.end, range.end);
    else joined.push({ ...range });
  }
  return joined;
}

/** Moves an offset off the second half of a surrogate pair, so no character is cut. */
function wholeCharacter(text: string, at: number): number {
  const code = text.charCodeAt(at);
  return code >= 0xdc00 && code <= 0xdfff ? at - 1 : at;
}

/**
 * A short excerpt of plain text around the first word found, with "…" where it
 * is cut, and where the words found are in the excerpt.
 */
export function excerptOf(
  text: string,
  terms: readonly (readonly string[])[],
): { excerpt: string; highlights: TextRange[] } {
  const ranges = findTerms(text, terms, EXCERPT_LENGTH);
  const first = ranges[0]?.start ?? 0;
  let start = first <= EXCERPT_BEFORE ? 0 : wholeCharacter(text, first - EXCERPT_BEFORE);
  // Not in the middle of a word, where words are split by spaces.
  const space = start > 0 ? text.slice(start, first).search(/\s/) : -1;
  if (space !== -1) start += space + 1;
  const end =
    start + EXCERPT_LENGTH >= text.length
      ? text.length
      : wholeCharacter(text, start + EXCERPT_LENGTH);
  const before = start > 0 ? "…" : "";
  const excerpt = `${before}${text.slice(start, end)}${end < text.length ? "…" : ""}`;
  const shift = before.length - start;
  const highlights = ranges
    .filter((range) => range.start >= start && range.end <= end)
    .map((range) => ({ start: range.start + shift, end: range.end + shift }));
  return { excerpt, highlights };
}
