/**
 * Searching conversations (ADR-0017): their names and messages are indexed as
 * Passages are (../documents/keywords): normalised, traditional Chinese read
 * as simplified, split into words with `Intl.Segmenter`. A search looks for
 * every word of the query, the last letters of each still to type; this finds
 * the same words again in a message's text, to mark them in a short excerpt.
 */
import type { TextRange } from "../api";
import { keywordTerms, keywordText } from "../documents/keywords";

const segmenter = new Intl.Segmenter("zh", { granularity: "word" });

/** How much of a message an excerpt shows before the first word found, and in all, in characters. */
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

/**
 * A message's text as plain text, for the index and for excerpts: its Citation
 * markers and the Markdown around words taken out, its lines joined by spaces.
 */
export function plainText(markdown: string): string {
  return markdown
    .replace(/[ \t]?\[\^\d{1,4}\]/g, "")
    .split(/\r?\n/)
    .map((line) =>
      line
        .replace(/^\s*(?:#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+|```\S*|\$\$)/, "")
        .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
        .replace(/\*\*|__|~~|`/g, ""),
    )
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

/** What the index stores for a piece of text: its words (see `keywordText`). */
export const indexText = (text: string) => keywordText(text);

interface Token {
  word: string;
  start: number;
  end: number;
}

/** The words of `text`, folded as the index folds them, each with where it is. */
function tokensOf(text: string): Token[] {
  const tokens: Token[] = [];
  for (const segment of segmenter.segment(text)) {
    if (!segment.isWordLike) continue;
    const start = segment.index;
    const end = start + segment.segment.length;
    // Folding can split a word in two ("聯合國大會" reads as "联合国 大会"): both point at it.
    for (const word of keywordText(segment.segment).split(" ")) {
      if (word) tokens.push({ word: withoutAccents(word), start, end });
    }
  }
  return tokens;
}

/** Where the terms are in `text`, in order, overlapping ones joined. */
export function findTerms(text: string, terms: readonly (readonly string[])[]): TextRange[] {
  const tokens = tokensOf(text);
  const found: TextRange[] = [];
  for (const term of terms) {
    const last = term.length - 1;
    for (let at = 0; at + last < tokens.length; at++) {
      const matches = term.every((word, index) => {
        const token = tokens[at + index]?.word ?? "";
        return index === last ? token.startsWith(word) : token === word;
      });
      const first = tokens[at];
      const end = tokens[at + last];
      if (matches && first && end) found.push({ start: first.start, end: end.end });
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
  const ranges = findTerms(text, terms);
  const first = ranges[0]?.start ?? 0;
  const start = first <= EXCERPT_BEFORE ? 0 : wholeCharacter(text, first - EXCERPT_BEFORE);
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
