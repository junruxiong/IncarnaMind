/**
 * A quote as a person reads it (#905). The Citation check matches a quote against a PDF's text
 * with its line breaks and the hyphens that split words at them, which the normaliser can only
 * guess at (see `normaliseText`); what a card shows is the words themselves: a hyphen at a line
 * end that split a word is gone ("em-\nploy" reads "employ"), a quote that stops at one runs to
 * the end of the word ("exist-" reads "existence"), and a real hyphen stays ("long-\nterm" reads
 * "long-term" when the page writes it so elsewhere). Shown only; matching is unchanged.
 */
import type { TextRange } from "./quoteMatch";
import { isCjk } from "./text";

const SOFT_HYPHEN = "\u00ad";
/** The hyphens that may split a word: a dash never does. */
const HYPHENS: ReadonlySet<string> = new Set(["-", "\u2010", "\u2011"]);

/** A hyphen or dash that ends a line, with the break and the indent after it, up to the next letter or digit. */
const LINE_END_HYPHEN = /([-\u2010-\u2015\u00ad])[^\S\n]*\r?\n[^\S\n]*(?=[\p{L}\p{N}])/gu;
/** A hyphen of the first kind, for where a range stops. */
const ANY_HYPHEN = /[-\u2010\u2011\u00ad]/u;
/** What a word is made of, for the hyphenated words around a break. */
const WORD_CHARACTER = /[\p{L}\p{N}'’-]/u;
const LETTER = /\p{L}/u;
const UPPERCASE = /\p{Lu}/u;
const LOWERCASE = /\p{Ll}/u;
/** The break after a hyphen, and the word it carries on with. */
const CONTINUES = /^[^\S\n]*\r?\n[^\S\n]*(\p{L}[\p{L}\p{N}'’]*)/u;

/** A letter of a script written with spaces, the only kind a hyphen splits. */
const isSpacedLetter = (character: string | undefined) =>
  character !== undefined && LETTER.test(character) && !isCjk(character);

/** The word that ends at `at`, with its own hyphens. */
function wordBefore(text: string, at: number): string {
  let start = at;
  while (start > 0 && WORD_CHARACTER.test(text[start - 1] as string)) start--;
  return text.slice(start, at);
}

/** The word that starts at `at`, with its own hyphens. */
function wordAfter(text: string, at: number): string {
  let end = at;
  while (end < text.length && WORD_CHARACTER.test(text[end] as string)) end++;
  return text.slice(at, end);
}

const escapePattern = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");

/** Whether `evidence` has `word` on its own, ignoring case. */
const hasWord = (evidence: string, word: string) =>
  new RegExp(`(?<![\\p{L}\\p{N}])${escapePattern(word)}(?![\\p{L}\\p{N}])`, "iu").test(evidence);

/**
 * Whether the hyphen ending this line is a real one: the same rule as matching (a word either side
 * of the break that has a hyphen of its own, or a small letter before a capital), then what the
 * page itself writes elsewhere, whole and unbroken: "longterm" joins it, "long-term" keeps it.
 */
function isRealHyphen(before: string, after: string, evidence: string): boolean {
  if (before.includes("-") || after.includes("-")) return true;
  if (LOWERCASE.test(before.at(-1) ?? "") && UPPERCASE.test(after[0] ?? "")) return true;
  if (hasWord(evidence, `${before}${after}`)) return false;
  return hasWord(evidence, `${before}-${after}`);
}

/** `raw` on one line, its split words whole. `evidence` is the text it is from, to tell a real hyphen by. */
function read(raw: string, evidence: string): string {
  const joined = raw.replace(LINE_END_HYPHEN, (match, hyphen: string, offset: number) => {
    if (hyphen === SOFT_HYPHEN) return "";
    const letters = isSpacedLetter(raw[offset - 1]) && isSpacedLetter(raw[offset + match.length]);
    if (letters && HYPHENS.has(hyphen)) {
      const before = wordBefore(raw, offset);
      const after = wordAfter(raw, offset + match.length);
      return isRealHyphen(before, after, evidence) ? hyphen : "";
    }
    // A dash, or a hyphen before a digit, stays with the break gone ("COVID-19", "1990–1995").
    return /\s/u.test(raw[offset - 1] ?? " ") ? match : hyphen;
  });
  // Chinese and Japanese put no space between words: a line break between them leaves none.
  return joined
    .replace(/\s+/gu, (run, offset: number) => {
      const before = joined[offset - 1];
      const after = joined[offset + run.length];
      return (before !== undefined && isCjk(before)) || (after !== undefined && isCjk(after))
        ? ""
        : " ";
    })
    .trim();
}

/** Where a range ends, run on to the end of the word it stops inside when that word was split at a line end. */
function wholeWord(text: string, { end }: TextRange): number {
  const stopsAtHyphen = ANY_HYPHEN.test(text[end - 1] ?? "") && isSpacedLetter(text[end - 2]);
  const hyphenFollows = isSpacedLetter(text[end - 1]) && ANY_HYPHEN.test(text[end] ?? "");
  if (!stopsAtHyphen && !hyphenFollows) return end;
  const from = end + (hyphenFollows ? 1 : 0);
  const next = CONTINUES.exec(text.slice(from));
  return next ? from + next[0].length : end;
}

/**
 * The quote found at `ranges` of a text (one for each part of a quote with an ellipsis, see
 * `findQuote`), read as words, and a part that stops at a split word run to its end.
 */
export function readableQuote(text: string, ranges: readonly TextRange[]): string {
  return ranges
    .map((range) => read(text.slice(range.start, wholeWord(text, range)), text))
    .join(" … ");
}

/** A quote read as words with nothing else to go by: a break in it is joined unless the hyphens are real by the rule. */
export const readableQuoteText = (quote: string): string => read(quote, "");
