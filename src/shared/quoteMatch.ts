/**
 * Finding a quote in a Document's text, so the viewer can highlight it.
 *
 * Both texts are normalised the same way as the Citation check (see
 * docs/designs/v1-product-validation.md, "Matching"), then matched as an exact
 * substring, so line breaks and PDF layout don't stop a match:
 * - Unicode NFKC;
 * - quote marks and dashes unified, invisible characters (soft hyphens,
 *   zero-width spaces) dropped;
 * - end-of-line hyphenation removed ("inter-\nnational");
 * - every run of whitespace collapsed to one space;
 * - whitespace between CJK characters removed, counting CJK punctuation and
 *   full-width forms as CJK ("图像。\n循环" reads as "图像。循环").
 */
import { CJK } from "../core/documents/text";

/** A span of the original text, in UTF-16 offsets; `end` is exclusive. */
export interface TextRange {
  start: number;
  end: number;
}

/** Part of a quote's match inside one piece of a text made of pieces. */
export interface PieceRange extends TextRange {
  /** Index of the piece. */
  piece: number;
}

export interface TextPiece {
  text: string;
  /** Whether a line break follows the piece, e.g. at the end of a PDF text line or page. */
  breakAfter?: boolean;
}

/** One UTF-16 code unit of normalised text, and the span of the original it came from. */
interface Unit {
  char: string;
  start: number;
  end: number;
  /** Whether the original character is CJK, judged before NFKC turns "，" into ",". */
  cjk: boolean;
}

/** CJK scripts, Bopomofo, CJK punctuation and symbols, CJK compatibility forms, full-width forms. */
const CJK_LIKE = new RegExp(
  `[${CJK}\\p{Script=Bopomofo}\\u3000-\\u303f\\ufe30-\\ufe4f\\uff00-\\uffef]`,
  "u",
);

const UNIFIED: Readonly<Record<string, string>> = {
  "‘": "'",
  "’": "'",
  "‚": "'",
  "‛": "'",
  "′": "'",
  "“": '"',
  "”": '"',
  "„": '"',
  "‟": '"',
  "″": '"',
  "‐": "-",
  "‑": "-",
  "‒": "-",
  "–": "-",
  "—": "-",
  "―": "-",
  "−": "-",
};

/** Dropped entirely: NUL, soft hyphen, zero-width space and joiners, word joiner, byte-order mark. */
const INVISIBLE: ReadonlySet<string> = new Set(["\u0000", "­", "​", "‌", "‍", "⁠", "﻿"]);

// A base character with its combining marks, so NFKC sees them together (e.g. "e" + U+0301).
const SEGMENT = /\P{M}\p{M}*|\p{M}+/gsu;
const WHITESPACE = /\s/u;
const LETTER = /\p{L}/u;

function normalisedUnits(text: string): Unit[] {
  const units: Unit[] = [];
  for (const match of text.matchAll(SEGMENT)) {
    const start = match.index;
    const end = start + match[0].length;
    const cjk = CJK_LIKE.test(match[0]);
    const normalised = match[0].normalize("NFKC");
    for (let index = 0; index < normalised.length; index++) {
      const raw = normalised[index] as string;
      if (INVISIBLE.has(raw)) continue;
      units.push({ char: UNIFIED[raw] ?? raw, start, end, cjk });
    }
  }
  return collapseWhitespace(units);
}

function collapseWhitespace(units: readonly Unit[]): Unit[] {
  const out: Unit[] = [];
  let index = 0;
  while (index < units.length) {
    const unit = units[index] as Unit;
    if (!WHITESPACE.test(unit.char)) {
      out.push(unit);
      index++;
      continue;
    }
    let next = index;
    let lineBreak = false;
    while (next < units.length && WHITESPACE.test((units[next] as Unit).char)) {
      if ((units[next] as Unit).char === "\n") lineBreak = true;
      next++;
    }
    const before = out.at(-1);
    const after = units[next];
    // Leading and trailing whitespace is dropped.
    if (before && after) {
      const hyphenated =
        lineBreak &&
        before.char === "-" &&
        LETTER.test(out.at(-2)?.char ?? "") &&
        LETTER.test(after.char);
      if (hyphenated) {
        out.pop();
      } else if (!(before.cjk && after.cjk)) {
        out.push({ char: " ", start: unit.start, end: (units[next - 1] as Unit).end, cjk: false });
      }
    }
    index = next;
  }
  return out;
}

/** Normalises text for matching (see the module comment). */
export function normaliseForMatching(text: string): string {
  return normalisedUnits(text)
    .map((unit) => unit.char)
    .join("");
}

/** Where `quote` first appears in `text` after both are normalised, or null if it doesn't. */
export function findQuote(text: string, quote: string): TextRange | null {
  const needle = normaliseForMatching(quote);
  if (!needle) return null;
  const units = normalisedUnits(text);
  const at = units
    .map((unit) => unit.char)
    .join("")
    .indexOf(needle);
  if (at < 0) return null;
  const first = units[at] as Unit;
  const last = units[at + needle.length - 1] as Unit;
  return { start: first.start, end: last.end };
}

/**
 * Finds `quote` in a text made of pieces (e.g. a PDF page's text runs, page
 * after page) and returns the part of each piece it covers, or null if it isn't there.
 */
export function findQuoteInPieces(
  pieces: readonly TextPiece[],
  quote: string,
): PieceRange[] | null {
  let text = "";
  const offsets: number[] = [];
  for (const piece of pieces) {
    offsets.push(text.length);
    text += piece.text;
    if (piece.breakAfter) text += "\n";
  }
  const range = findQuote(text, quote);
  if (!range) return null;
  const covered: PieceRange[] = [];
  pieces.forEach((piece, index) => {
    const offset = offsets[index] as number;
    const start = Math.max(range.start, offset) - offset;
    const end = Math.min(range.end, offset + piece.text.length) - offset;
    if (start < end) covered.push({ piece: index, start, end });
  });
  return covered;
}
