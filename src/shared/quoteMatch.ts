/**
 * Finding a quote in a Document's text, so the viewer can highlight it. The
 * Citation check (#30) uses the same matching.
 *
 * Both texts are normalised with the shared normaliser (`normaliseText` in
 * ./text: NFKC, CJK radical look-alikes folded, quote marks and dashes
 * unified, line-break hyphenation handled, whitespace next to CJK removed and
 * the rest collapsed), then matched as an exact substring, so line breaks and
 * PDF layout don't stop a match. One tolerance: a hyphen that ended a line
 * between two letters may be in the quote or not ("inter-\nnational" matches
 * both "international" and "inter-national"), because the normaliser can only
 * guess whether it was a real hyphen.
 */
import { type NormalisedUnit, normaliseText, normaliseWithOffsets } from "./text";

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

/** The index of the last unit of a match of `needle` starting at `first`, or -1. */
function matchAt(units: readonly NormalisedUnit[], needle: string, first: number): number {
  let at = first;
  let last = -1;
  for (let index = 0; index < needle.length; ) {
    const unit = units[at];
    if (!unit) return -1;
    at++;
    if (unit.optional && needle[index] !== "-") continue; // skip a line-end hyphen
    if (!unit.optional && unit.removed) continue;
    if (unit.char !== needle[index]) return -1;
    last = at - 1;
    index++;
  }
  return last;
}

/** Where `quote` first appears in `text` after both are normalised, or null if it doesn't. */
export function findQuote(text: string, quote: string): TextRange | null {
  const needle = normaliseText(quote);
  if (!needle) return null;
  const { units } = normaliseWithOffsets(text);
  for (let first = 0; first < units.length; first++) {
    const unit = units[first] as NormalisedUnit;
    if (unit.removed || unit.char !== needle[0]) continue;
    const last = matchAt(units, needle, first);
    if (last >= 0) return { start: unit.start, end: (units[last] as NormalisedUnit).end };
  }
  return null;
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
