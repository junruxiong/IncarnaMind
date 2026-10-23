import type { CitationAttributes } from "../core/api";
import { isStopword, keywordText } from "../core/documents/keywords";

/** What a Citation, or the Passage a check found, points at. */
export type CitedPlace = Pick<CitationAttributes, "documentId" | "pageFrom" | "pageTo" | "quote">;

function wordsOf(quote: string): string[] {
  // The keyword index's words, so Chinese splits as it does there and punctuation, width and
  // traditional characters never tell two quotes apart.
  const words = keywordText(quote).split(" ").filter(Boolean);
  const content = words.filter((word) => !isStopword(word));
  return content.length > 0 ? content : words;
}

function sharedInOrder(short: readonly string[], long: readonly string[]): number {
  let row = new Array<number>(long.length + 1).fill(0);
  for (const word of short) {
    const next = [0];
    for (let at = 1; at <= long.length; at++) {
      next[at] =
        word === long[at - 1] ? (row[at - 1] ?? 0) + 1 : Math.max(row[at] ?? 0, next[at - 1] ?? 0);
    }
    row = next;
  }
  return row[long.length] ?? 0;
}

const pagesMeet = (a: CitedPlace, b: CitedPlace): boolean =>
  a.pageFrom !== null &&
  b.pageFrom !== null &&
  a.pageFrom <= (b.pageTo ?? b.pageFrom) &&
  b.pageFrom <= (a.pageTo ?? a.pageFrom);

/**
 * Whether a Citation and the Passage a check found are the same words of the
 * same Document: one quote inside the other, or at least half the words of the
 * shorter in the longer, in order, whatever span each quotes. It compares place
 * and words, never meaning. Where it can't tell, because a quote has no words,
 * the pages meeting decides, so a doubtful case counts as already cited.
 */
export function overlapsCitation(cited: CitedPlace, found: CitedPlace): boolean {
  if (!cited.documentId || cited.documentId !== found.documentId) return false;
  const a = wordsOf(cited.quote ?? "");
  const b = wordsOf(found.quote ?? "");
  if (a.length === 0 || b.length === 0) return pagesMeet(cited, found);
  const [short, long] = a.length <= b.length ? [a, b] : [b, a];
  return sharedInOrder(short, long) * 2 >= short.length;
}
