/**
 * How well a page was read (eval/ocr/README.md): its character error rate
 * against the expected text, and the share of quotes from the expected text
 * that the Citation check's own matcher finds in what was read: what a
 * Citation of a scanned page needs.
 */
import { findQuote } from "../../../src/shared/quoteMatch";
import { isCjk, normaliseText } from "../../../src/shared/text";

/** A page of the set (tests/fixtures/scans/set.json). */
export interface SetPage {
  id: string;
  language: string;
  condition: "clean" | "noisy";
  kind: "text" | "table";
  about: string;
  /** Drawn after the reading was tuned on the other pages. */
  heldout?: boolean;
  file: string;
  lines: string[];
}

/** Levenshtein distance between two strings, by code point. */
export function editDistance(a: string, b: string): number {
  const x = [...a];
  const y = [...b];
  let previous = Array.from({ length: y.length + 1 }, (_, index) => index);
  for (let i = 1; i <= x.length; i++) {
    const current = [i];
    for (let j = 1; j <= y.length; j++) {
      const cost = x[i - 1] === y[j - 1] ? 0 : 1;
      current[j] = Math.min(
        (previous[j] as number) + 1,
        (current[j - 1] as number) + 1,
        (previous[j - 1] as number) + cost,
      );
    }
    previous = current;
  }
  return previous[y.length] as number;
}

/**
 * The character error rate: edits to turn what was read into the expected
 * text, per expected character, both normalised as the Citation check
 * normalises (spacing, full-width forms, quote marks). Lines are compared in
 * order, so reading a table's columns instead of its rows counts against it.
 */
export function characterErrorRate(expected: string, read: string): number {
  const want = normaliseText(expected);
  return editDistance(want, normaliseText(read)) / Math.max(1, [...want].length);
}

/** Quotes of `words` words (Latin) or `characters` characters (Chinese) starting at each line, running on into the next. */
export function quotesOf(page: SetPage, words = 8, characters = 12): string[] {
  if (page.kind === "table") return page.lines.filter((line) => line.trim().length >= 6);
  const quotes: string[] = [];
  page.lines.forEach((line, index) => {
    const rest = [line, ...page.lines.slice(index + 1)].join("\n");
    if ([...line].some(isCjk)) {
      const chars = [...rest.replace(/\s+/g, "")];
      if (chars.length >= characters) quotes.push(chars.slice(0, characters).join(""));
    } else {
      const tokens = rest.split(/\s+/).filter(Boolean);
      if (tokens.length >= words) quotes.push(tokens.slice(0, words).join(" "));
    }
  });
  return quotes;
}

export interface PageScore {
  cer: number;
  quotes: number;
  found: number;
}

export function scorePage(page: SetPage, read: string): PageScore {
  const quotes = quotesOf(page);
  return {
    cer: characterErrorRate(page.lines.join("\n"), read),
    quotes: quotes.length,
    found: quotes.filter((quote) => findQuote(read, quote) !== null).length,
  };
}
