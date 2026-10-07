/**
 * Splits a Document's text into Passages. Pure: it runs in the processing
 * worker and in tests.
 */
import { CJK, isCjk } from "../../shared/text";

export interface PassageParameters {
  /** The most approximate tokens a Passage holds. */
  maxTokens: number;
  /** The most approximate tokens a Passage shares with the one before it. */
  overlapTokens: number;
  /** How many consecutive Passages a sliding window spans. */
  windowSize: number;
  /** How far the window moves on from one Passage to the next. */
  windowStep: number;
}

/**
 * About 500-token Passages overlapping by 200 tokens, each recording a sliding
 * window of 3 Passages with step 1 (ADR-0009: with the built-in model, hybrid
 * search found 17 of 20 evaluation questions at 500/200 against 15 at the old
 * backend's 400/200). Token counts are approximate; there is no tokenizer
 * dependency. Changing these needs a new `PROCESSING_VERSION`.
 */
export const PASSAGE_PARAMETERS: PassageParameters = {
  maxTokens: 500,
  overlapTokens: 200,
  windowSize: 3,
  windowStep: 1,
};

/** A Document's text, page by page. */
export interface PageText {
  /** From 1, or null for Documents without pages (TXT, Markdown). */
  page: number | null;
  text: string;
}

export interface BuiltPassage {
  /** From 0, in reading order. */
  position: number;
  /** The pages the Passage covers; it can cross a page break. Null without pages. */
  pageFrom: number | null;
  pageTo: number | null;
  /** Positions of the first and last Passage in this Passage's sliding window. */
  windowFrom: number;
  windowTo: number;
  text: string;
}

/**
 * Joins pages in the laid-out text, so a page break is a blank line. The
 * Citation check finds where each page starts in a Passage by this
 * (`withPageMarks` in ../answers/citations).
 */
const PAGE_BREAK = "\n\n";

/**
 * A run of non-CJK, non-space characters (a word), or one CJK character, each
 * with its trailing whitespace. Leading whitespace is its own match. Together
 * they cover the text.
 */
const WORD = new RegExp(`\\s+|[${CJK}]\\s*|[^\\s${CJK}]+\\s*`, "gu");

interface Piece {
  start: number;
  end: number;
  tokens: number;
}

/**
 * Roughly how many tokens a language model would count, as the retrieval
 * prototype counted them (ADR-0009): one per CJK character, and one per four
 * other characters, whitespace included.
 */
export function approximateTokens(text: string): number {
  let tokens = 0;
  for (const character of text) {
    tokens += character.charCodeAt(0) >= 128 && isCjk(character) ? 1 : 0.25;
  }
  return tokens;
}

/**
 * A line's tokens: its CJK characters, plus a quarter of the rest, rounded up.
 * The retrieval prototype counted each line this way, so these are the sizes
 * of the Passages that ADR-0009 measured.
 */
const lineTokens = (line: string) => Math.ceil(approximateTokens(line));

function item<T>(list: readonly T[], index: number): T {
  const value = list[index];
  if (value === undefined) throw new RangeError(`No item at index ${index}.`);
  return value;
}

/**
 * Splits [start, end) of `text`, a line too long for a Passage, into words. A
 * word longer than `maxWordTokens` (a URL, a base64 blob) is cut into equal
 * parts, never inside a surrogate pair.
 */
function pushWords(text: string, start: number, end: number, maxWordTokens: number, out: Piece[]) {
  for (const match of text.slice(start, end).matchAll(WORD)) {
    const wordStart = start + match.index;
    const word = match[0];
    const tokens = approximateTokens(word);
    if (tokens <= maxWordTokens) {
      out.push({ start: wordStart, end: wordStart + word.length, tokens });
      continue;
    }
    const size = Math.max(1, Math.floor(maxWordTokens * 4));
    let cut = 0;
    while (cut < word.length) {
      let next = Math.min(word.length, cut + size);
      const code = word.charCodeAt(next);
      if (code >= 0xdc00 && code <= 0xdfff) next++; // a low surrogate: keep the pair together
      out.push({
        start: wordStart + cut,
        end: wordStart + next,
        tokens: approximateTokens(word.slice(cut, next)),
      });
      cut = next;
    }
  }
}

/**
 * The pieces Passages are made of: the text's lines, each with its line break
 * (pdf.js ends each line of a page with one), leaving out blank lines. A line
 * longer than a Passage is split into words, as LangChain's recursive splitter
 * falls back from lines to words.
 */
function toPieces(text: string, maxTokens: number): Piece[] {
  const pieces: Piece[] = [];
  let start = 0;
  while (start < text.length) {
    const lineBreak = text.indexOf("\n", start);
    const end = lineBreak < 0 ? text.length : lineBreak + 1;
    const line = text.slice(start, end);
    if (line.trim()) {
      const tokens = lineTokens(line);
      if (tokens <= maxTokens) pieces.push({ start, end, tokens });
      // A word at most a quarter of a Passage leaves room for the overlap.
      else pushWords(text, start, end, maxTokens / 4, pieces);
    }
    start = end;
  }
  return pieces;
}

/** Lays the pages out as one text and returns it with a lookup from offset to page. */
function layOut(pages: readonly PageText[]) {
  let text = "";
  const starts: { offset: number; page: number | null }[] = [];
  for (const { page, text: pageText } of pages) {
    const trimmed = pageText.trim();
    if (!trimmed) continue;
    if (text) text += PAGE_BREAK;
    starts.push({ offset: text.length, page });
    text += trimmed;
  }
  const pageAt = (offset: number): number | null => {
    let low = 0;
    let high = starts.length - 1;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      if (item(starts, middle).offset <= offset) low = middle;
      else high = middle - 1;
    }
    return starts[low]?.page ?? null;
  };
  return { text, pageAt };
}

/**
 * Builds overlapping Passages of at most `maxTokens`, as the retrieval
 * prototype did (ADR-0009) and the old backend's LangChain splitter before
 * it: whole lines are added to a Passage until the next one would take it
 * over `maxTokens`; the next Passage then starts with the last lines of this
 * one, at most `overlapTokens` of them. Passages run across page breaks,
 * recording the range of pages they cover.
 */
export function buildPassages(
  pages: readonly PageText[],
  parameters: PassageParameters = PASSAGE_PARAMETERS,
): BuiltPassage[] {
  const { maxTokens, overlapTokens, windowSize, windowStep } = parameters;
  const { text, pageAt } = layOut(pages);
  const pieces = toPieces(text, maxTokens);
  if (pieces.length === 0) return [];

  // LangChain's _merge_splits: [from, to] of the pieces in each Passage.
  const spans: [number, number][] = [];
  let from = 0;
  let total = 0;
  pieces.forEach((piece, index) => {
    if (total + piece.tokens > maxTokens && index > from) {
      spans.push([from, index - 1]);
      // Keep a tail of at most `overlapTokens` that leaves room for this piece.
      while (from < index && (total > overlapTokens || total + piece.tokens > maxTokens)) {
        total -= item(pieces, from).tokens;
        from++;
      }
    }
    total += piece.tokens;
  });
  spans.push([from, pieces.length - 1]);

  // The old backend's sliding window: each Passage joins the window, and once
  // it holds more than `windowSize` Passages the oldest `windowStep` leave.
  const window: number[] = [];
  return spans.map(([first, last], position) => {
    window.push(position);
    if (window.length > windowSize) window.splice(0, windowStep);
    const start = item(pieces, first).start;
    const end = item(pieces, last).end;
    return {
      position,
      pageFrom: pageAt(start),
      pageTo: pageAt(end - 1),
      windowFrom: item(window, 0),
      windowTo: position,
      text: text.slice(start, end).trim(),
    };
  });
}
