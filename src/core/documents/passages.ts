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
 * PROVISIONAL (ticket #21, the retrieval prototype, decides these): the old
 * backend's parameters. About 400-token Passages overlapping by 200 tokens,
 * each recording a sliding window of 3 Passages with step 1. Token counts are
 * approximate; there is no tokenizer dependency.
 */
export const PASSAGE_PARAMETERS: PassageParameters = {
  maxTokens: 400,
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

/** Joins pages in the laid-out text, so a page break counts as a paragraph break. */
const PAGE_BREAK = "\n\n";

/** Where a Passage may end, weakest to strongest. */
const BREAK = { none: 0, word: 1, line: 2, sentence: 3, paragraph: 4 } as const;

const SENTENCE_END = /[.!?。！？…;；][\]"'”’)）」』]*$/u;

/**
 * The smallest pieces a Passage is built from, each with its trailing
 * whitespace: a run of non-CJK, non-space characters (a word), or one CJK
 * character. Leading whitespace is its own piece. Together they cover the text.
 */
const PIECE = new RegExp(`\\s+|[${CJK}]\\s*|[^\\s${CJK}]+\\s*`, "gu");

interface Piece {
  start: number;
  end: number;
  tokens: number;
  /** How good a place the end of this piece is to end a Passage. */
  strength: number;
}

const isSpaceCode = (code: number) => code === 32 || (code >= 9 && code <= 13);

/**
 * Roughly how many tokens a language model would count: about one per CJK
 * character, one per four other characters, and a run of whitespace as one character.
 */
export function approximateTokens(text: string): number {
  let tokens = 0;
  let inSpace = false;
  for (const character of text) {
    const code = character.charCodeAt(0);
    const space = code < 128 ? isSpaceCode(code) : /\s/u.test(character);
    if (space) {
      if (!inSpace) tokens += 0.25;
      inSpace = true;
      continue;
    }
    inSpace = false;
    tokens += code >= 128 && isCjk(character) ? 1 : 0.25;
  }
  return tokens;
}

function strengthAfter(piece: string): number {
  const word = piece.trimEnd();
  const space = piece.slice(word.length);
  if (/\n[^\S\n]*\n/.test(space)) return BREAK.paragraph;
  if (SENTENCE_END.test(word)) return BREAK.sentence;
  if (space.includes("\n")) return BREAK.line;
  if (space) return BREAK.word;
  // CJK characters run on without spaces, so any gap between them is a word break.
  const last = Array.from(word).at(-1);
  return last !== undefined && isCjk(last) ? BREAK.word : BREAK.none;
}

function toPieces(text: string, maxPieceTokens: number): Piece[] {
  const pieces: Piece[] = [];
  for (const match of text.matchAll(PIECE)) {
    const start = match.index;
    const value = match[0];
    const tokens = approximateTokens(value);
    if (tokens <= maxPieceTokens) {
      pieces.push({ start, end: start + value.length, tokens, strength: strengthAfter(value) });
      continue;
    }
    // A "word" too long for a Passage (a URL, a base64 blob): cut it into
    // equal pieces, never inside a surrogate pair.
    const size = Math.max(1, Math.floor(maxPieceTokens * 4));
    let cut = 0;
    while (cut < value.length) {
      let next = Math.min(value.length, cut + size);
      const code = value.charCodeAt(next);
      if (code >= 0xdc00 && code <= 0xdfff) next++; // a low surrogate: keep the pair together
      const part = value.slice(cut, next);
      const strength = next === value.length ? strengthAfter(part) : BREAK.none;
      pieces.push({
        start: start + cut,
        end: start + next,
        tokens: approximateTokens(part),
        strength,
      });
      cut = next;
    }
  }
  return pieces;
}

function item<T>(list: readonly T[], index: number): T {
  const value = list[index];
  if (value === undefined) throw new RangeError(`No item at index ${index}.`);
  return value;
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
 * Builds overlapping Passages of at most `maxTokens`, each sharing at most
 * `overlapTokens` with the one before it. A Passage prefers to end at a
 * paragraph or sentence break, and the next one to start at one. Passages run
 * across page breaks, recording the range of pages they cover.
 */
export function buildPassages(
  pages: readonly PageText[],
  parameters: PassageParameters = PASSAGE_PARAMETERS,
): BuiltPassage[] {
  const { maxTokens, overlapTokens, windowSize, windowStep } = parameters;
  const { text, pageAt } = layOut(pages);
  // A piece at most a quarter of a Passage always leaves room after the overlap.
  const pieces = toPieces(text, maxTokens / 4);
  if (pieces.length === 0) return [];

  const sums = [0];
  for (const piece of pieces) sums.push(item(sums, sums.length - 1) + piece.tokens);
  const tokens = (from: number, to: number) => item(sums, to + 1) - item(sums, from);
  const last = pieces.length - 1;

  // The strongest break in the second half of the Passage, past the previous Passage's end.
  const preferredEnd = (start: number, end: number, previousEnd: number) => {
    let best = end;
    let bestStrength = item(pieces, end).strength;
    for (let k = end - 1; k > previousEnd && tokens(start, k) >= maxTokens / 2; k--) {
      const { strength } = item(pieces, k);
      if (strength > bestStrength) {
        best = k;
        bestStrength = strength;
      }
    }
    return best;
  };

  // The earliest start that overlaps the Passage by at most `overlapTokens`,
  // moved on to the strongest break while at least half that overlap remains.
  const nextStart = (start: number, end: number) => {
    let first = end + 1;
    while (first - 1 > start && tokens(first - 1, end) <= overlapTokens) first--;
    let best = first;
    let bestStrength = item(pieces, first - 1).strength;
    for (let s = first + 1; s <= end && tokens(s, end) >= overlapTokens / 2; s++) {
      const { strength } = item(pieces, s - 1);
      if (strength > bestStrength) {
        best = s;
        bestStrength = strength;
      }
    }
    return best;
  };

  const spans: [number, number][] = [];
  let start = 0;
  let previousEnd = -1;
  for (;;) {
    let end = start;
    while (end < last && tokens(start, end + 1) <= maxTokens) end++;
    if (end < last) end = preferredEnd(start, end, previousEnd);
    spans.push([start, end]);
    if (end === last) break;
    start = nextStart(start, end);
    previousEnd = end;
  }

  // The old backend's sliding window: each Passage joins the window, and once
  // it holds more than `windowSize` Passages the oldest `windowStep` leave.
  const window: number[] = [];
  return spans.map(([from, to], position) => {
    window.push(position);
    if (window.length > windowSize) window.splice(0, windowStep);
    const first = item(pieces, from);
    const final = item(pieces, to);
    return {
      position,
      pageFrom: pageAt(first.start),
      pageTo: pageAt(final.start),
      windowFrom: item(window, 0),
      windowTo: position,
      text: text.slice(first.start, final.end).trim(),
    };
  });
}
