/**
 * The Document's words nearest to words an Answer put in quotation marks that
 * the Citation check didn't find (#483), for the Citation's card to set beside
 * the Answer's own. Each stretch of the Answer's words is looked for in the
 * cited text sentence by sentence with `findQuote`, the Citation check's own
 * matching, so what the card says it found is what the check would find. The
 * longest stretch found, in the sentence that holds the longest one, is the
 * match: the Answer's words around it, and the sentence's around it, are what
 * differs.
 */

import { joinUnits, numbersIn } from "../../shared/locations";
import { findQuote, type MatchOptions, type TextRange } from "../../shared/quoteMatch";
import type { UnitText } from "../../shared/units";
import type { QuotedPiece, QuotedWords } from "../api";

/** The fewest words a stretch must be worth for its sentence to count as near: 3 words, or 6 Chinese or Japanese characters. */
const MIN_NEAR_WORDS = 3;

/** The most sentences looked through: a Unit of a sheet may hold thousands. */
const MAX_SENTENCES = 300;

/** The longest the Document's words are shown, in characters; a longer sentence is cut around the match. */
const MAX_SHOWN = 240;

const sentenceSegmenter = new Intl.Segmenter(undefined, { granularity: "sentence" });

const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u;
const LETTER_OR_DIGIT = /[\p{L}\p{N}]/u;

/**
 * A line break where a line wrapped inside a sentence, not where a heading or a
 * sentence ended: no full stop or colon before it, and no capital after it.
 */
const WRAPPED_LINE = /(?<![.!?:。！？：\n])\n(?![\p{Lu}\n])/gu;

/**
 * What the Answer's words are cut into: a Chinese or Japanese character (no
 * dictionary decides where a word starts), a number, a word, or one mark.
 */
const TOKEN =
  /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]|\p{N}+(?:[.,]\p{N}+)*|[\p{L}\p{N}]+(?:['’]\p{L}+)?|\S/gu;

interface Token {
  start: number;
  end: number;
  /** How many words it is worth: a word or a number 1, a Chinese or Japanese character a half, a mark none. */
  weight: number;
}

/** The Answer's words cut into tokens, each with where it is. */
function tokensOf(text: string): Token[] {
  return [...text.matchAll(TOKEN)].map((match) => ({
    start: match.index,
    end: match.index + match[0].length,
    weight: CJK.test(match[0]) ? 0.5 : LETTER_OR_DIGIT.test(match[0]) ? 1 : 0,
  }));
}

/** A stretch of the Answer's words found in one sentence, and where in the sentence. */
interface Stretch {
  first: number;
  last: number;
  ranges: TextRange[];
}

/** The longest run of the Answer's tokens that `findQuote` finds in `sentence`, or null. */
function longestStretch(
  sentence: string,
  text: string,
  tokens: readonly Token[],
  options: MatchOptions,
): Stretch | null {
  const find = (first: number, last: number) =>
    findQuote(
      sentence,
      text.slice((tokens[first] as Token).start, (tokens[last] as Token).end),
      options,
    );
  let best: Stretch | null = null;
  let length = 0;
  // A run is found only if the runs inside it are, so only one a token longer than the best is tried.
  for (let first = 0; first + length < tokens.length; first++) {
    let last = first + length;
    let ranges = find(first, last);
    if (!ranges) continue;
    for (let more = last + 1; more < tokens.length; more++) {
      const longer = find(first, more);
      if (!longer) break;
      last = more;
      ranges = longer;
    }
    best = { first, last, ranges };
    length = last - first + 1;
  }
  return best;
}

const wordsIn = (tokens: readonly Token[], stretch: Stretch) =>
  tokens.slice(stretch.first, stretch.last + 1).reduce((sum, token) => sum + token.weight, 0);

/** Whether `at` is inside a word written with spaces between its words. */
function insideWord(text: string, at: number): boolean {
  const before = text[at - 1] ?? "";
  const after = text[at] ?? "";
  return (
    LETTER_OR_DIGIT.test(before) &&
    LETTER_OR_DIGIT.test(after) &&
    !CJK.test(before) &&
    !CJK.test(after)
  );
}

/** Text cut into pieces by `ranges`: the ranges `found`, the rest not. */
function pieces(text: string, ranges: readonly TextRange[]): QuotedPiece[] {
  const result: QuotedPiece[] = [];
  const push = (piece: string, found: boolean) => {
    if (piece === "") return;
    const last = result.at(-1);
    if (last?.found === found) last.text += piece;
    else result.push({ text: piece, found });
  };
  let at = 0;
  for (const { start, end } of [...ranges].sort((a, b) => a.start - b.start)) {
    if (end <= at) continue;
    push(text.slice(at, Math.max(at, start)), false);
    push(text.slice(Math.max(at, start), end), true);
    at = end;
  }
  push(text.slice(at), false);
  return result;
}

/** The sentence as shown: cut around the match when long, with line breaks read as spaces. */
function shownSentence(sentence: string, ranges: readonly TextRange[]): QuotedPiece[] {
  const lead = sentence.length - sentence.trimStart().length;
  const text = sentence.trim();
  const inside = ranges
    .map((range) => ({
      start: Math.max(0, range.start - lead),
      end: Math.min(text.length, range.end - lead),
    }))
    .filter((range) => range.start < range.end);
  const matchStart = Math.min(...inside.map((range) => range.start));
  const matchEnd = Math.max(...inside.map((range) => range.end));
  let from = 0;
  let to = text.length;
  if (text.length > MAX_SHOWN) {
    const size = Math.max(MAX_SHOWN, matchEnd - matchStart);
    const room = size - (matchEnd - matchStart);
    from = Math.min(Math.max(0, matchStart - Math.floor(room / 2)), text.length - size);
    to = from + size;
    while (from < matchStart && insideWord(text, from)) from++;
    while (to > matchEnd && insideWord(text, to)) to--;
  }
  const cut = pieces(
    text.slice(from, to),
    inside.map((range) => ({ start: range.start - from, end: range.end - from })),
  );
  const first = cut[0];
  const last = cut.at(-1);
  if (from > 0 && first) first.text = `…${first.text}`;
  if (to < text.length && last) last.text = `${last.text}…`;
  return cut
    .map((piece) => ({ ...piece, text: piece.text.replace(/\s+/g, " ") }))
    .filter((piece) => piece.text !== "");
}

/**
 * The Document's words nearest `quoted` in the text of `units`, and how near:
 * `near` is how many words the longest stretch the Answer's words share with a
 * sentence is worth (0 when none is worth `MIN_NEAR_WORDS`, and then `closest`
 * is null).
 */
export function closestWords(
  units: readonly UnitText[],
  quoted: string,
  matching: Pick<MatchOptions, "lostLigatures">,
): { near: number; shown: QuotedWords } {
  const tokens = tokensOf(quoted);
  const options: MatchOptions = { ...matching, numbers: numbersIn(units) };
  const { text } = joinUnits(units);
  const flowing = text.replace(WRAPPED_LINE, " ");
  let best: { sentence: string; stretch: Stretch; near: number } | null = null;
  let seen = 0;
  for (const { segment, index } of sentenceSegmenter.segment(flowing)) {
    if (++seen > MAX_SENTENCES) break;
    const sentence = text.slice(index, index + segment.length);
    const stretch = longestStretch(sentence, quoted, tokens, options);
    const near = stretch ? wordsIn(tokens, stretch) : 0;
    if (stretch && near > (best?.near ?? 0)) best = { sentence, stretch, near };
  }
  if (!best || best.near < MIN_NEAR_WORDS) {
    return { near: 0, shown: { answer: [{ text: quoted, found: false }], closest: null } };
  }
  const first = tokens[best.stretch.first] as Token;
  const last = tokens[best.stretch.last] as Token;
  return {
    near: best.near,
    shown: {
      answer: pieces(quoted, [{ start: first.start, end: last.end }]),
      closest: shownSentence(best.sentence, best.stretch.ranges),
    },
  };
}
