/**
 * Counting a Mind's writing (the count in the Mind's header). Words for text
 * written with spaces, characters for Chinese and Japanese, which have none:
 * each Han, hiragana or katakana character is one, as in Word. Digits and
 * their separators ("1,234", "3.14") are one word. Pure, with no DOM or
 * Node imports, so the editor and tests share it.
 */

/** What the count shows: both kinds of unit together, or all characters with or without spaces. */
export type CountBasis = "words" | "characters" | "charactersNoSpaces";

export const COUNT_BASES: readonly CountBasis[] = ["words", "characters", "charactersNoSpaces"];

export interface TextCount {
  /** Words of text written with spaces. */
  words: number;
  /** Han, hiragana and katakana characters, one each. */
  cjk: number;
  /** Every character but line breaks, spaces included. */
  characters: number;
  /** Every character but line breaks and spaces. */
  charactersNoSpaces: number;
}

export const NO_COUNT: TextCount = { words: 0, cjk: 0, characters: 0, charactersNoSpaces: 0 };

const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/gu;
const WORD_PART = /[\p{L}\p{N}]/u;
const HYPHEN = /^[-‐‑]$/u;
const SPACE = /\s/gu;
const LINE_BREAK = /[\r\n\u2028\u2029]/gu;

const segmenter = new Intl.Segmenter("en", { granularity: "word" });

/** Counts the text; Blocks are separated by line breaks, which are not characters. */
export function countText(text: string): TextCount {
  if (text === "") return NO_COUNT;
  let words = 0;
  let cjk = 0;
  // Whether the segments just before are a word of letters and a hyphen: "well-" in "well-known".
  let afterHyphen = false;
  let previous = "";
  for (const { segment, isWordLike } of segmenter.segment(text)) {
    if (!isWordLike) {
      afterHyphen = HYPHEN.test(segment) && previous !== "";
      previous = "";
      continue;
    }
    const han = segment.match(CJK)?.length ?? 0;
    cjk += han;
    const rest = segment.replace(CJK, "");
    // "A股": the Han character counts as a character and the letter before it as a word.
    // A hyphenated compound is one word, as Word counts it.
    if ((han === 0 || WORD_PART.test(rest)) && !(afterHyphen && han === 0)) words += 1;
    afterHyphen = false;
    previous = han === 0 ? segment : "";
  }
  const characters = text.replace(LINE_BREAK, "");
  return {
    words,
    cjk,
    characters: Array.from(characters).length,
    charactersNoSpaces: Array.from(characters.replace(SPACE, "")).length,
  };
}

/** The number a basis shows. */
export function countFor(count: TextCount, basis: CountBasis): number {
  return basis === "words"
    ? count.words + count.cjk
    : basis === "characters"
      ? count.characters
      : count.charactersNoSpaces;
}

/** Whether the words basis should be called characters: the writing is mostly Chinese or Japanese. */
export const mostlyCjk = (count: TextCount): boolean => count.cjk > count.words;

/**
 * A target the User typed: "8000", "8,000 words", "3万字", "1.5万", "5k". A
 * 万 is 10,000, a 千 or k 1,000. Null when it is not a whole number above zero.
 */
export function parseTarget(input: string): number | null {
  const match = input
    .normalize("NFKC")
    .trim()
    .match(/^(\d[\d,]*(?:\.\d+)?)\s*(万|千|k)?/iu);
  if (!match?.[1]) return null;
  const unit = match[2]?.toLowerCase();
  const factor = unit === "万" ? 10_000 : unit === "千" || unit === "k" ? 1_000 : 1;
  const value = Math.round(Number(match[1].replaceAll(",", "")) * factor);
  return Number.isSafeInteger(value) && value > 0 ? value : null;
}
