/**
 * The one paragraph a request names by its place, in English and Chinese:
 * "the second paragraph", "paragraph 2", "the last paragraph", "第二段",
 * "最后一段". Counted from 0, or "last"; null when it names none, names more
 * than one place ("move the second paragraph after the third"), or names one
 * to leave alone ("everything but the second paragraph", "除了第二段").
 */
export type ParagraphAsked = number | "last" | null;

const ORDINALS = [
  "first",
  "second",
  "third",
  "fourth",
  "fifth",
  "sixth",
  "seventh",
  "eighth",
  "ninth",
  "tenth",
] as const;
const ZH_NUMBERS = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"] as const;

const PLACE = new RegExp(
  `\\b(?:(${ORDINALS.join("|")}|last|final|(\\d{1,2})(?:st|nd|rd|th))\\s+paragraph\\b(?!s)|paragraph\\s+(\\d{1,2})\\b)`,
  "g",
);
const ANY_PLACE = new RegExp(
  `\\b(?:${ORDINALS.join("|")}|last|final|\\d{1,2}(?:st|nd|rd|th))\\b`,
  "g",
);
const LEFT_ALONE =
  /\b(?:except(?: for)?|but(?: not)?|not|other than|apart from|besides|without|leave|keep|skip)\s+(?:the\s+|my\s+)?$/;

const ZH_PLACE = /第([一二三四五六七八九十]|\d{1,2})段|(最后)一段/g;
const ZH_ANY_PLACE = /第[一二三四五六七八九十\d]+[段句]|最后|前[一二两三]段|后[一二两三]段/g;
const ZH_LEFT_ALONE_BEFORE = /(?:除了|不要[改动碰]?|别[改动碰]?|不[改动]|保留|保持)$/;
const ZH_LEFT_ALONE_AFTER = /^(?:以外|之外|除外|不变|保持不变|别动|不要动|不动)/;

function english(question: string): ParagraphAsked | undefined {
  const text = question.toLocaleLowerCase();
  const found = [...text.matchAll(PLACE)];
  const [match] = found;
  if (!match || found.length > 1 || (text.match(ANY_PLACE) ?? []).length > 1) return undefined;
  if (LEFT_ALONE.test(text.slice(0, match.index))) return null;
  const [, word, digits, after] = match;
  if (word === "last" || word === "final") return "last";
  const place = digits ?? after;
  if (place) return Number(place) - 1;
  return ORDINALS.findIndex((ordinal) => ordinal === word);
}

function chinese(question: string): ParagraphAsked | undefined {
  const found = [...question.matchAll(ZH_PLACE)];
  const [match] = found;
  if (!match || found.length > 1 || (question.match(ZH_ANY_PLACE) ?? []).length > 1) {
    return undefined;
  }
  const before = question.slice(0, match.index);
  const after = question.slice(match.index + match[0].length);
  if (ZH_LEFT_ALONE_BEFORE.test(before) || ZH_LEFT_ALONE_AFTER.test(after)) return null;
  const [, place, last] = match;
  if (last) return "last";
  const number = ZH_NUMBERS.findIndex((each) => each === place);
  return number >= 0 ? number : Number(place) - 1;
}

export function paragraphAsked(question: string): ParagraphAsked {
  const named = [english(question), chinese(question)].filter((each) => each !== undefined);
  const [one] = named;
  return named.length === 1 && one !== undefined && one !== -1 ? one : null;
}
