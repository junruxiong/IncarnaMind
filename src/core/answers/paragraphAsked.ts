/**
 * The one paragraph a request names by its place, in English and Chinese:
 * "the second paragraph", "paragraph 2", "the last paragraph", "第二段",
 * "最后一段". Counted from 0, or "last"; null when it names none, names more
 * than one place ("move the second paragraph after the third"), or isn't
 * plainly about that paragraph alone: a negation or a wider scope anywhere
 * ("Don't change the second paragraph, shorten the rest", "第二段不要改，其他的缩短",
 * "Make the whole thing more formal, especially the second paragraph"). Only
 * "…and leave the others as they are", "其他段落保持不变" may follow it.
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
const KEEP_THE_REST =
  /,?\s*(?:and\s+)?(?:leave|keep)\s+(?:the\s+)?(?:other|others|rest)(?:\s+(?:paragraphs?|sections?|text|of it))?(?:\s+(?:as\s+(?:they\s+are|it\s+is|is)|unchanged|alone|the\s+same))?/g;
const NOT_ALONE =
  /\b(?:rest|others?|whole|all|everything|entire|especially|particularly|don't|do not|doesn't|not|never|no need|fine|good|leave|keep|skip|except|but|besides|apart|without)\b/;

const ZH_PLACE = /第([一二三四五六七八九十]|\d{1,2})段|(最后)一段/g;
const ZH_ANY_PLACE = /第[一二三四五六七八九十\d]+[段句]|最后|前[一二两三]段|后[一二两三]段/g;
const ZH_KEEP_THE_REST = /，?其他(?:的|段落|部分)?都?(?:保持不变|不变|不要动|别动|不用改|不要改)/g;
const ZH_NOT_ALONE =
  /其他|其余|别的|整体|全部|所有|尤其|特别是|不要|别|不用|不改|不变|不动|保持|保留|没问题|挺好|很好|不错|除了|以外|之外|除外/;

function english(question: string): ParagraphAsked | undefined {
  const text = question.toLocaleLowerCase().replace(KEEP_THE_REST, "");
  const found = [...text.matchAll(PLACE)];
  const [match] = found;
  if (!match || found.length > 1 || (text.match(ANY_PLACE) ?? []).length > 1) return undefined;
  if (NOT_ALONE.test(text)) return null;
  const [, word, digits, after] = match;
  if (word === "last" || word === "final") return "last";
  const place = digits ?? after;
  if (place) return Number(place) - 1;
  return ORDINALS.findIndex((ordinal) => ordinal === word);
}

function chinese(question: string): ParagraphAsked | undefined {
  const text = question.replace(ZH_KEEP_THE_REST, "");
  const found = [...text.matchAll(ZH_PLACE)];
  const [match] = found;
  if (!match || found.length > 1 || (text.match(ZH_ANY_PLACE) ?? []).length > 1) {
    return undefined;
  }
  if (ZH_NOT_ALONE.test(text)) return null;
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
