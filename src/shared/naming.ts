/**
 * The one naming rule for Minds and conversations: the first line of what the
 * User wrote or asked, cut to about this many characters. A conversation's short
 * title from the model is cleaned by the same cut, so the rule is the same everywhere.
 */
export const NAME_LENGTH = 60;

/** The fewest characters a cut at a clause mark keeps, so an opening like "Yes, but" isn't all that is left. */
const MIN_CLAUSE = 16;

const cjkLetter = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u;
const trailingMarks = /[\s.,;:!?…。，；：！？、\-–—]+$/u;
const quotes = /^["'“”‘’「」『』《》]+|["'“”‘’「」『』《》]+$/gu;
const lastWord = /\s([\p{Script=Latin}'’]+)$/u;

/** Words that start something a cut leaves unsaid; a name never ends on one. */
const DANGLING_WORDS: ReadonlySet<string> = new Set(
  (
    "a an the of to in on at for from with without by as and or but nor plus than that which who whom whose " +
    "if so per via into onto over under about between among after before during while when where what how why " +
    "is are was were be been do does did has have had this these those its their our your can will would should could may might must"
  ).split(" "),
);
const DANGLING_CHARACTERS = "的和与及或并且在对把被为之而但若如则以从向比同跟给将也都还就又才却";
const DANGLING_PAIRS = [
  "以及",
  "而且",
  "并且",
  "或者",
  "因为",
  "所以",
  "如果",
  "但是",
  "然而",
  "同时",
  "其中",
  "关于",
  "对于",
  "根据",
  "按照",
  "通过",
  "为了",
  "由于",
  "还是",
  "加上",
  "另外",
  "此外",
];

/** Whether `characters` can be cut before the one at `end`: between words, or beside a Chinese or Japanese character. */
function isBoundary(characters: readonly string[], end: number): boolean {
  const before = characters[end - 1] ?? "";
  const after = characters[end] ?? "";
  return (
    end >= characters.length ||
    after === " " ||
    before === " " ||
    cjkLetter.test(before) ||
    cjkLetter.test(after)
  );
}

/** The last place at or before `limit` where the text can be cut; `limit` itself inside a word that is too long. */
function boundaryAtOrBefore(characters: readonly string[], limit: number): number {
  for (let end = Math.min(limit, characters.length); end > 0; end--) {
    if (isBoundary(characters, end)) return end;
  }
  return Math.min(limit, characters.length);
}

/** `text` without the words a cut leaves hanging at its end ("…per year, plus"), and without trailing marks. */
function withoutDanglingEnd(text: string): string {
  let kept = text.replace(trailingMarks, "");
  for (;;) {
    const word = lastWord.exec(kept);
    if (word?.[1] && DANGLING_WORDS.has(word[1].toLowerCase())) {
      kept = kept.slice(0, word.index).replace(trailingMarks, "");
      continue;
    }
    const characters = Array.from(kept);
    const pair = DANGLING_PAIRS.find((each) => kept.endsWith(each));
    if (pair && characters.length > pair.length) {
      kept = characters.slice(0, -pair.length).join("").replace(trailingMarks, "");
      continue;
    }
    const last = characters.at(-1) ?? "";
    if (characters.length > 1 && DANGLING_CHARACTERS.includes(last)) {
      kept = characters.slice(0, -1).join("").replace(trailingMarks, "");
      continue;
    }
    return kept;
  }
}

/** Whether the character at `index` ends a clause: a Chinese mark, or a comma, semicolon or colon before a space. */
function endsClause(characters: readonly string[], index: number): boolean {
  const mark = characters[index];
  if (mark === undefined) return false;
  return "，；：。？！".includes(mark) || (",;:".includes(mark) && characters[index + 1] === " ");
}

/**
 * The text's first line, cut to about `NAME_LENGTH` characters without trailing
 * punctuation: at a word boundary where words are separated by spaces, at a
 * character in Chinese or Japanese. It counts whole characters, so none is broken.
 * The cut never leaves a half sentence: it falls back to the last clause mark
 * when that keeps at least half of the cut ("…per year" rather than "…per year, plus"),
 * and drops a word such as "the" or "plus", or a Chinese 的 or 和, left at its end.
 * Empty when the text has no first line.
 */
export function cutTitle(raw: string): string {
  const first = raw.split(/\r?\n/).find((line) => line.trim() !== "") ?? "";
  const characters = Array.from(first.replace(/\s+/g, " ").trim());
  if (characters.length <= NAME_LENGTH) return characters.join("").replace(trailingMarks, "");
  const end = boundaryAtOrBefore(characters, NAME_LENGTH);
  for (let index = end - 1; index >= Math.max(MIN_CLAUSE, end / 2); index--) {
    if (endsClause(characters, index)) {
      return withoutDanglingEnd(characters.slice(0, index).join(""));
    }
  }
  return withoutDanglingEnd(characters.slice(0, end).join(""));
}

/**
 * `text` cut to at most `limit` characters at a word boundary, as `cutTitle` cuts,
 * for a name that has room for more than a title, such as a file's.
 */
export function cutAtWord(text: string, limit: number): string {
  const characters = Array.from(text);
  if (characters.length <= limit) return text;
  return withoutDanglingEnd(characters.slice(0, boundaryAtOrBefore(characters, limit)).join(""));
}

/**
 * `name` shortened to one line: kept whole if `fits` says it fits, else cut at a
 * word boundary (a character in Chinese or Japanese) and ended with "…", so the
 * longest start that fits shows.
 */
export function fitName(name: string, fits: (text: string) => boolean): string {
  if (fits(name)) return name;
  const characters = Array.from(name);
  let low = 0;
  let high = characters.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (fits(`${characters.slice(0, middle).join("")}…`)) low = middle;
    else high = middle - 1;
  }
  const end = boundaryAtOrBefore(characters, Math.max(low, 1));
  const kept = withoutDanglingEnd(characters.slice(0, end).join(""));
  return `${kept || characters.slice(0, Math.max(low, 1)).join("")}…`;
}

/** What a Mind is called: the title the User typed, else its first line of text. Empty when it has neither. */
export const mindName = (mind: { title: string; autoTitle: string }): string =>
  mind.title || mind.autoTitle;

/**
 * What a conversation is called: the title the User typed, else its own name (its first
 * Question, cut, or the model's short title). Empty when it has neither: "New chat".
 */
export const conversationName = (conversation: { title: string; autoTitle: string }): string =>
  conversation.title || conversation.autoTitle;

/** What the quick-task model is asked to title a conversation with, in the language of the Question. */
export const SHORT_TITLE_INSTRUCTION =
  "Write a title for the conversation that starts with the Question below: at most 6 words in English, or at most 12 characters in Chinese, " +
  "in the same language as the Question. Reply with the title only: no quotes, no full stop.";

/** The model's short title, cleaned: its first line, without quotes or trailing punctuation. Empty if it gave none. */
export const cleanShortTitle = (reply: string): string =>
  cutTitle(reply.trim().replace(quotes, "")).replace(quotes, "");
