/**
 * Keyword search terms (ADR-0009): text is normalised, its traditional
 * Chinese characters are read as simplified ones, then it is split into words
 * with `Intl.Segmenter`, which finds the words in Chinese and Japanese text
 * too. The FTS5 index stores the words joined by spaces, so its default
 * `unicode61` tokenizer sees the same words; queries go through the same steps,
 * minus a short list of stopwords. `Intl.Segmenter` uses the ICU data bundled
 * with Node and Electron, so this adds no dependency.
 *
 * The characters are folded one by one, by the Citation check's table
 * (../../shared/hanVariants), before the text is split, so both scripts split
 * into the same words: split first, "聯合國大會" is one word and "联合国大会"
 * three, and the evaluation's Chinese fixtures, turned into traditional
 * characters by OpenCC, gave 7% other Chinese words than in simplified ones.
 * Only the index and the queries are folded: a Passage's stored text, which
 * Citations quote and the viewer shows, keeps the characters it was written in.
 */
import { SIMPLIFIED } from "../../shared/hanVariants";
import { normaliseText } from "../../shared/text";

// The "zh" locale makes ICU use its Chinese and Japanese dictionary; other scripts split as usual.
const segmenter = new Intl.Segmenter("zh", { granularity: "word" });

/** Keeps a pasted paragraph from becoming a huge query. */
const MAX_TERMS = 64;

// Generic stopwords: function words and question words, not tuned to any evaluation set.
const STOPWORDS: ReadonlySet<string> = new Set([
  ..."a an and are as at be by can could did do does for from had has have how i in into is it its of on or should than that the their them then there these they this those to was were what when where which who why will with would you your about".split(
    " ",
  ),
  ..."的 了 是 在 和 与 及 或 什么 哪 哪些 多少 如何 怎么 怎样 为什么 吗 呢 吧 这个 那个 这 那 一个 分别 时 时候 里 中 上 对 把 被 有 都 要 会 能 可以 谁 为 每 个".split(
    " ",
  ),
]);

/** The blocks the traditional characters of `SIMPLIFIED` are in: CJK Unified Ideographs and Extension A. */
const HAN = /[\u3400-\u9fff]/g;

/** Text with each traditional Chinese character read as its simplified one. */
const simplified = (text: string) => text.replace(HAN, (char) => SIMPLIFIED.get(char) ?? char);

/** The words in normalised text, folded to simplified characters and lowercased, in order. */
function words(normalised: string): string[] {
  const found: string[] = [];
  for (const segment of segmenter.segment(simplified(normalised))) {
    if (segment.isWordLike) found.push(segment.segment.toLowerCase());
  }
  return found;
}

/** What the keyword index stores for a piece of text: its words, joined by spaces. */
export function keywordText(text: string): string {
  return words(normaliseText(text)).join(" ");
}

/** Whether a word (lowercased, as `keywordText` gives it) is left out of searches. */
export const isStopword = (word: string) => STOPWORDS.has(word);

/** The words a search looks for: each distinct word of the query that isn't a stopword, in order. */
export function keywordTerms(query: string): string[] {
  return [...new Set(words(normaliseText(query)))]
    .filter((word) => !STOPWORDS.has(word))
    .slice(0, MAX_TERMS);
}

/**
 * The FTS5 query for a search: each of its terms (see `keywordTerms`) as a
 * phrase, any of them. Null when nothing is left to search for.
 */
export function keywordQuery(query: string): string | null {
  const terms = keywordTerms(query);
  if (terms.length === 0) return null;
  return terms.map((term) => `"${term.replaceAll('"', '""')}"`).join(" OR ");
}
