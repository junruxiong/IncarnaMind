/**
 * Sentences about the assistant's own searching, in the first person ("I
 * searched both papers…", "my search limit ran out", "我检索了…") or about
 * the searches and Passages of this Answer ("Where the searches found
 * little…", "检索到的段落中没有…"), belong in its reply, never in the
 * User's Mind, where they stay after a later request fills the gap and the
 * page contradicts itself. A Document's words about search or retrieval stay
 * ("We found that search results…", "我们的检索方法…"), and so do the
 * User's own ("I found three issues in the lease").
 */

import { rewriteSentences } from "./proseSentences";

// The first person singular: a Document's "we" is its authors.
const SELF = /\bI\b|\b[Mm](?:y|e)\b/;
const SEARCHING = /\bsearch(?:es|ed|ing)?\b|\bpassages?\b|\bretriev(?:e|es|ed|al|ing)\b/i;
// This Answer's: "the Passages retrieved…" (IncarnaMind's word, capitalised as only it writes it),
// "Where the searches found little…".
const THIS_ANSWERS = /\b[Tt]h(?:e|ese|ose) (?:Passages?|searches)\b/;
const CHINESE =
  /我(?!们)[^，。！？]{0,6}?(?:搜索|检索)|我的(?:搜索|检索)|检索到的?段落|段落中(?:没有|未)/;
// A cited sentence is the Document's.
const CITED = /\[\^?\d+\]/;

const aboutSearching = (sentence: string) =>
  !CITED.test(sentence) &&
  ((SELF.test(sentence) && SEARCHING.test(sentence)) ||
    THIS_ANSWERS.test(sentence) ||
    CHINESE.test(sentence));

/**
 * Markdown to write into a Mind without its sentences about searching, and
 * how many were left out. A paragraph left with nothing goes too.
 */
export function withoutSearchTalk(markdown: string): { markdown: string; left: number } {
  const kept = rewriteSentences(markdown, (sentence) => (aboutSearching(sentence) ? "" : sentence));
  return { markdown: kept.markdown, left: kept.changed };
}
