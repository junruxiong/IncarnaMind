/**
 * Sentences about the assistant's own searching, in the first person ("I
 * searched both papers…", "my search limit ran out", "我搜索了…") or about
 * the Passages its searches returned, belong in its reply, never in the
 * User's Mind, where they stay after a later request fills the gap and the
 * page contradicts itself. A Document's words about search or retrieval stay
 * ("We found that search results…", "我们的检索方法…").
 */

// The first person singular: a Document's "we" is its authors.
const SELF = /\bI\b|\b[Mm](?:y|e)\b/;
const SEARCHING =
  /\bsearch(?:es|ed|ing)?\b|\bpassages?\b|\bretriev(?:e|ed|al)\b|\blooked\b|\bfound\b/i;
// IncarnaMind's word, capitalised as only it writes it: "the Passages retrieved…".
const PASSAGES = /\b[Tt]h(?:e|ese|ose) Passages?\b/;
const CHINESE = /我(?!们)[^，。！？]{0,6}?(?:搜索|检索|查找|找到|搜到|查到)|我的(?:搜索|检索)/;
// A cited sentence is the Document's.
const CITED = /\[\^?\d+\]/;

const SENTENCE = /[^.!?。！？\n]+(?:[.!?。！？]+["”’)]*|$)\s*/g;

const aboutSearching = (sentence: string) =>
  !CITED.test(sentence) &&
  ((SELF.test(sentence) && SEARCHING.test(sentence)) ||
    PASSAGES.test(sentence) ||
    CHINESE.test(sentence));

/**
 * Markdown to write into a Mind without its sentences about searching, and
 * how many were left out. A paragraph left with nothing goes too.
 */
export function withoutSearchTalk(markdown: string): { markdown: string; left: number } {
  let left = 0;
  const kept = markdown.split(/\n{2,}/).flatMap((paragraph) => {
    if (/^\s*(?:#{1,6}\s|```|\||>)/.test(paragraph)) return [paragraph];
    const lines = paragraph.split("\n").map((line) => {
      const lead = /^\s*(?:[-*+]|\d{1,3}[.)])\s+/.exec(line)?.[0] ?? "";
      const sentences = line.slice(lead.length).match(SENTENCE) ?? [];
      const staying = sentences.filter((sentence) => {
        const out = aboutSearching(sentence);
        if (out) left++;
        return !out;
      });
      if (staying.length === sentences.length) return line;
      const rest = staying.join("").trim();
      return rest ? `${lead}${rest}` : null;
    });
    const remaining = lines.filter((line): line is string => line !== null);
    return remaining.length > 0 ? [remaining.join("\n")] : [];
  });
  return { markdown: kept.join("\n\n"), left };
}
