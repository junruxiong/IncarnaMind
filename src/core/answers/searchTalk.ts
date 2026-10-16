/**
 * Sentences about the assistant's own searching ("The Passages retrieved did
 * not include their content", "my search limit ran out", "检索到的段落中没有…")
 * belong in its reply, never in the User's Mind, where they stay after a
 * later request fills the gap and the page contradicts itself.
 */

const PROCESS =
  /\b(?:passages?|search(?:es|ed|ing)?|search results?)\b[^.!?]{0,80}?\b(?:found|retrieved|returned|turned up|did not|didn't|do not|don't|does not|doesn't|covered?|include[sd]?|show(?:ed|s)?|gave|give)\b|\b(?:retrieved|found) passages?\b/i;
// The assistant's own: in the first person, or the Passages and searches of this Answer. A
// Document's words about retrieval ("Passages retrieved by dense retrieval…") have neither.
const SELF =
  /\b(?:I|my|me|we|our)\b|\b(?:the|these|those) (?:passages?|searches)\b|\b(?:the|these|your) (?:documents|files)\b/i;
const LIMIT = /\bsearch(?:ing)? limit\b|\blimit (?:on|of) (?:my )?searches\b/i;
const CHINESE =
  /检索到的?(?:段落|内容|资料)|未检索到|没有检索到|(?:检索|搜索)结果(?:中|里)?(?:没有|未|不包括|不含)|搜索(?:次数|上限|限制)|找到的段落|段落中(?:没有|未)|我们?的?(?:检索|搜索)/;
// A cited sentence is the Document's.
const CITED = /\[\^?\d+\]/;

const SENTENCE = /[^.!?。！？\n]+(?:[.!?。！？]+["”’)]*|$)\s*/g;

const aboutSearching = (sentence: string) =>
  !CITED.test(sentence) &&
  (LIMIT.test(sentence) ||
    CHINESE.test(sentence) ||
    (PROCESS.test(sentence) && SELF.test(sentence)));

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
