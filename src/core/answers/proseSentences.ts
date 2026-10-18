const SENTENCE = /[^.!?。！？\n]+(?:[.!?。！？]+["”’)]*|$)\s*/g;
const FENCE = /^\s*(?:```|~~~)/;
const NOT_PROSE = /^\s*(?:#{1,6}\s|\||>)/;
const LEAD = /^\s*(?:[-*+]|\d{1,3}[.)])\s+/;

/**
 * Markdown with each sentence of its paragraphs and list items passed through
 * `rewrite`, which returns it as it is, changed, or "" to take it out, and how
 * many it changed or took out. Headings, tables, quotes and code stay as they
 * are; a line or paragraph left with nothing goes. Unchanged, the Markdown
 * comes back as it was given.
 */
export function rewriteSentences(
  markdown: string,
  rewrite: (sentence: string) => string,
): { markdown: string; changed: number } {
  let changed = 0;
  let fenced = false;
  const kept = markdown.split(/\n{2,}/).flatMap((paragraph) => {
    const fences = paragraph.split("\n").filter((line) => FENCE.test(line)).length;
    const code = fenced || fences > 0;
    if (fences % 2 === 1) fenced = !fenced;
    if (code || NOT_PROSE.test(paragraph)) return [paragraph];
    const lines = paragraph.split("\n").map((line) => {
      const lead = LEAD.exec(line)?.[0] ?? "";
      const sentences = line.slice(lead.length).match(SENTENCE) ?? [];
      const rewritten = sentences.map((sentence) => {
        const space = /\s*$/.exec(sentence)?.[0] ?? "";
        const body = sentence.slice(0, sentence.length - space.length);
        const after = rewrite(body);
        if (after === body) return sentence;
        changed++;
        return after ? `${after}${space}` : "";
      });
      if (rewritten.every((sentence, index) => sentence === sentences[index])) return line;
      const rest = rewritten.join("").trim();
      return rest ? `${lead}${rest}` : null;
    });
    const remaining = lines.filter((line): line is string => line !== null);
    return remaining.length > 0 ? [remaining.join("\n")] : [];
  });
  return { markdown: changed > 0 ? kept.join("\n\n") : markdown, changed };
}
