/**
 * Find support and Check against Documents (`checkReply`): a sentence that shares
 * three words with a Passage's line is supported, or contradicted when it has a
 * number the line doesn't; otherwise unsupported. A sentence the User cites a
 * quote for, whose last clause shares none of the line's words, is unsupported
 * and names that clause as the part. Offered calculate, a sentence
 * that states a multiple ("1.1x") has it worked out as the line's first number
 * over its second (#707).
 */

import { passagesIn, recordFor } from "./passages";
import { lastUserText, type Prompt, type Reply, toolResults } from "./prompt";

/** A check Run's instructions (#96, src/core/answers/checks/prompt.ts). */
export const CHECK_RUN = /^You check one sentence the User wrote against the User's Documents/m;

/** Words that decide a check: those of four letters or more, and every number. */
const checkWords = (text: string): string[] => [
  ...(text.toLowerCase().match(/[a-z]{4,}|\d+(?:[.,]\d+)?/g) ?? []),
];

/** The clause after a sentence's last comma when the line shares none of its words: what the line doesn't cover. */
function uncoveredClause(sentence: string, line: string): string | null {
  const at = sentence.lastIndexOf(", ");
  const clause = at < 0 ? "" : sentence.slice(at + 2).replace(/[.。]$/, "");
  const words = checkWords(clause);
  const covered = checkWords(line);
  return words.length > 0 && !words.some((word) => covered.includes(word)) ? clause : null;
}

/**
 * A check of one sentence (#96): it searches for the sentence, then takes the
 * Passage line that shares the most words with it. Three shared words or more
 * support it, unless the sentence has a number the line doesn't while the
 * line has numbers: that contradicts it. Fewer leave it unsupported. A
 * sentence with "blank reply" gets an empty reply, one with "refuse" a refusal,
 * and one with "no reason" an unsupported verdict with no reason.
 * Offered calculate, a sentence that states a multiple has it worked out first.
 */
export function checkReply(prompt: Prompt, tools: readonly string[]): Reply {
  const sentence = /<sentence>\n([\s\S]*?)\n<\/sentence>/.exec(lastUserText(prompt))?.[1] ?? "";
  if (/blank reply/i.test(sentence)) return { text: "" };
  if (/\brefuse\b/i.test(sentence)) return { text: "I'm sorry, I can't help with that." };
  const searched = toolResults(prompt, "search_documents").at(-1);
  if (searched === undefined) return { tool: "search_documents", input: { query: sentence } };
  const words = checkWords(sentence);
  let best: { record: Record<string, unknown>; line: string; shared: number } | null = null;
  for (const passage of passagesIn(searched)) {
    const found = recordFor(passage, sentence, 1);
    const line = typeof found?.record.quote === "string" ? found.record.quote : "";
    const shared = checkWords(line).filter((word) => words.includes(word)).length;
    if (found && (!best || shared > best.shared)) best = { record: found.record, line, shared };
  }
  if (!best || best.shared < 3 || /\bno reason\b/i.test(sentence)) {
    return { text: "Verdict: unsupported" };
  }
  const cited = /^The User cites this for it/m.test(lastUserText(prompt));
  const uncovered = cited ? uncoveredClause(sentence, best.line) : null;
  if (uncovered) return { text: `Verdict: unsupported\nPart: ${uncovered}` };
  const lineNumbers = checkWords(best.line).filter((word) => /\d/.test(word));
  const contradicted =
    lineNumbers.length > 0 && words.some((word) => /\d/.test(word) && !lineNumbers.includes(word));
  if (toolResults(prompt, "cite").length === 0) {
    return { tool: "cite", input: { citations: [best.record] } };
  }
  const multiple = /\d+(?:\.\d+)?x\b/.exec(sentence)?.[0];
  if (tools.includes("calculate") && multiple && toolResults(prompt, "calculate").length === 0) {
    const inputs = [1, 2].map((position) => ({
      name: `n${position}`,
      label: position === 1 ? "First" : "Second",
      citation: 1,
      position,
    }));
    return {
      tool: "calculate",
      input: {
        figures: [
          {
            marker: 1,
            label: "Multiple",
            inputs,
            formula: "n1 / n2",
            as: "times",
            stated: multiple,
          },
        ],
      },
    };
  }
  // The figure worked out is the only number the sentence has: the rest of it is supported.
  if (multiple && tools.includes("calculate")) {
    return { text: "Verdict: supported\nThe Document's figures give it [^1]." };
  }
  return {
    text: contradicted
      ? `Verdict: contradicted\nThe Document says: ${best.line} [^1]`
      : `Verdict: supported\nThe Document says so [^1].`,
  };
}
