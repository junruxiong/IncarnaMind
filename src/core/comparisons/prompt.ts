/**
 * What a comparison table's cell asks its model (#98): the column's Question,
 * framed as one cell of a table about one Document. The framing goes before
 * the Question in the User's message, so the Question alone is what a model
 * without Tools searches for and what decides the Answer's language.
 */
import type { MarkedCitation } from "../api";
import type { QuestionContext } from "../answers/context";

/** What a cell's model replies when its Document doesn't answer the Question. */
export const NOT_FOUND = "NOT_FOUND";

export function cellContext(
  question: string,
  documentName: string,
  instruction: string | null,
): QuestionContext {
  const framing = [
    `This is one cell of a comparison table: its row is the Document "${documentName}", its column the Question below. Answer from that Document alone.`,
    "- Answer in as few words as the answer needs: the figure, date, name or short phrase, with its citation marker. No preamble, and no sentence around it unless the answer is one.",
    `- If the Document doesn't answer the Question, reply with exactly ${NOT_FOUND} and nothing else.`,
    ...(instruction ? [`- The User asked the table to show this too: ${instruction}`] : []),
  ].join("\n");
  return { messages: [{ role: "user", content: `${framing}\n\n${question}` }], question };
}

/**
 * Whether a cell's reply says its Document doesn't answer: `NOT_FOUND`, or
 * an Answer with no Citation, which a comparison table of cited Answers can't show.
 */
export function saysNotFound(markdown: string, citations: readonly MarkedCitation[]): boolean {
  if (citations.length === 0) return true;
  // "NOT_FOUND", "Not found.", "**NOT FOUND**" [^1]…
  const bare = markdown.replace(/\[\^\d+\]|[\s*_`"'“”.。!！-]/g, "").toUpperCase();
  return bare === "NOTFOUND";
}
