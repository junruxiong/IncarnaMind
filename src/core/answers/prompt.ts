/** The instructions every Answer is written with. Kept short: every word is sent with every Question. */
import type { CitationSupport } from "../api";

/** A language a Question is clearly written in, judged by its script alone. */
function scriptLanguage(question: string): string | null {
  if (/[぀-ヿ]/.test(question)) return "Japanese";
  if (/[가-힯]/.test(question)) return "Korean";
  if (/[一-鿿]/.test(question)) return "Chinese";
  return null;
}

/** The system prompt for answering `question`, before anything about Documents. */
export function answerInstructions(question: string): string {
  const language = scriptLanguage(question);
  return [
    "You answer Questions in IncarnaMind, a notebook where the User writes Notes and asks Questions in place.",
    "The messages are the part of the User's notebook above the Question: the User's Notes and earlier Questions as user messages, your earlier Answers as assistant messages. The last user message ends with the Question to answer now; use the rest as context.",
    "",
    "- Answer in the language the Question is written in, even when the Notes or Documents are in another language.",
    "- If you don't know the answer, or it isn't in what you were given and you aren't sure of it, say so plainly instead of guessing.",
    "- Write Markdown: paragraphs, headings, lists, fenced code blocks with a language, and LaTeX math as $…$ inline or $$…$$ on lines of its own. Don't use tables or HTML.",
    ...(language ? [`- The Question is written in ${language}, so answer in ${language}.`] : []),
  ].join("\n");
}

const NOT_COVERED =
  "If the Documents don't cover the Question, say so plainly, then answer from your own knowledge if you can, without markers.";

const RECORD =
  "the Passage's id, the page or two consecutive pages the quote is on (a Passage marks where each new page starts, like [p. 4]; leave pages out for a Passage without pages), and a short quote copied exactly from the Passage, without page marks";

/**
 * What to do with the User's Documents, for a way of citing. `passages`: what
 * the one search found, for a model that can't call Tools. `scoped`: the User
 * limited the Question to some of their Documents (its Search scope), and
 * `documentCount` counts only those.
 */
export function documentInstructions(
  mode: CitationSupport | "no-documents",
  documentCount: number,
  passages = "",
  scoped = false,
): string {
  const documents = `${documentCount} Document${documentCount === 1 ? "" : "s"}`;
  switch (mode) {
    case "no-documents":
      return "";
    case "tools":
      return [
        `${scoped ? `The User limited this Question to ${documents} of theirs` : `The User has added ${documents}`} (PDF, text and Markdown files). Search them with search_documents whenever they may help; search again with other words if the Passages don't answer the Question.`,
        "",
        "Cite every claim you draw from a Passage:",
        `- Before writing the Answer, call cite once with a record for each quote you will use: a marker number (1, 2, …), ${RECORD}.`,
        "- In the Answer, put the marker as [^1] right after each claim it supports. Use only markers you recorded, and add no list of sources.",
        "- Write nothing before your Tool calls: only the Answer, after them.",
        NOT_COVERED,
      ].join("\n");
    case "structured-output":
      return [
        "Passages found in the User's Documents for this Question:",
        passages,
        "",
        'Reply with one JSON object: {"answer": "…", "citations": [{"marker": 1, "passage": "P1", "pageFrom": 3, "pageTo": 3, "quote": "…"}]}.',
        "- answer: the Answer, in Markdown. Cite every claim you draw from a Passage by putting a marker such as [^1] right after it. Add no list of sources.",
        `- citations: one record for each marker: ${RECORD}.`,
        NOT_COVERED,
      ].join("\n");
    case "none":
      return [
        "Passages found in the User's Documents for this Question:",
        passages,
        "",
        "Use them where they help. Don't write citation markers or a list of sources.",
        "If the Passages don't cover the Question, say so plainly, then answer from your own knowledge if you can.",
      ].join("\n");
  }
}
