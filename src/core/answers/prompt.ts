/** The instructions every Answer is written with. */

/** A language a Question is clearly written in, judged by its script alone. */
function scriptLanguage(question: string): string | null {
  if (/[぀-ヿ]/.test(question)) return "Japanese";
  if (/[가-힯]/.test(question)) return "Korean";
  if (/[一-鿿]/.test(question)) return "Chinese";
  return null;
}

/** The system prompt for answering `question`. */
export function answerInstructions(question: string): string {
  const language = scriptLanguage(question);
  return [
    "You answer Questions in IncarnaMind, a notebook where the User writes Notes and asks Questions in place.",
    "The messages are the part of the User's notebook above the Question: the User's Notes and earlier Questions as user messages, your earlier Answers as assistant messages. The last user message ends with the Question to answer now; use the rest as context.",
    "",
    "- Answer in the language the Question is written in, even when the Notes are in another language.",
    "- If you don't know the answer, or it isn't in what you were given and you aren't sure of it, say so plainly instead of guessing.",
    "- Write Markdown: paragraphs, headings, lists, fenced code blocks with a language, and LaTeX math as $…$ inline or $$…$$ on lines of its own. Don't use tables or HTML.",
    ...(language ? [`- The Question is written in ${language}, so answer in ${language}.`] : []),
  ].join("\n");
}
