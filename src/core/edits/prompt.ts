/**
 * What Edit with AI asks the model (#93), and how its reply is read. The
 * model sees the selected passage with its Citations and footnotes as
 * numbered markers ("[1]", "[^1]") and each heading, list item and quote with
 * its paragraph mark ("## ", "- ", "1. ", "> "), the paragraphs around it for meaning, the
 * words just before and after it when it is part of a sentence (#94: a
 * change made again is often a few words), and the task; it gives back the
 * passage's new text, which ../proposals/write compares with the old.
 */
import type { EditAction } from "../api";

/** For every task but "explain": rewrite the passage, keeping its markers. */
export const EDIT_INSTRUCTIONS = [
  "You edit a passage of the User's own notes, as they ask. The passage is between <passage> and </passage>.",
  "The paragraphs it sits in are between <context> and </context>: they are there for meaning only.",
  "When the passage is part of a sentence, the words just before it are between <before> and </before>, and the words just after it between <after> and </after>. They stay as they are.",
  "",
  "Reply with the new text of the passage and nothing else: no preamble, no explanation, no quotation marks around it, no code fence, no Markdown formatting but the paragraph marks below.",
  "- The new text must fit between the words before and after it. Never repeat those words in it.",
  "- Keep every Citation marker, such as [1], and every footnote marker, such as [^1], right after the words it belongs to. Never add a marker the passage doesn't have, and never change a marker's number. Leave a marker out only if the words it belongs to are gone.",
  "- Keep each formula written as $…$ exactly as it is.",
  "- Keep the passage's paragraphs apart with a blank line. Don't join or split them unless the task asks for it.",
  "- A paragraph that starts with #, ## or ### is a heading, one with - a bullet, 1. a numbered item, > a quote; one with none is plain text. Start each of yours the same way, so a heading stays a heading and an item an item, unless the task asks otherwise.",
  "- Write in the passage's language, unless the task is to translate it.",
  "- Change only what the task asks for. Keep names, numbers and facts as they are.",
].join("\n");

/** For "explain": a few sentences to go under the passage. */
export const EXPLAIN_INSTRUCTIONS = [
  "You explain a passage of the User's own notes. The passage is between <passage> and </passage>.",
  "The paragraphs it sits in are between <context> and </context>: they are there for meaning only.",
  "",
  "Reply with the explanation and nothing else: two to four plain sentences for someone reading the passage for the first time, in the passage's language. No preamble, no markers, no Markdown formatting.",
].join("\n");

/** What each action asks for; `instruction` is the language or the User's words. */
function taskFor(action: EditAction, instruction: string): string {
  switch (action) {
    case "improve":
      return "Improve the writing: make it clearer, smoother and correct, with the same meaning and tone. Leave what is fine as it is.";
    case "shorten":
      return "Make it shorter: say the same in fewer words. Keep each marker whose words you keep.";
    case "translate":
      return `Translate it into ${instruction}. Keep names, numbers and markers.`;
    case "explain":
      return "Explain what the passage says.";
    case "instruction":
      return `Do what the User asks: ${instruction}`;
  }
}

/**
 * The request: the context, the words around the passage (each only when
 * there are any), the passage and the task.
 */
export function editPrompt(input: {
  action: EditAction;
  instruction: string;
  passage: string;
  context: string;
  before?: string;
  after?: string;
}): string {
  const near = (tag: "before" | "after", words: string | undefined) =>
    words ? [`<${tag}>`, words, `</${tag}>`, ""] : [];
  return [
    "<context>",
    input.context,
    "</context>",
    "",
    ...near("before", input.before),
    "<passage>",
    input.passage,
    "</passage>",
    "",
    ...near("after", input.after),
    `<task action="${input.action}">${taskFor(input.action, input.instruction)}</task>`,
  ].join("\n");
}

const QUOTES: readonly [string, string][] = [
  ['"', '"'],
  ["“", "”"],
  ["「", "」"],
  ["'", "'"],
];

/**
 * The passage's new text in the model's reply: without thinking, a code
 * fence, the passage's tags, or quotation marks around all of it that the
 * passage didn't have. Empty when it gave nothing.
 */
export function readEditReply(reply: string, passage: string): string {
  let text = reply
    .replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, "")
    .replace(/\r\n?/g, "\n")
    .trim();
  const fence = /^```[^\n]*\n([\s\S]*?)\n?```$/.exec(text);
  if (fence) text = (fence[1] ?? "").trim();
  const tags = /^<passage>\s*([\s\S]*?)\s*<\/passage>$/.exec(text);
  if (tags) text = tags[1] ?? "";
  for (const [open, close] of QUOTES) {
    if (
      text.length > 1 &&
      text.startsWith(open) &&
      text.endsWith(close) &&
      !passage.startsWith(open) &&
      !text.slice(1, -1).includes(close)
    ) {
      text = text.slice(open.length, -close.length).trim();
    }
  }
  return text;
}

/** Two texts that read the same once their spaces are made alike: the model changed nothing. */
export const sameText = (a: string, b: string) =>
  a.replace(/\s+/g, " ").trim() === b.replace(/\s+/g, " ").trim();
