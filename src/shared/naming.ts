/**
 * The one naming rule for Minds and Chats: the first line of what the User
 * wrote or asked, cut to about this many characters. A Chat's short title from
 * the model is cleaned by the same cut, so the rule is the same everywhere.
 */
export const NAME_LENGTH = 60;

const cjkLetter = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u;
const trailingMarks = /[\s.,;:!?…。，；：！？、\-–—]+$/u;
const quotes = /^["'“”‘’「」『』《》]+|["'“”‘’「」『』《》]+$/gu;

/**
 * The text's first line, cut to about `NAME_LENGTH` characters without trailing
 * punctuation: at a word boundary where words are separated by spaces, at a
 * character in Chinese or Japanese. It counts whole characters, so none is broken.
 * Empty when the text has no first line.
 */
export function cutTitle(raw: string): string {
  const first = raw.split(/\r?\n/).find((line) => line.trim() !== "") ?? "";
  const characters = Array.from(first.replace(/\s+/g, " ").trim());
  if (characters.length <= NAME_LENGTH) return characters.join("").replace(trailingMarks, "");
  let end = NAME_LENGTH;
  const last = characters[end - 1] ?? "";
  const next = characters[end] ?? "";
  const insideWord = next !== " " && last !== " " && !cjkLetter.test(last) && !cjkLetter.test(next);
  if (insideWord) {
    const space = characters.lastIndexOf(" ", end - 1);
    if (space > 0) end = space;
  }
  return characters.slice(0, end).join("").replace(trailingMarks, "");
}

/** What a Mind is called: the title the User typed, else its first line of text. Empty when it has neither. */
export const mindName = (mind: { title: string; autoTitle: string }): string =>
  mind.title || mind.autoTitle;

/** What the quick-task model is asked to title a Chat with, in the language of the Question. */
export const SHORT_TITLE_INSTRUCTION =
  "Write a title for the conversation that starts with the Question below: at most 6 words in English, or at most 12 characters in Chinese, " +
  "in the same language as the Question. Reply with the title only: no quotes, no full stop.";

/** The model's short title, cleaned: its first line, without quotes or trailing punctuation. Empty if it gave none. */
export const cleanShortTitle = (reply: string): string =>
  cutTitle(reply.trim().replace(quotes, "")).replace(quotes, "");
