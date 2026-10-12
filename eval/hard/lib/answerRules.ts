/**
 * The two Answer rules of #548, scored on the hard tier's Answers: does an
 * Answer to a false-premise Question say first that the Documents don't
 * support its premise, and does an Answer to a disagreeing Question lead with
 * what the newer Document says? Phrase and position rules in English and
 * Chinese, a first pass as `declines` is (../../lib/declines): `answers.md`
 * holds each Answer for a reviewer to check.
 */
import { declines } from "../../lib/declines";
import type { Saying } from "./questions";

/** The text in lower case, without citation markers ("[^1]"), with straight apostrophes. */
const readable = (text: string) =>
  text
    .replace(/\[\^\d+\]/g, "")
    .replace(/[‘’]/g, "'")
    .toLowerCase();

/** The Answer's first paragraph that isn't a heading: where a premise is to be flagged. */
export function opening(text: string): string {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter((paragraph) => paragraph !== "" && !paragraph.startsWith("#"));
  return paragraphs[0] ?? "";
}

/** Words that say the Question's premise doesn't hold: a negation, or naming the premise. */
const ENGLISH = [
  /\b(?:did|does|do|was|were|is|are|has|have|had|could|would|will)(?:n't| not)\b/,
  /\b(?:can't|cannot|won't|never|none)\b/,
  /\b(?:premise|assum(?:es|ed|ing|ption))\b/,
  /\b(?:in fact|actually|contrary to|rather than)\b/,
];
const CHINESE = [
  /并未|并没有|并非|并不|没有|尚未|未经|不是|而是|实际上|事实上|前提|假设/,
  /(?:^|[，。；：、\s])未/,
];

/** Whether an Answer's opening says the Question takes for granted what the Documents don't say. */
export function flagsPremise(text: string): boolean {
  const first = opening(text);
  const plain = readable(first);
  return declines(first) || [...ENGLISH, ...CHINESE].some((phrase) => phrase.test(plain));
}

/** Digits grouped with commas or spaces, ungrouped: "116,000" and "116 000" read "116000". */
const ungrouped = (text: string) =>
  text.replace(/(\d)[,\u00a0\u2009\u202f ](?=\d{3}(?!\d))/g, "$1");

/** A letter or digit that joins a word: Chinese characters stand on their own. */
const ALPHANUMERIC = /[\p{Script=Latin}\p{N}]/u;

/**
 * Where `form` first stands on its own in `text`, or -1: not inside a longer
 * word or number, so "4.8" isn't found in "14.8" or "4.85", nor "six" in "sixty".
 */
function standingAt(text: string, form: string): number {
  const startsWord = ALPHANUMERIC.test(form.at(0) ?? "");
  const endsWord = ALPHANUMERIC.test(form.at(-1) ?? "");
  for (let at = text.indexOf(form); at >= 0; at = text.indexOf(form, at + 1)) {
    const before = text[at - 1] ?? "";
    const after = text[at + form.length] ?? "";
    const insideBefore =
      ALPHANUMERIC.test(before) || (before === "." && /\d/.test(text[at - 2] ?? ""));
    const insideAfter =
      ALPHANUMERIC.test(after) ||
      (/[.,]/.test(after) && /\d/.test(text[at + form.length + 1] ?? ""));
    if ((!startsWord || !insideBefore) && (!endsWord || !insideAfter)) return at;
  }
  return -1;
}

/** Where an Answer first gives what a Document says, in any of its forms, or -1. */
function saidAt(text: string, saying: Saying): number {
  const plain = ungrouped(readable(text));
  const found = saying.says
    .map((form) => standingAt(plain, ungrouped(form.toLowerCase())))
    .filter((at) => at >= 0);
  return found.length > 0 ? Math.min(...found) : -1;
}

/**
 * For a disagreeing Question: `leads`, the Answer gives what the newer
 * Document says before anything the older one says (or gives only the
 * newer's); `namesOlder`, it gives the older's too. Whether it cites the
 * newer Document is for the caller, which knows the Citations.
 */
export function newerLeads(
  text: string,
  newer: Saying,
  older: Saying,
): { leads: boolean; namesOlder: boolean } {
  const fromNewer = saidAt(text, newer);
  const fromOlder = saidAt(text, older);
  return {
    leads: fromNewer >= 0 && (fromOlder < 0 || fromNewer < fromOlder),
    namesOlder: fromOlder >= 0,
  };
}
