/**
 * One-click help during a call (#839): the instructions for a button's
 * Question, whose Answer the User reads in a second or two while the call
 * goes on. The floating panel lays each line out by its first character
 * (src/renderer/src/helpReply.ts), so the shape is fixed here.
 */

import { HELP_SHAPE, type HelpShape, type MeetingHelpKind } from "../../shared/meetingHelp";
import type { WindowedPassage } from "../documents/search";
import { LANGUAGE_NAMES, type Language } from "../language";
import { estimateTokens } from "./context";
import type { AnswerMessage } from "./engine";

const TASKS: Readonly<Record<MeetingHelpKind, string>> = {
  say: "What could the User say next, given the last thing said?",
  answer: "Answer the newest question the other side asked, from the User's Documents.",
  want: "What does the other side want? Read it from what they said in the last few minutes.",
  sum: "Sum up the call so far: what was said and decided, the most important first.",
  no: "Decline what was just asked of the User, politely, keeping the relationship; the option you recommend keeps things moving.",
  ask: "What should the User ask next, to move the call forward? Each option is a question.",
  check:
    "Check the newest claims in the transcript against the User's Documents: each line is a claim, then whether the Documents support it, contradict it or don't say.",
  objection: "Answer the other side's newest objection.",
  nextStep:
    "Propose a concrete next step that fits where the call is: a meeting, a trial, something to send.",
  facts:
    "The facts from the User's Documents that matter for what is being discussed now: prices, the product, terms. Each line is one fact.",
  followUp:
    "A follow-up question that digs into what the interviewee just said, without suggesting an answer.",
  leading:
    "Is the User's own newest question (their lines in the transcript) leading the interviewee? Start with one short line that says whether it does and why, then give one or two better questions as the options.",
  notCovered:
    "Compare the interview guide in the User's Documents with the transcript: each line is a topic or question from the guide not covered yet, cited to the guide.",
  decisions: "The decisions made so far, each with its owner.",
  disagree: "Where people in the call disagree: each line is one point, and who holds which view.",
};

const SHAPES: Readonly<Record<HelpShape, readonly string[]>> = {
  options: [
    '- One or two options, never more. The one you recommend comes first, its line starting with "★ ".',
    '- Each option is two lines: what the User could say, as they would say it aloud, one sentence of at most 25 words with no quotation marks around it (the panel shows it as a quote), its Citation markers at its end; then a line starting with "→ " and what saying it leads to, in at most eight words, with no Citation marker.',
    "For example:",
    "★ We can't go below list for Enterprise, but we can extend the pilot to 60 days [^1]",
    "→ keeps the deal moving; protects the margin",
    "",
    "Let me check with our team and come back to you by Friday",
    "→ buys time; momentum may drop",
  ],
  readings: [
    '- One or two readings, never more. The likeliest comes first, its line starting with "★ ".',
    '- Each reading is three lines: what they want, in one sentence, as a reading of what they said and never as a fact; then the moment of the transcript it rests on: its time and a few words they said in quotation marks, with no punctuation before the closing mark, then its Citation marker; then a line starting with "→ " and the move it suggests, in at most ten words.',
    "For example:",
    "★ They want the total for 40 seats under £10k",
    "12:04 “our budget stops at ten thousand” [^1]",
    "→ give the tiered price, then ask about their budget cycle",
  ],
  lines: [
    '- At most three short lines, each starting with "- ", each one sentence of at most 20 words, its Citation markers at its end.',
  ],
};

/** The most of the User's Documents a button's Question takes, found ahead for the newest lines. */
export const FOUND_TOKEN_LIMIT = 2_000;

/** What the newest lines say, to search the Documents for: the last two Passages of the transcript. */
export const newestSaid = (heard: readonly WindowedPassage[]): string =>
  heard
    .slice(-2)
    .map((passage) => passage.text)
    .join("\n");

/** The Passages found that fit `tokens`, the most relevant first. */
export function foundWithin(found: readonly WindowedPassage[], tokens: number): WindowedPassage[] {
  const kept: WindowedPassage[] = [];
  let left = tokens;
  for (const passage of [...found].sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0))) {
    const cost = estimateTokens(passage.text);
    if (cost > left) break;
    kept.push(passage);
    left -= cost;
  }
  return found.filter((passage) => kept.includes(passage));
}

/** The Passages found ahead (`shown`), right after the live transcript in the Question's message. */
export function withFound(messages: readonly AnswerMessage[], shown: string): AnswerMessage[] {
  const last = messages.at(-1);
  if (last?.role !== "user") return [...messages];
  const found = `<found-in-documents>\n${shown}\n</found-in-documents>`;
  const end = "</live-transcript>";
  const at = last.content.indexOf(end);
  const content =
    at === -1
      ? `${found}\n\n${last.content}`
      : `${last.content.slice(0, at + end.length)}\n\n${found}${last.content.slice(at + end.length)}`;
  return [...messages.slice(0, -1), { ...last, content }];
}

/**
 * What the model is told for a button's Question during a call. `found`: Passages of the User's
 * Documents were found ahead for the newest lines. `language`: the interface's, which the reply
 * is in.
 */
export function meetingHelpInstructions(
  kind: MeetingHelpKind,
  { found, language }: { found: boolean; language: Language },
): string {
  const shape = HELP_SHAPE[kind];
  const task = TASKS[kind];
  return [
    "The User is in a call and pressed a button for help: the Question is the button's name. They will read your reply in a second or two while the call goes on.",
    `What they need: ${task}`,
    "Reply in exactly this form, with nothing before or after it: no heading, no introduction, no closing line.",
    ...SHAPES[shape],
    `- Write in ${LANGUAGE_NAMES[language]}, the User's language${shape === "options" ? ", except what the User could say, which is in the language the people in the call speak" : ""}. This comes before the rule on the Question's language.`,
    "- Base it on the newest lines of <live-transcript>: what was said last matters most.",
    "- Never say what you can't read or which button this is.",
    "- A figure the other side says (an offer, a price they ask for) is theirs: never state it as the User's.",
    "- A fact (a price, a date, a figure, a term) comes from the User's Documents or the transcript, with its Citation marker; never invent one. When they don't have it, say what the User doesn't know yet instead.",
    ...(found
      ? [
          "- <found-in-documents> holds Passages of the User's Documents already found for the newest lines: cite them by id, and search only if they don't answer.",
        ]
      : []),
  ].join("\n");
}
