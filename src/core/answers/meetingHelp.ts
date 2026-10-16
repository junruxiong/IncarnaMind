/**
 * One-click help during a call (#839): the instructions for a button's
 * Question, whose Answer the User reads in a second or two while the call
 * goes on. The floating panel lays each line out by its first character
 * (src/shared/helpReply.ts), so the shape is fixed here.
 */

import type { MeetingHelpKind } from "../../shared/meetingHelp";
import type { WindowedPassage } from "../documents/search";
import { estimateTokens } from "./context";
import type { AnswerMessage } from "./engine";

/** How a button's Answer is laid out: options to say, readings of what they want, or short lines. */
type Shape = "options" | "readings" | "lines";

const TASKS: Readonly<Record<MeetingHelpKind, { shape: Shape; task: string }>> = {
  say: {
    shape: "options",
    task: "What could the User say next, given the last thing said?",
  },
  answer: {
    shape: "options",
    task: "Answer the newest question the other side asked, from the User's Documents.",
  },
  want: {
    shape: "readings",
    task: "What does the other side want? Read it from what they said in the last few minutes.",
  },
  sum: {
    shape: "lines",
    task: "Sum up the call so far: what was said and decided, the most important first.",
  },
  no: {
    shape: "options",
    task: "Decline what was just asked of the User, politely, keeping the relationship; the option you recommend keeps things moving.",
  },
  ask: {
    shape: "options",
    task: "What should the User ask next, to move the call forward? Each option is a question.",
  },
  check: {
    shape: "lines",
    task: "Check the newest claims in the transcript against the User's Documents: each line is a claim, then whether the Documents support it, contradict it or don't say.",
  },
  objection: {
    shape: "options",
    task: "Answer the other side's newest objection.",
  },
  nextStep: {
    shape: "options",
    task: "Propose a concrete next step that fits where the call is: a meeting, a trial, something to send.",
  },
  facts: {
    shape: "lines",
    task: "The facts from the User's Documents that matter for what is being discussed now: prices, the product, terms. Each line is one fact.",
  },
  followUp: {
    shape: "options",
    task: "A follow-up question that digs into what the interviewee just said, without suggesting an answer.",
  },
  leading: {
    shape: "options",
    task: "Is the User's own newest question (their lines in the transcript) leading the interviewee? Start with one short line that says whether it does and why, then give one or two better questions as the options.",
  },
  notCovered: {
    shape: "lines",
    task: "Compare the interview guide in the User's Documents with the transcript: each line is a topic or question from the guide not covered yet, cited to the guide.",
  },
  decisions: {
    shape: "lines",
    task: "The decisions made so far, each with its owner.",
  },
  disagree: {
    shape: "lines",
    task: "Where people in the call disagree: each line is one point, and who holds which view.",
  },
};

const SHAPES: Readonly<Record<Shape, readonly string[]>> = {
  options: [
    '- One or two options, never more. The one you recommend comes first, its line starting with "★ ".',
    '- Each option is two lines: one sentence the User could say, in quotation marks; then a line starting with "→ " and what saying it leads to, in a few words.',
    '- Write the sentences to say in the language the people in the call speak; write the "→" lines in the language of the Question.',
  ],
  readings: [
    '- One or two readings, never more. The likeliest comes first, its line starting with "★ ".',
    '- Each reading is three lines: one sentence that says what they want ("They want the total for 40 seats under £10k"), as a reading of what they said, never as a fact; then the moment of the transcript it rests on, as its time and a short quote, with its Citation marker; then a line starting with "→ " and the move it suggests.',
    "- Every reading cites the transcript.",
  ],
  lines: [
    '- At most three short lines, each starting with "- ".',
    "- Cite the transcript's times or the Documents for what each line says.",
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
 * What the model is told for a button's Question during a call. `searches`: it is offered the
 * Documents' search, and `found` holds what was found ahead for the newest lines.
 */
export function meetingHelpInstructions(
  kind: MeetingHelpKind,
  { searches, found }: { searches: boolean; found: boolean },
): string {
  const { shape, task } = TASKS[kind];
  return [
    "The User is in a call and pressed a button for help: the Question is the button's name. They will read your reply in a second or two while the call goes on.",
    `What they need: ${task}`,
    "Reply in exactly this form, with nothing before or after it: no heading, no introduction, no closing line.",
    ...SHAPES[shape],
    "- Base it on the newest lines of <live-transcript>: what was said last matters most. Each line is short.",
    "- A fact (a price, a date, a figure, a term) comes from the User's Documents or the transcript, with its Citation marker; never invent one. When they don't have it, say what the User doesn't know yet instead.",
    searches
      ? found
        ? "- <found-in-documents> holds Passages of the User's Documents already found for the newest lines: cite them by id, and search only if they don't answer."
        : "- Search the User's Documents once for what was said last."
      : "- search_documents isn't offered for this button: answer from <live-transcript> and the notebook.",
  ].join("\n");
}
