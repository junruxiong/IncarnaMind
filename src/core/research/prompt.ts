/**
 * What Research asks its model (#130): a plan for a topic, and each step of
 * it as an Answer's Question. A step's framing goes before the step in the
 * User's message, so the step alone is what decides the Answer's language.
 */
import { MAX_RESEARCH_STEP, MAX_RESEARCH_STEPS } from "../../shared/research";
import type { QuestionContext } from "../answers/context";
import type { ResearchPlan, ResearchSources } from "../api";

/** The most of each earlier step's notes a step reads, in characters. */
const NOTES_PER_STEP = 1_500;

export const PLAN_INSTRUCTIONS = [
  "You plan research for IncarnaMind, a notebook that answers from the User's Documents. A research Task works in the background: it searches, reads what it finds, and at the end writes a cited draft into the User's notebook.",
  "Reply with the plan only, in this form:",
  "TITLE: the topic as one clear question",
  "1. a step",
  "2. a step",
  "",
  "- Give 3 to 5 steps in the order to take them, each one short line of about 12 words that says what to look for, read or compare.",
  '- The User reads the plan: call their own Documents "the Library".',
  "- Leave out writing the draft and saving the pages: those come last on their own.",
  "- Write in the language the topic is written in.",
].join("\n");

const where = (sources: ResearchSources) =>
  sources.web && sources.library
    ? "the web and the User's Documents"
    : sources.web
      ? "the web"
      : "the User's Documents";

const library = (sources: ResearchSources) =>
  sources.web && sources.library
    ? "the web and the Library"
    : sources.web
      ? "the web"
      : "the Library";

export function planPrompt(
  topic: string,
  sources: ResearchSources,
  change: { steps: readonly string[]; text: string } | null,
): string {
  const lines = [`It will search ${library(sources)}.`, "", `Topic: ${topic}`];
  if (change) {
    lines.push(
      "",
      "The plan now:",
      ...change.steps.map((step, index) => `${index + 1}. ${step}`),
      "",
      `Change it as the User asks: ${change.text}`,
    );
  }
  return lines.join("\n");
}

const clean = (text: string) =>
  text
    .replace(/\*\*|__|`/g, "")
    .replace(/\s+/g, " ")
    .trim();

/** A plan in a model's reply: its title (the topic when it gives none) and its numbered steps; null with no step. */
export function readPlan(reply: string, topic: string): ResearchPlan | null {
  const lines = reply.split("\n").map((line) => line.trim());
  const titled = lines.find((line) => /^(?:\*\*)?title(?:\*\*)?\s*[:：]/i.test(line));
  const title = clean(titled?.replace(/^(?:\*\*)?title(?:\*\*)?\s*[:：]/i, "") ?? "");
  const steps = lines
    .map((line) => /^(?:\d+\s*[.)、．]|[-*•])\s*(.+)$/.exec(line)?.[1])
    .filter((step): step is string => step !== undefined)
    .map((step) => clean(step).slice(0, MAX_RESEARCH_STEP))
    .filter((step) => step !== "")
    .slice(0, MAX_RESEARCH_STEPS);
  if (steps.length === 0) return null;
  return { title: (title || clean(topic)).slice(0, MAX_RESEARCH_STEP), steps };
}

/** One step of a plan as an Answer's Question, with the plan and what the steps before it found. */
export function stepContext(input: {
  topic: string;
  plan: ResearchPlan;
  index: number;
  sources: ResearchSources;
  notes: readonly string[];
}): QuestionContext {
  const { plan, index } = input;
  const step = plan.steps[index] ?? "";
  const found = input.notes
    .map((note, at) => (note.trim() ? `Step ${at + 1}: ${note.slice(0, NOTES_PER_STEP)}` : ""))
    .filter(Boolean);
  const framing = [
    `This is step ${index + 1} of ${plan.steps.length} of research the User asked for, done in the background.`,
    `The research question: ${plan.title}`,
    `The User asked: ${input.topic}`,
    "The plan:",
    ...plan.steps.map((each, at) => `${at + 1}. ${each}`),
    ...(found.length > 0 ? ["", "What the steps before found:", ...found] : []),
    "",
    `Do only this step. Search ${where(input.sources)} for what it needs${input.sources.web ? ", read the few most useful pages" : ""}, and cite what you use. Then write short notes of what you found, a few sentences or a short list, each claim with its citation marker. Say plainly what you couldn't find.`,
  ].join("\n");
  return { messages: [{ role: "user", content: `${framing}\n\n${step}` }], question: step };
}

/** A step's notes as later steps read them: its Citation markers left out. */
export const notesOf = (markdown: string) =>
  markdown
    .replace(/\s*\[\^\d+\]/g, "")
    .replace(/\s*\[=\d+\]/g, "")
    .trim();
