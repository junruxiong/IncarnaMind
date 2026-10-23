/**
 * The AI flows `npm run eval:speed` times (eval/speed/README.md), each driven through the core as
 * the app drives it, in English and Chinese. A flow's run is timed from the User's action, as
 * the events the window shows arrive: the first sign of work, the first words, the end, and
 * the longest wait with no sign between them. The model requests and local steps it made come
 * from the core's own timing records (src/core/timing), tagged with the flow.
 */
import type { Core, CoreEventName } from "../../src/core";
import { ask } from "../everyday/asking";

export type Language = "en" | "zh-CN";

export interface FlowContext {
  core: Core;
  language: Language;
}

/** When the window saw each kind of sign, in milliseconds from the User's action. */
export interface FlowSigns {
  firstSignMs: number | null;
  firstWordsMs: number | null;
  doneMs: number;
  /** The longest stretch with no new sign of work, from the action to the end. */
  longestQuietMs: number;
}

export interface Flow {
  id: string;
  /** What a User does, for the report. */
  label: string;
  run(context: FlowContext): Promise<FlowSigns>;
}

/** The events that show the User something is happening, and those that carry the first words. */
const SIGNS: readonly CoreEventName[] = [
  "answer.started",
  "answer.phase",
  "answer.reasoning",
  "answer.delta",
  "answer.toolCallStarted",
  "answer.toolCallFinished",
  "answer.citationAdded",
];
const WORDS: ReadonlySet<CoreEventName> = new Set(["answer.delta"]);

/** Times `act` by the signs the core sends while it runs. */
export async function timed(core: Core, act: () => Promise<unknown>): Promise<FlowSigns> {
  const start = performance.now();
  const at = () => performance.now() - start;
  const signs: number[] = [];
  let firstWordsMs: number | null = null;
  const stops = SIGNS.map((event) =>
    core.on(event, () => {
      signs.push(at());
      if (WORDS.has(event)) firstWordsMs ??= at();
    }),
  );
  try {
    await act();
  } finally {
    for (const stop of stops) stop();
  }
  const doneMs = at();
  const marks = [0, ...signs, doneMs];
  const longestQuietMs = marks
    .slice(1)
    .reduce((longest, mark, index) => Math.max(longest, mark - (marks[index] ?? 0)), 0);
  return { firstSignMs: signs[0] ?? null, firstWordsMs, doneMs, longestQuietMs };
}

const QUESTION: Record<Language, string> = {
  en: "What does the JP Morgan report say about emissions?",
  "zh-CN": "联合国 2022 年度报告的重点是什么？",
};

export const FLOWS: readonly Flow[] = [
  {
    id: "question",
    label: "Ask a Question in the assistant",
    async run({ core, language }) {
      const mind = await core.createMind({ title: language === "en" ? "Briefing" : "简报" });
      return timed(core, () =>
        ask(core, { text: QUESTION[language], mindId: mind.id, conversationId: null, focus: null }),
      );
    },
  },
];
