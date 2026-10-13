/** The kind of Task that does research (#130). */
export const RESEARCH_TASK = "research";

/** The longest topic Research takes, in characters. */
export const MAX_RESEARCH_TOPIC = 2_000;
/** The longest step of a plan, in characters. */
export const MAX_RESEARCH_STEP = 300;
/** The most steps a plan has before its draft. */
export const MAX_RESEARCH_STEPS = 8;

/**
 * What a plan of so many steps costs, in Answers' worth (`estimateRun`'s
 * items): a step searches and reads pages, about two Answers, and the draft
 * is one more.
 */
export const researchItems = (steps: number) => steps * 2 + 1;

/** How long a plan of so many steps takes, in minutes: about one to two a step, and the draft. */
export const researchMinutes = (steps: number) => ({ from: steps + 1, to: steps * 2 + 2 });
