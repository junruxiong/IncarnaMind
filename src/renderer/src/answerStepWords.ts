import type { Translate } from "./objects";

/** The check's step line for this many Citations. */
export const checkedWords = (t: Translate, count: number): string =>
  t(count === 1 ? "steps.check.one" : "steps.check", { count });

/** The live line while this many Citations are checked. */
export const checkingWords = (t: Translate, count: number): string =>
  t(count === 1 ? "steps.live.checking.one" : "steps.live.checking", { count });
