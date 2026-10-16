import type { Translate } from "./objects";

const SILENT_MS = 15_000;
let silentMs = SILENT_MS;

/** How long an Answer may be silent before its line says how long it usually takes; test builds shorten it. */
export const setSilentWaitMs = (ms: number): void => {
  silentMs = ms;
};

export const silentWaitMs = (): number => silentMs;

/**
 * The line's words: how long the model usually takes, in the rough measures
 * people say ("about 40 seconds", "about a minute", "about 3 minutes"), when
 * that is more than the wait so far; else that it is still working.
 */
export function waitText(t: Translate, usualSeconds: number | null, model: string | null): string {
  if (usualSeconds === null || model === null || usualSeconds * 1000 <= silentMs) {
    return t("answer.wait.still");
  }
  const tens = Math.round(usualSeconds / 10) * 10;
  if (tens < 60) return t("answer.wait.usually.seconds", { count: tens, model });
  const minutes = Math.round(usualSeconds / 60);
  return minutes <= 1
    ? t("answer.wait.usually.minute", { model })
    : t("answer.wait.usually.minutes", { count: minutes, model });
}
