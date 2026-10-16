import { type LiveStep, liveStepKey } from "./answerSteps";

/** The least time a live step stays up while another replaces it, so the line never flickers. */
export const MIN_STEP_MS = 1000;

const waitsForUser = (step: LiveStep) => step.kind === "waiting" || step.kind === "consent";

export interface StepHold {
  /** The step the Answer is on now. */
  offer(step: LiveStep): void;
  stop(): void;
}

/**
 * Shows the steps of an Answer one at a time, `shown` hearing each: a new
 * step shows once the one before has been up `MIN_STEP_MS`, then the latest
 * of those that came meanwhile; one that waits for the User shows at once.
 */
export function holdSteps(initial: LiveStep, shown: (step: LiveStep) => void): StepHold {
  let current = initial;
  let latest = initial;
  let since = Date.now();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const show = () => {
    timer = undefined;
    since = Date.now();
    current = latest;
    shown(latest);
  };
  const stop = () => {
    clearTimeout(timer);
    timer = undefined;
  };
  return {
    offer(step) {
      latest = step;
      stop();
      if (liveStepKey(step) === liveStepKey(current)) return;
      const left = MIN_STEP_MS - (Date.now() - since);
      if (waitsForUser(step) || left <= 0) show();
      else timer = setTimeout(show, left);
    },
    stop,
  };
}
