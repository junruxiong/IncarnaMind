/**
 * Tagging on a ChatGPT plan (#180). The plan's limits are shared with ChatGPT
 * itself, and adding a folder of 500 Documents could use them up in minutes,
 * so tagging takes it gently:
 *
 * - one Document at a time, its requests at least `PLAN_PACE.gapMs` apart;
 * - when the plan's limit is reached, nothing more is sent until it resets,
 *   or, when OpenAI doesn't say when, for a wait that doubles each time the
 *   limit is still reached (`firstWaitMs` up to `longestWaitMs`). Documents
 *   wait meanwhile, and one tries first when the wait is over: it never
 *   retries in a loop.
 */
import { ChatGptPlanError, isPlanLimit } from "../providers/chatgpt/errors";
import type { Pace, PaceTimers } from "../sources/pace";

export const PLAN_PACE = {
  gapMs: 3_000,
  firstWaitMs: 15 * 60_000,
  longestWaitMs: 4 * 60 * 60_000,
};

/** The Pace's lane: every plan request tagging makes, Organize's and automatic tagging's. */
const LANE = "chatgpt-plan-tagging";

export type PlanPace = ReturnType<typeof createPlanPace>;

export function createPlanPace(options: {
  pace: Pace;
  timers: PaceTimers;
  /** The wait is over: Documents waiting for the plan may go on. */
  resumed(): void;
  /** Stops the wait when the core closes. */
  signal: AbortSignal;
}) {
  const { pace, timers } = options;
  let pausedUntil: number | null = null;
  let nextWait = PLAN_PACE.firstWaitMs;

  const paused = () => pausedUntil !== null && timers.now() < pausedUntil;
  const waiting = () =>
    new ChatGptPlanError(
      "plan-limit",
      `Tagging waits for your ChatGPT plan's limit to reset, until ${new Date(pausedUntil ?? timers.now()).toISOString()}.`,
      429,
    );

  function pause(error: ChatGptPlanError): void {
    const now = timers.now();
    const until = error.resetsAt ?? now + nextWait;
    nextWait = Math.min(nextWait * 2, PLAN_PACE.longestWaitMs);
    if (pausedUntil !== null && pausedUntil >= until) return;
    pausedUntil = Math.max(until, now);
    const ends = pausedUntil;
    timers.sleep(ends - now, options.signal).then(
      () => {
        if (pausedUntil !== ends) return;
        pausedUntil = null;
        options.resumed();
      },
      () => undefined,
    );
  }

  return {
    /** Until when tagging on the plan waits (ISO 8601), or null while it doesn't. */
    pausedUntil: (): string | null => (paused() ? new Date(pausedUntil ?? 0).toISOString() : null),

    /**
     * Makes one plan request in turn, `gapMs` after the last one ended. While
     * the plan's limit holds, nothing is sent: it throws the plan's limit.
     */
    run<T>(task: () => Promise<T>, signal: AbortSignal): Promise<T> {
      if (paused()) return Promise.reject(waiting());
      return pace.run(
        LANE,
        PLAN_PACE.gapMs,
        async () => {
          if (paused()) throw waiting();
          try {
            const result = await task();
            nextWait = PLAN_PACE.firstWaitMs;
            return result;
          } catch (error) {
            if (isPlanLimit(error)) pause(error);
            throw error;
          }
        },
        signal,
      );
    },
  };
}
