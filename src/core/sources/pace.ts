/**
 * Keeps the requests to one service apart, one at a time, as its operator
 * asks: arXiv "no more than one request every three seconds, and … a single
 * connection at a time", NCBI three a second, Crossref's public pool one
 * list query a second (ADR-0018). Each service is a lane; one Pace serves
 * the whole app, so Look up (#286) and Find sources (#296) share arXiv's.
 */

/** How the Pace tells and waits for the time: the real clock, or a test's. */
export interface PaceTimers {
  now(): number;
  /** Resolves after `ms` (at once for 0 or less); rejects with the signal's reason if it aborts first. */
  sleep(ms: number, signal: AbortSignal): Promise<void>;
}

export interface Pace {
  /**
   * Runs `task` on `lane` once the requests queued before it are done and
   * `intervalMs` has passed since the last of them ended. A task that waits
   * and is aborted never runs, and the lane goes on.
   */
  run<T>(lane: string, intervalMs: number, task: () => Promise<T>, signal: AbortSignal): Promise<T>;
}

export const realTimers: PaceTimers = {
  now: () => Date.now(),
  sleep: (ms, signal) =>
    new Promise<void>((resolve, reject) => {
      if (signal.aborted) return reject(signal.reason);
      if (ms <= 0) return resolve();
      const timer = setTimeout(() => {
        signal.removeEventListener("abort", stop);
        resolve();
      }, ms);
      const stop = () => {
        clearTimeout(timer);
        reject(signal.reason);
      };
      signal.addEventListener("abort", stop, { once: true });
    }),
};

export function createPace(timers: PaceTimers = realTimers): Pace {
  const lanes = new Map<string, { tail: Promise<void>; free: number }>();
  return {
    run(lane, intervalMs, task, signal) {
      const state = lanes.get(lane) ?? { tail: Promise.resolve(), free: 0 };
      lanes.set(lane, state);
      const turn = state.tail.then(async () => {
        signal.throwIfAborted();
        await timers.sleep(state.free - timers.now(), signal);
        try {
          return await task();
        } finally {
          // From when it ended: a slow answer is never followed at once by the next request.
          state.free = timers.now() + intervalMs;
        }
      });
      state.tail = turn.then(
        () => undefined,
        () => undefined,
      );
      return turn;
    },
  };
}
