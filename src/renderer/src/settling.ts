import { useEffect, useRef, useState } from "react";

/** A wait shorter than this shows nothing, so it never flashes (DESIGN.md › Feel). */
export const SHOWN_AFTER_MS = 300;

interface Clock {
  set(run: () => void, ms: number): unknown;
  clear(timer: unknown): void;
}

/**
 * Tracks which of the things that are there have been there for `after` ms:
 * `update` says what is there now, and `onChange` hears when the settled set changes.
 */
export function createSettling(
  onChange: (settled: ReadonlySet<string>) => void,
  after = SHOWN_AFTER_MS,
  clock: Clock = {
    set: (run, ms) => setTimeout(run, ms),
    clear: (timer) => clearTimeout(timer as ReturnType<typeof setTimeout>),
  },
) {
  let settled: ReadonlySet<string> = new Set();
  const waiting = new Map<string, unknown>();
  const change = (next: ReadonlySet<string>) => {
    settled = next;
    onChange(next);
  };
  return {
    update(present: readonly string[]): void {
      const here = new Set(present);
      for (const [id, timer] of waiting) {
        if (here.has(id)) continue;
        clock.clear(timer);
        waiting.delete(id);
      }
      const kept = [...settled].filter((id) => here.has(id));
      if (kept.length !== settled.size) change(new Set(kept));
      for (const id of here) {
        if (settled.has(id) || waiting.has(id)) continue;
        waiting.set(
          id,
          clock.set(() => {
            waiting.delete(id);
            change(new Set(settled).add(id));
          }, after),
        );
      }
    },
    stop(): void {
      for (const timer of waiting.values()) clock.clear(timer);
      waiting.clear();
    },
  };
}

/** The ids, of those given, that have been there for `SHOWN_AFTER_MS`. */
export function useSettled(ids: readonly string[]): ReadonlySet<string> {
  const [settled, setSettled] = useState<ReadonlySet<string>>(new Set());
  const settling = useRef<ReturnType<typeof createSettling> | null>(null);
  settling.current ??= createSettling(setSettled);
  const key = ids.join("\n");
  // biome-ignore lint/correctness/useExhaustiveDependencies: `key` stands for `ids`.
  useEffect(() => settling.current?.update(ids), [key]);
  useEffect(() => () => settling.current?.stop(), []);
  return settled;
}
