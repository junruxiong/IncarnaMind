import { describe, expect, test } from "vitest";
import type { ProviderError } from "../../src/core";
import { RATE_LIMIT_PAUSES_MS, RunItemError, runItems } from "../../src/core/runs/stop";

const failure = (kind: ProviderError["kind"], extra: Partial<ProviderError> = {}) =>
  new RunItemError({ kind, message: kind, ...extra });

/** A wait that returns at once, noting how long it was asked to wait. */
function noWait() {
  const waited: number[] = [];
  return { waited, wait: async (ms: number) => void waited.push(ms) };
}

describe("Running a run's items", () => {
  test("runs at most `concurrency` at once, starting them in order", async () => {
    let running = 0;
    let most = 0;
    const started: number[] = [];
    const run = await runItems(
      [1, 2, 3, 4, 5, 6, 7, 8, 9],
      async (item) => {
        started.push(item);
        running++;
        most = Math.max(most, running);
        await new Promise((resolve) => setImmediate(resolve));
        running--;
        return item * 10;
      },
      { concurrency: 4 },
    );
    expect(most).toBe(4);
    expect(started).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(run.outcomes.map((each) => (each.status === "done" ? each.value : null))).toEqual([
      10, 20, 30, 40, 50, 60, 70, 80, 90,
    ]);
  });

  test("an item the provider limited is tried again after a pause, longer each time, and fails when the pauses run out", async () => {
    const { waited, wait } = noWait();
    const tries = new Map<string, number>();
    const run = await runItems(
      ["busy once", "always busy", "fine"],
      async (item) => {
        const tried = (tries.get(item) ?? 0) + 1;
        tries.set(item, tried);
        if (item === "always busy" || (item === "busy once" && tried === 1)) {
          throw failure("rate-limit");
        }
        return item;
      },
      { wait },
    );
    expect(run.outcomes.map((each) => each.status)).toEqual(["done", "failed", "done"]);
    expect(tries.get("busy once")).toBe(2);
    expect(tries.get("always busy")).toBe(RATE_LIMIT_PAUSES_MS.length + 1);
    // The pauses grow for the item limited again: 1 s, 2 s, 4 s…
    expect(waited.length).toBeGreaterThanOrEqual(RATE_LIMIT_PAUSES_MS.length);
    expect(Math.max(...waited)).toBeLessThanOrEqual(RATE_LIMIT_PAUSES_MS.at(-1) ?? 0);
    expect(Math.max(...waited)).toBeGreaterThan(RATE_LIMIT_PAUSES_MS[0]);
  });

  test("a plan's limit stops the run: items running finish, those not started aren't checked", async () => {
    const heard: string[] = [];
    const run = await runItems(
      ["a", "b", "c", "d", "e", "f"],
      async (item) => {
        // b's provider refuses at once, while a and c are still running.
        if (item === "b") throw failure("plan-limit", { resetsAt: "2026-10-12T14:30:00.000Z" });
        await new Promise((resolve) => setImmediate(resolve));
        return item;
      },
      { concurrency: 3, onOutcome: (item, outcome) => heard.push(`${item}:${outcome.status}`) },
    );
    expect(run.stop).toEqual({ reason: "plan-limit", resetsAt: "2026-10-12T14:30:00.000Z" });
    expect(run.outcomes.map((each) => each.status)).toEqual([
      "done",
      "not-checked",
      "done",
      "not-checked",
      "not-checked",
      "not-checked",
    ]);
    // Each was heard as it ended, once.
    expect(heard.sort()).toEqual(
      ["a:done", "b:not-checked", "c:done", "d:not-checked", "e:not-checked", "f:not-checked"].sort(),
    );
  });

  test("stopping the run leaves the items not yet done stopped", async () => {
    const controller = new AbortController();
    const run = await runItems(
      [1, 2, 3],
      async (item) => {
        if (item === 2) {
          controller.abort();
          throw new DOMException("Stopped.", "AbortError");
        }
        return item;
      },
      { signal: controller.signal },
    );
    expect(run.outcomes.map((each) => each.status)).toEqual(["done", "stopped", "stopped"]);
    expect(run.stop).toBeNull();
  });
});
