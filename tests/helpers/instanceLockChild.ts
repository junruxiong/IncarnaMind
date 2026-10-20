import { lockInstance } from "../../src/main/instanceLock";

/**
 * One of the copies in tests/main/instanceLock.test.ts. `race`: asks for the lock at the start of
 * each round, together with the other copy, holds it a moment and says when it did; `hold`: takes
 * it, says so and waits to be killed.
 */
const [mode, folder, ...rest] = process.argv.slice(2);

const report = (what: string, error: unknown) => console.error(what, error);
const now = () => process.hrtime.bigint();
const waitUntil = (until: bigint) => {
  while (now() < until) {
    // Spinning keeps the two copies within microseconds of each other.
  }
};

if (mode === "hold") {
  const lock = lockInstance(folder ?? "", report);
  console.log(JSON.stringify({ held: lock.held }));
  setInterval(() => undefined, 1_000);
} else {
  const [start, rounds, roundMs, holdMs, askMs] = rest.map(BigInt) as [
    bigint,
    bigint,
    bigint,
    bigint,
    bigint,
  ];
  const results: { held: boolean; from?: string; to?: string }[] = [];
  for (let round = 0n; round < rounds; round++) {
    waitUntil(start + round * roundMs * 1_000_000n);
    const lock = lockInstance(folder ?? "", report, Number(askMs));
    if (!lock.held) {
      results.push({ held: false });
      continue;
    }
    const from = now();
    waitUntil(from + holdMs * 1_000_000n);
    const to = now();
    lock.release();
    results.push({ held: true, from: String(from), to: String(to) });
  }
  console.log(JSON.stringify(results));
}
