import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { openDatabase } from "../../src/core/storage";
import { createTempDataFolder } from "../helpers/core";

const CHILD = fileURLToPath(new URL("../helpers/taskCrashChild.ts", import.meta.url));
const ROOT = fileURLToPath(new URL("../..", import.meta.url));

const STEPS = ["write:1", "write:2", "change:A", "write:3"];
/** Every point the child can be killed at, in the order it reaches them. */
const POINTS = [
  "plan:started",
  ...STEPS.flatMap((key) => [`${key}:before`, `${key}:in`, `${key}:written`, `${key}:after`]),
];

/** A small seeded generator, so a failing run can be told again from its seed. */
function random(seed: number) {
  let state = seed >>> 0 || 1;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/** Up to three points to kill at, in the order the child reaches them; the first given, if any. */
function killPlan(next: () => number, first?: string): string[] {
  const start = first ? POINTS.indexOf(first) : Math.floor(next() * POINTS.length);
  const plan = [POINTS[start] as string];
  let at = start;
  const more = Math.floor(next() * 3);
  for (let count = 0; count < more && at < POINTS.length - 1; count++) {
    at += 1 + Math.floor(next() * (POINTS.length - 1 - at));
    plan.push(POINTS[at] as string);
  }
  return plan;
}

let bundleDir = "";
let bundle = "";

beforeAll(async () => {
  bundleDir = await mkdtemp(join(tmpdir(), "incarnamind-harness-"));
  await build({
    configFile: false,
    root: ROOT,
    logLevel: "warn",
    build: {
      ssr: CHILD,
      outDir: bundleDir,
      emptyOutDir: false,
      minify: false,
      rollupOptions: { output: { format: "es", entryFileNames: "taskCrashChild.mjs" } },
    },
  });
  bundle = join(bundleDir, "taskCrashChild.mjs");
}, 60_000);

afterAll(() => rm(bundleDir, { recursive: true, force: true }));

/** One launch of the child: killed at `point`, or run until its Task is finished. */
function launch(folder: string, point = "") {
  const run = spawnSync(process.execPath, [bundle, folder, point], {
    encoding: "utf8",
    timeout: 30_000,
  });
  return { signal: run.signal, status: run.status, stdout: run.stdout, stderr: run.stderr };
}

describe("the Task journal survives a force-quit", () => {
  test("20 of 20 Tasks killed at random points, right after a file write too, resume without doing an Effect twice", async () => {
    const seed = Date.now();
    const next = random(seed);
    // The first runs kill right after a file write, and on each side of the Connector's change.
    const forced = ["write:1:written", "change:A:written", "change:A:in"];
    const outcomes: string[] = [];
    for (let run = 0; run < 20; run++) {
      const folder = await createTempDataFolder();
      const plan = killPlan(next, forced[run]);
      const where = `seed ${seed}, run ${run}, killed at ${plan.join(" → ")}`;
      let kills = 0;
      for (const point of plan) {
        const killed = launch(folder, point);
        if (killed.signal === "SIGKILL") kills++;
        else expect(killed.status, `${where}: ${killed.stderr}`).toBe(0);
      }
      expect(kills, where).toBeGreaterThan(0);
      const last = launch(folder);
      expect(last.status, `${where}: ${last.stderr}`).toBe(0);
      expect(JSON.parse(last.stdout.trim().split("\n").at(-1) ?? "{}"), where).toEqual({
        status: "done",
      });

      const ledger = existsSync(join(folder, "ledger.log"))
        ? readFileSync(join(folder, "ledger.log"), "utf8").split("\n").filter(Boolean)
        : [];
      expect(
        ledger.filter((line) => line.startsWith("duplicate")),
        where,
      ).toEqual([]);
      for (const key of ["write:1", "write:2", "write:3"]) {
        expect(readFileSync(join(folder, "out", `${key.slice(6)}.txt`), "utf8"), where).toBe(
          `${key}\n`,
        );
      }
      expect(
        readFileSync(join(folder, "service.log"), "utf8").split("\n").filter(Boolean),
        where,
      ).toEqual(["change:A"]);
      const db = openDatabase(join(folder, "journal.db"));
      try {
        const steps = db.all<{ key: string; status: string }>(
          "SELECT key, status FROM run_steps WHERE effect IS NOT NULL ORDER BY rowid",
        );
        expect(steps.map((step) => step.key)).toEqual(STEPS);
        expect(
          steps.every((step) => step.status === "completed" || step.status === "skipped"),
          `${where}: ${JSON.stringify(steps)}`,
        ).toBe(true);
      } finally {
        db.close();
      }
      outcomes.push("resumed");
    }
    expect(outcomes).toHaveLength(20);
  }, 300_000);
});
