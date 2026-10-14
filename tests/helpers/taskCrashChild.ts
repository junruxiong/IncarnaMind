/**
 * One launch of the force-quit harness (tests/core/taskCrash.test.ts), in a
 * process of its own: the real Task journal on real SQLite, and a Task that
 * writes files and makes a change through a stand-in Connector. At the point
 * named on its command line it kills itself with SIGKILL, as a force-quit
 * does: nothing is closed or flushed. Launched again on the same folder, it
 * goes on, answering what the Task asks as the User would after looking.
 *
 *     node taskCrashChild.mjs <folder> [<point>]
 *
 * Its own record (`ledger.log`, synced line by line) says which Effects ran,
 * and "duplicate" for any that ran again after it had happened.
 */
import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  writeSync,
} from "node:fs";
import { join } from "node:path";
import type { Task } from "../../src/core/api";
import { fileWrite } from "../../src/core/runs/effects";
import { createTasks, type TaskKind } from "../../src/core/runs/tasks";
import { migrate, openDatabase } from "../../src/core/storage";

const [folder = "", killAt = ""] = process.argv.slice(2);
const out = join(folder, "out");
const ledger = join(folder, "ledger.log");
const service = join(folder, "service.log");
mkdirSync(out, { recursive: true });

/** Appends a line and syncs it, so a kill right after keeps it. */
function append(path: string, line: string): void {
  const fd = openSync(path, "a");
  try {
    writeSync(fd, `${line}\n`);
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

const linesOf = (path: string) =>
  existsSync(path) ? readFileSync(path, "utf8").split("\n").filter(Boolean) : [];

/** A force-quit at this point, when it is the one asked for. */
function point(name: string): void {
  if (name === killAt) process.kill(process.pid, "SIGKILL");
}

/** The Task's steps with Effects, in order. */
export const HARNESS_STEPS = ["write:1", "write:2", "change:A", "write:3"] as const;

const kind: TaskKind = {
  kind: "harness",
  async run(task) {
    task.started("plan");
    point("plan:started");
    task.settled("plan", "completed");
    for (const key of HARNESS_STEPS) {
      point(`${key}:before`);
      if (key.startsWith("write:")) {
        const path = join(out, `${key.slice(6)}.txt`);
        const contents = `${key}\n`;
        const write = await fileWrite(path, contents);
        await task.effect(key, write.effect, async () => {
          point(`${key}:in`);
          if (existsSync(path) && readFileSync(path, "utf8") === contents) {
            append(ledger, `duplicate ${key}`);
          }
          await write.perform();
          point(`${key}:written`);
          append(ledger, `performed ${key}`);
        });
      } else {
        const effect = {
          kind: "connector" as const,
          connectorId: "service",
          connectorName: "Service",
          tool: "change",
        };
        await task.effect(key, effect, async () => {
          point(`${key}:in`);
          if (linesOf(service).includes(key)) append(ledger, `duplicate ${key}`);
          append(service, key);
          point(`${key}:written`);
          append(ledger, `performed ${key}`);
        });
      }
      point(`${key}:after`);
    }
    return { status: "done" };
  },
};

const db = openDatabase(join(folder, "journal.db"));
migrate(db);

function finish(task: Task): void {
  process.stdout.write(`${JSON.stringify({ status: task.status })}\n`);
  db.close();
  process.exit(0);
}

/** What the User does on seeing the ask: looks in the service, and skips the change if it's there. */
function answer(task: Task): void {
  if (task.ask?.kind !== "unknown-outcome") return;
  const key = task.ask.stepKey;
  const there = key.startsWith("change:")
    ? linesOf(service).includes(key)
    : existsSync(join(out, `${key.slice(6)}.txt`));
  void tasks.answer(task.id, there ? "skip" : "redo");
}

const tasks = createTasks({
  db,
  now: () => new Date().toISOString(),
  kinds: [kind],
  approvals: { decide: () => "ask", request: async () => true },
  retention: { retain: () => {}, release: () => {} },
  cost: () => null,
  changed: (task) => {
    if (task.finishedAt) finish(task);
    else if (task.ask) answer(task);
  },
  notify: () => {},
  stepsRunning: () => {},
  reportError: (error) => {
    process.stderr.write(`${String(error)}\n`);
    process.exit(3);
  },
});

const started = db.get<{ count: number }>("SELECT count(*) AS count FROM runs")?.count ?? 0;
if (started === 0) {
  tasks.start("harness", null, {});
} else {
  tasks.settleOrphans();
  for (const task of tasks.list()) {
    if (task.finishedAt) finish(task);
    else if (task.ask) answer(task);
  }
}
