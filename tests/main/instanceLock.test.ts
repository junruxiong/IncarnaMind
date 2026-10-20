import { type ChildProcess, spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";
import { build } from "vite";
import { afterAll, afterEach, beforeAll, describe, expect, test } from "vitest";
import { lockInstance } from "../../src/main/instanceLock";

const CHILD = fileURLToPath(new URL("../helpers/instanceLockChild.ts", import.meta.url));
const ROOT = fileURLToPath(new URL("../..", import.meta.url));

let bundleDir = "";
let bundle = "";
beforeAll(async () => {
  bundleDir = await mkdtemp(join(tmpdir(), "incarnamind-lock-"));
  await build({
    configFile: false,
    root: ROOT,
    logLevel: "warn",
    build: {
      ssr: CHILD,
      outDir: bundleDir,
      emptyOutDir: false,
      minify: false,
      rollupOptions: { output: { format: "es", entryFileNames: "instanceLockChild.mjs" } },
    },
  });
  bundle = join(bundleDir, "instanceLockChild.mjs");
}, 60_000);
afterAll(() => rm(bundleDir, { recursive: true, force: true }));

const folders: string[] = [];
const folder = async () => {
  const made = await mkdtemp(join(tmpdir(), "incarnamind-instance-"));
  folders.push(made);
  return made;
};
afterEach(async () => {
  for (const each of folders.splice(0)) await rm(each, { recursive: true, force: true });
});

const report = (what: string, error: unknown) => {
  throw new Error(`${what}: ${String(error)}`);
};

describe("one copy of the app holds a data folder", () => {
  test("the first has it; a second, while the first runs, doesn't; once it lets go, a start has it", async () => {
    const where = await folder();
    const first = lockInstance(where, report);
    expect(first.held).toBe(true);
    expect(lockInstance(where, report).held).toBe(false);
    first.release();
    const next = lockInstance(where, report);
    expect(next.held).toBe(true);
    next.release();
  });

  test("two data folders are held side by side", async () => {
    const [one, other] = [await folder(), await folder()];
    const both = [lockInstance(one, report), lockInstance(other, report)];
    expect(both.map((each) => each.held)).toEqual([true, true]);
    for (const each of both) each.release();
  });

  test("letting go twice is harmless", async () => {
    const lock = lockInstance(await folder(), report);
    lock.release();
    lock.release();
  });

  test("a copy killed outright lets the folder go, with nothing left to clean up", async () => {
    const where = await folder();
    const child: ChildProcess = spawn(process.execPath, [bundle, "hold", where], {
      stdio: ["ignore", "pipe", "inherit"],
    });
    const [line] = (await once(child.stdout as NodeJS.ReadableStream, "data")) as [Buffer];
    expect(JSON.parse(line.toString())).toEqual({ held: true });
    expect(lockInstance(where, report).held).toBe(false);

    child.kill("SIGKILL");
    await once(child, "exit");

    const next = lockInstance(where, report);
    expect(next.held).toBe(true);
    next.release();
  });

  test("a folder whose lock is busy with nobody holding it is waited for, and a held one is refused at once", async () => {
    const where = await folder();
    const hold = (file: string) => {
      const db = new DatabaseSync(join(where, file));
      db.exec("PRAGMA locking_mode = EXCLUSIVE");
      db.exec("BEGIN EXCLUSIVE");
      return db;
    };
    const ask = () =>
      new Promise<{ held: boolean; after: number }>((resolve, reject) => {
        const began = Date.now();
        const child = spawn(process.execPath, [bundle, "try", where], {
          stdio: ["ignore", "pipe", "inherit"],
        });
        child.on("error", reject);
        child.stdout.on("data", (line: Buffer) =>
          resolve({ ...JSON.parse(line.toString()), after: Date.now() - began }),
        );
      });

    // Another copy's try: it holds instance.lock for a moment and not instance.owner.
    const trying = hold("instance.lock");
    const waited = ask();
    await new Promise((done) => setTimeout(done, 700));
    trying.close();
    expect(await waited).toMatchObject({ held: true, after: expect.any(Number) });

    // A copy that holds the folder holds both files.
    const [lock, owner] = [hold("instance.lock"), hold("instance.owner")];
    const refused = await ask();
    expect(refused.held).toBe(false);
    expect(refused.after).toBeLessThan(5_000);
    lock.close();
    owner.close();
  });

  test("a lock file that isn't a database is reported, and the app starts all the same", async () => {
    const where = await folder();
    await writeFile(
      join(where, "instance.lock"),
      "not a database, only text that is long enough ".repeat(8),
    );
    const reported: string[] = [];
    const lock = lockInstance(where, (what) => reported.push(what));
    expect(reported).toEqual(["app.instanceLockFailed"]);
    expect(lock.held).toBe(true);
  });

  // protects=two copies asking in the same instant never both hold the folder; fails_when=two holders overlap, or none holds it; no_claim=Electron's own lock.
  test("two copies asking in the same instant, 400 times: one holds it each time", async () => {
    const where = await folder();
    const rounds = 400;
    const start = process.hrtime.bigint() + 3_000_000_000n;
    const run = () =>
      new Promise<{ held: boolean; from?: string; to?: string }[]>((resolve, reject) => {
        const child = spawn(
          process.execPath,
          [bundle, "race", where, String(start), String(rounds), "60", "40"],
          { stdio: ["ignore", "pipe", "inherit"] },
        );
        let out = "";
        child.stdout.on("data", (chunk: Buffer) => {
          out += chunk.toString();
        });
        child.on("error", reject);
        child.on("exit", (code) => {
          if (code !== 0) reject(new Error(`A copy exited with ${code}.`));
          else resolve(JSON.parse(out));
        });
      });
    const [one, other] = await Promise.all([run(), run()]);

    const overlapping: number[] = [];
    const nobody: number[] = [];
    for (let round = 0; round < rounds; round++) {
      const holders = [one[round], other[round]].filter((each) => each?.held);
      if (holders.length === 0) nobody.push(round);
      const [a, b] = holders;
      if (
        a?.from &&
        b?.from &&
        BigInt(a.from) < BigInt(b.to ?? 0) &&
        BigInt(b.from) < BigInt(a.to ?? 0)
      )
        overlapping.push(round);
    }
    expect({ overlapping, nobody }).toEqual({ overlapping: [], nobody: [] });
  }, 120_000);
});
