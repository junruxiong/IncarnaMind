import { existsSync, mkdtempSync, readFileSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import { claimStart } from "../../src/main/startClaim";

const folders: string[] = [];
const folder = () => {
  const made = mkdtempSync(join(tmpdir(), "incarnamind-claim-"));
  folders.push(made);
  return made;
};
afterEach(() => {
  for (const each of folders.splice(0)) rmSync(each, { recursive: true, force: true });
});

const live = process.pid;
/** A process id nothing runs under. */
const gone = 2 ** 22 - 1;

describe("one of two copies started together takes a data folder", () => {
  test("the first to ask has it, and the second, while the first runs, doesn't", () => {
    const where = folder();
    expect(claimStart(where, { pid: live }).held).toBe(true);
    expect(claimStart(where, { pid: live + 1 }).held).toBe(false);
  });

  test("a start after the first has let go has it", () => {
    const where = folder();
    const first = claimStart(where, { pid: live });
    first.release();
    expect(existsSync(join(where, "starting.lock"))).toBe(false);
    expect(claimStart(where, { pid: live + 1 }).held).toBe(true);
  });

  test("a claim left by a copy that is gone is taken over", () => {
    const where = folder();
    writeFileSync(join(where, "starting.lock"), String(gone));
    expect(claimStart(where, { pid: live }).held).toBe(true);
    expect(readFileSync(join(where, "starting.lock"), "utf8")).toBe(String(live));
  });

  test("a claim older than a start takes is taken over, whoever holds that process id now", () => {
    const where = folder();
    const file = join(where, "starting.lock");
    writeFileSync(file, String(live));
    const old = new Date(Date.now() - 60_000);
    utimesSync(file, old, old);
    expect(claimStart(where, { pid: live + 1 }).held).toBe(true);
  });

  test("a copy that lost can't take the claim away from the one that holds it", () => {
    const where = folder();
    claimStart(where, { pid: live });
    claimStart(where, { pid: live + 1 }).release();
    expect(readFileSync(join(where, "starting.lock"), "utf8")).toBe(String(live));
  });

  test("a claim with no process id in it is taken over", () => {
    const where = folder();
    writeFileSync(join(where, "starting.lock"), "");
    expect(claimStart(where, { pid: live }).held).toBe(true);
  });
});
