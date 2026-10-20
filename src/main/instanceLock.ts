import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

const SQLITE_BUSY = 5;

// A folder that stays busy with nobody holding it is a lock this can't make sense of: the app
// starts, as it did before the lock, rather than never opening.
const GIVE_UP_AFTER_MS = 10_000;

export interface InstanceLock {
  held: boolean;
  release(): void;
}

const sleep = (ms: number) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);

const isBusy = (error: unknown) =>
  ((error as { errcode?: number }).errcode ?? 0) % 256 === SQLITE_BUSY;

/** Opens `file` and takes it exclusively until the connection is closed. */
function takeExclusively(file: string): DatabaseSync {
  const db = new DatabaseSync(file);
  try {
    db.exec("PRAGMA busy_timeout = 0");
    db.exec("PRAGMA locking_mode = EXCLUSIVE");
    db.exec("BEGIN EXCLUSIVE");
    return db;
  } catch (error) {
    db.close();
    throw error;
  }
}

/** Whether another copy holds `file` exclusively: a read, which takes only a shared lock, is refused. */
function isHeld(file: string): boolean {
  const db = new DatabaseSync(file);
  try {
    db.exec("PRAGMA busy_timeout = 0");
    db.exec("BEGIN");
    db.prepare("SELECT count(*) FROM sqlite_master").get();
    return false;
  } catch (error) {
    if (isBusy(error)) return true;
    throw error;
  } finally {
    db.close();
  }
}

/**
 * Takes the data folder for this copy of the app, for as long as it runs: `instance.lock`, a
 * SQLite database opened in exclusive mode inside an exclusive transaction. The operating system
 * lets the lock go when the process ends, however it ends, so there is nothing stale to clean up.
 * Electron's single-instance lock isn't enough: two copies asking in the same millisecond can both
 * get it, and two cores then migrate one database (#1352).
 *
 * The copy that holds `instance.lock` also takes `instance.owner`, and nothing else does. A copy
 * that finds `instance.lock` busy reads `instance.owner`: held, a copy really has the folder and
 * this one gives up; free, the busy lock was another copy's try, which lets go in a moment, so
 * this one asks again. That way exactly one of two copies holds the folder, however the two are
 * scheduled. `report` gets a failure to lock that isn't another copy holding it; the app then
 * starts, as it did without the lock.
 */
export function lockInstance(
  folder: string,
  report: (what: string, error: unknown) => void,
): InstanceLock {
  const give = Date.now() + GIVE_UP_AFTER_MS;
  const held: DatabaseSync[] = [];
  try {
    mkdirSync(folder, { recursive: true });
    for (;;) {
      try {
        held.push(takeExclusively(join(folder, "instance.lock")));
        held.push(takeExclusively(join(folder, "instance.owner")));
        return holding(held, report);
      } catch (error) {
        for (const db of held.splice(0)) db.close();
        if (!isBusy(error)) throw error;
        if (isHeld(join(folder, "instance.owner")))
          return { held: false, release: () => undefined };
        if (Date.now() >= give) throw error;
        sleep(1 + Math.random() * 5);
      }
    }
  } catch (error) {
    for (const db of held.splice(0)) db.close();
    report("app.instanceLockFailed", error);
    return { held: true, release: () => undefined };
  }
}

function holding(
  databases: DatabaseSync[],
  report: (what: string, error: unknown) => void,
): InstanceLock {
  let open = true;
  return {
    held: true,
    release() {
      if (!open) return;
      open = false;
      for (const db of databases) {
        try {
          db.close();
        } catch (error) {
          report("app.instanceLockReleaseFailed", error);
        }
      }
    },
  };
}
