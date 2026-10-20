import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

const SQLITE_BUSY = 5;

/** How long a copy keeps asking while the folder reads as taken: two asking at once can each refuse the other. */
const ASK_FOR_MS = 400;

export interface InstanceLock {
  held: boolean;
  /** Lets the data folder go: at quit, or just before an update restarts the app. */
  release(): void;
}

const sleep = (ms: number) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);

/**
 * Takes the data folder for this copy of the app, for as long as it runs: a SQLite database,
 * `instance.lock`, opened in exclusive mode inside an exclusive transaction. The operating system
 * lets the lock go when the process ends, however it ends, so there is nothing stale to clean up.
 * Electron's single-instance lock isn't enough: two copies asking in the same millisecond can both
 * get it, and two cores then migrate one database (#1352). `report` gets a failure to lock that
 * isn't another copy holding it; the app then starts, as it did without the lock.
 */
export function lockInstance(
  folder: string,
  report: (what: string, error: unknown) => void,
  askForMs = ASK_FOR_MS,
): InstanceLock {
  mkdirSync(folder, { recursive: true });
  const give = Date.now() + askForMs;
  for (;;) {
    let db: DatabaseSync | null = null;
    try {
      db = new DatabaseSync(join(folder, "instance.lock"));
      db.exec("PRAGMA busy_timeout = 0");
      db.exec("PRAGMA locking_mode = EXCLUSIVE");
      db.exec("BEGIN EXCLUSIVE");
      return holding(db, report);
    } catch (error) {
      const busy = (error as { errcode?: number }).errcode === SQLITE_BUSY;
      if (!busy) report("app.instanceLockFailed", error);
      try {
        db?.close();
      } catch (closing) {
        report("app.instanceLockReleaseFailed", closing);
      }
      if (!busy) return { held: true, release: () => undefined };
      if (Date.now() >= give) return { held: false, release: () => undefined };
      // Another start letting go of what it read can free the folder; a random wait keeps two
      // starts from asking in step again.
      sleep(1 + Math.random() * 5);
    }
  }
}

function holding(db: DatabaseSync, report: (what: string, error: unknown) => void): InstanceLock {
  let open = true;
  return {
    held: true,
    release() {
      if (!open) return;
      open = false;
      try {
        db.close();
      } catch (error) {
        report("app.instanceLockReleaseFailed", error);
      }
    },
  };
}
