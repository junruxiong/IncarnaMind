import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** A copy that took the claim this recently, and still runs, was started together with this one. */
const STARTED_TOGETHER_MS = 15_000;

const FILE = "starting.lock";

const isRunning = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
};

export interface StartClaim {
  held: boolean;
  /** Lets the next start have it; only the copy that holds it takes it away. */
  release(): void;
}

/**
 * Which of two copies started at the same moment on one data folder goes on. Electron's
 * single-instance lock lets both of them take it when they ask in the same millisecond (#1352),
 * and two cores then migrate one database. The file is made exclusively: one copy makes it, the
 * other finds it. A claim older than a start takes, or left by a copy that is gone, is taken over.
 */
export function claimStart(
  folder: string,
  { pid = process.pid, now = Date.now() }: { pid?: number; now?: number } = {},
): StartClaim {
  mkdirSync(folder, { recursive: true });
  const file = join(folder, FILE);
  const release = () => {
    try {
      if (readFileSync(file, "utf8").trim() === String(pid)) rmSync(file, { force: true });
    } catch {
      // Already gone: nothing to let go of.
    }
  };
  try {
    writeFileSync(file, String(pid), { flag: "wx" });
    return { held: true, release };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
  }
  const holder = Number.parseInt(readFileSync(file, "utf8"), 10);
  const recent = now - statSync(file).mtimeMs < STARTED_TOGETHER_MS;
  if (recent && Number.isInteger(holder) && holder !== pid && isRunning(holder)) {
    return { held: false, release: () => undefined };
  }
  writeFileSync(file, String(pid));
  return { held: true, release };
}
