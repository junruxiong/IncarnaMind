/**
 * A lock that processes on this computer respect: a file created only if it
 * doesn't exist yet, holding its owner's token, and removed by that owner
 * when the work is done. A lock whose process died is taken over once it is
 * older than `staleMs`, which must stay well above the longest task it guards.
 */
import { randomUUID } from "node:crypto";
import { open, readFile, rm, stat } from "node:fs/promises";

export interface FileLock {
  /** Runs `task` while holding the lock, waiting for it first. */
  run<T>(task: () => Promise<T>): Promise<T>;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function createFileLock(
  path: string,
  options: { staleMs?: number; retryMs?: number; waitMs?: number } = {},
): FileLock {
  const { staleMs = 60_000, retryMs = 100, waitMs = 2 * 60_000 } = options;

  const acquire = async (owner: string) => {
    const started = Date.now();
    for (;;) {
      try {
        const handle = await open(path, "wx", 0o600);
        await handle.writeFile(owner);
        await handle.close();
        return;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      }
      const held = await stat(path).catch(() => null);
      if (held && Date.now() - held.mtimeMs > staleMs) {
        // Left by a process that stopped before it let go. Removed only if it is still that
        // same file: another waiter may have taken it over meanwhile.
        const again = await stat(path).catch(() => null);
        if (again && again.ino === held.ino && again.mtimeMs === held.mtimeMs) {
          await rm(path, { force: true });
        }
        continue;
      }
      if (Date.now() - started > waitMs) {
        throw new Error(`Another process has held ${path} for too long.`);
      }
      await sleep(retryMs);
    }
  };

  return {
    async run(task) {
      const owner = `${process.pid}:${randomUUID()}`;
      await acquire(owner);
      try {
        return await task();
      } finally {
        // A lock another process took over as stale is theirs now: it stays.
        const current = await readFile(path, "utf8").catch(() => null);
        if (current === owner) await rm(path, { force: true });
      }
    },
  };
}
