import { readFile, utimes, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "vitest";
import { createFileLock } from "../../src/core/fileLock";
import { createTempDataFolder } from "../helpers/core";

test("two holders take turns, and the lock is gone afterwards", async () => {
  const path = join(await createTempDataFolder(), "refresh.lock");
  const first = createFileLock(path, { retryMs: 5 });
  const second = createFileLock(path, { retryMs: 5 });
  const order: string[] = [];
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });

  const a = first.run(async () => {
    order.push("a starts");
    await held;
    order.push("a ends");
  });
  await expect.poll(() => order).toEqual(["a starts"]);
  const b = second.run(async () => {
    order.push("b runs");
  });
  release();
  await Promise.all([a, b]);

  expect(order).toEqual(["a starts", "a ends", "b runs"]);
  await expect(readFile(path, "utf8")).rejects.toThrow();
});

test("a lock left by a process that stopped is taken over once it is stale", async () => {
  const path = join(await createTempDataFolder(), "refresh.lock");
  await writeFile(path, "4242:gone");
  const tenMinutesAgo = new Date(Date.now() - 10 * 60_000);
  await utimes(path, tenMinutesAgo, tenMinutesAgo);

  let ran = false;
  await createFileLock(path, { retryMs: 5 }).run(async () => {
    ran = true;
  });

  expect(ran).toBe(true);
  await expect(readFile(path, "utf8")).rejects.toThrow();
});

test("a holder whose lock another process took over leaves that process's lock in place", async () => {
  const path = join(await createTempDataFolder(), "refresh.lock");

  await createFileLock(path).run(async () => {
    // Another process decided this lock was stale and took it.
    await writeFile(path, "5151:other");
  });

  expect(await readFile(path, "utf8")).toBe("5151:other");
});
