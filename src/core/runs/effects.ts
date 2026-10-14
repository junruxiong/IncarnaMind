/**
 * A Task's Effects (#126): what its steps change outside IncarnaMind's
 * database, and how a step whose outcome a crash left unknown is checked
 * before anything is done again. Nothing is replayed blindly: a file is read
 * back, and what can't be read back (a Connector's change) is asked about.
 */
import { createHash } from "node:crypto";
import { open, readFile, rename } from "node:fs/promises";
import type { TaskEffect } from "../api";

/** What checking an Effect found: it happened, it didn't, or IncarnaMind can't tell. */
export type EffectVerdict = "done" | "not-done" | "unknown";

const sha256 = (data: string | Uint8Array) => createHash("sha256").update(data).digest("hex");

async function hashOf(path: string): Promise<string | null> {
  try {
    return sha256(await readFile(path));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }
}

/**
 * Whether an Effect happened. A file written holds what the step wrote
 * (done), or what was there before, or still nothing (not done); anything
 * else, someone else's change, can't be told. A Connector's change can't be
 * read back.
 */
export async function verifyEffect(effect: TaskEffect): Promise<EffectVerdict> {
  if (effect.kind !== "file-write") return "unknown";
  const now = await hashOf(effect.path);
  if (now === effect.sha256) return "done";
  if (now === effect.previous) return "not-done";
  return "unknown";
}

/**
 * A whole file to write as a step's Effect, and the write itself: to a
 * partial file beside it, synced, then renamed over it, so the file holds
 * either what was there or all of what is written, never half of it.
 */
export async function fileWrite(
  path: string,
  contents: string | Uint8Array,
): Promise<{ effect: TaskEffect; perform(): Promise<void> }> {
  const effect: TaskEffect = {
    kind: "file-write",
    path,
    sha256: sha256(contents),
    previous: await hashOf(path),
  };
  return {
    effect,
    async perform() {
      const partial = `${path}.incarnamind-partial`;
      const handle = await open(partial, "w");
      try {
        await handle.writeFile(contents);
        await handle.sync();
      } finally {
        await handle.close();
      }
      await rename(partial, path);
    },
  };
}
