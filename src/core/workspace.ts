/**
 * The workspace (ADR-0020): everything one data folder holds, known by one
 * random id made at its first start. No account is behind it and nothing is
 * sent: a later sync between a User's computers will tell its data apart from
 * another workspace's by this id, which they will share.
 */
import { randomUUID } from "node:crypto";
import type { Database } from "./storage";

/** The workspace's id, made once: the same at every start, and in a copy of the data folder. */
export function ensureWorkspace(db: Database, now: () => string): string {
  return db.transaction(() => {
    const live = db.get<{ id: string }>(
      "SELECT id FROM workspace WHERE deleted_at IS NULL ORDER BY created_at, id LIMIT 1",
    );
    if (live) return live.id;
    const id = randomUUID();
    const at = now();
    db.run("INSERT INTO workspace (id, created_at, updated_at) VALUES (?, ?, ?)", [id, at, at]);
    return id;
  });
}
