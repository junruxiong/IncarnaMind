import { randomUUID } from "node:crypto";
import type { CreateMindInput, Mind } from "./api";
import { InvalidInputError, isRecord } from "./errors";
import type { Database } from "./storage";

interface MindRow {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

const toMind = (row: MindRow): Mind => ({
  id: row.id,
  title: row.title,
  createdAt: row.created_at,
  updatedAt: row.updated_at,
});

function parseCreateMindInput(input: unknown): Required<CreateMindInput> {
  if (input === undefined) return { title: "" };
  if (!isRecord(input)) throw new InvalidInputError("createMind expects an object.");
  const { title = "" } = input;
  if (typeof title !== "string") throw new InvalidInputError("A Mind's title must be text.");
  return { title: title.trim() };
}

export function createMinds(db: Database, now: () => string) {
  return {
    create(input: unknown): Mind {
      const { title } = parseCreateMindInput(input);
      const id = randomUUID();
      const at = now();
      db.run("INSERT INTO minds (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)", [
        id,
        title,
        at,
        at,
      ]);
      return { id, title, createdAt: at, updatedAt: at };
    },

    list(): Mind[] {
      return db
        .all<MindRow>(
          `SELECT id, title, created_at, updated_at FROM minds
           WHERE deleted_at IS NULL
           ORDER BY updated_at DESC, rowid DESC`,
        )
        .map(toMind);
    },
  };
}
