import { randomUUID } from "node:crypto";
import type { CreateMindInput, Mind } from "./api";
import { InvalidInputError, isRecord, NotFoundError } from "./errors";
import type { Database } from "./storage";

/**
 * Editing a Mind's content moves its `updatedAt` at most this often, so typing
 * doesn't rewrite the row and reorder the list on every keystroke. Editing a
 * different Mind in between always moves it, so the order stays right.
 */
export const EDIT_TOUCH_INTERVAL_MS = 10_000;

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

function parseTitle(title: unknown): string {
  if (typeof title !== "string") throw new InvalidInputError("A Mind's title must be text.");
  return title.trim();
}

function parseCreateMindInput(input: unknown): Required<CreateMindInput> {
  if (input === undefined) return { title: "" };
  if (!isRecord(input)) throw new InvalidInputError("createMind expects an object.");
  const { title = "" } = input;
  return { title: parseTitle(title) };
}

export function parseMindId(mindId: unknown): string {
  if (typeof mindId !== "string" || mindId === "") {
    throw new InvalidInputError("A Mind id must be a non-empty string.");
  }
  return mindId;
}

export function createMinds(db: Database, now: () => string) {
  /** The Mind whose `updatedAt` was set last, and when: the edit throttle compares against it. */
  let lastTouched: { mindId: string; time: number } | null = null;
  const touched = (mindId: string, at: string) => {
    lastTouched = { mindId, time: Date.parse(at) };
  };

  const get = (mindId: unknown): Mind => {
    const row = db.get<MindRow>(
      "SELECT id, title, created_at, updated_at FROM minds WHERE id = ? AND deleted_at IS NULL",
      [parseMindId(mindId)],
    );
    if (!row) throw new NotFoundError("That Mind doesn't exist or has been deleted.");
    return toMind(row);
  };

  return {
    get,

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
      touched(id, at);
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

    rename(mindId: unknown, title: unknown): Mind {
      const mind = get(mindId);
      const newTitle = parseTitle(title);
      const at = now();
      db.run("UPDATE minds SET title = ?, updated_at = ? WHERE id = ?", [newTitle, at, mind.id]);
      touched(mind.id, at);
      return { ...mind, title: newTitle, updatedAt: at };
    },

    /** Marks the Mind deleted at `at`. Its content rows are the caller's to mark. */
    delete(mindId: unknown, at: string): Mind {
      const mind = get(mindId);
      db.run("UPDATE minds SET deleted_at = ?, updated_at = ? WHERE id = ?", [at, at, mind.id]);
      if (lastTouched?.mindId === mind.id) lastTouched = null;
      return mind;
    },

    /**
     * Records that the Mind's content was edited: moves its `updatedAt` unless
     * that happened less than EDIT_TOUCH_INTERVAL_MS ago with no other Mind
     * touched since. Returns whether it moved.
     */
    markEdited(mindId: string): boolean {
      const at = now();
      const time = Date.parse(at);
      if (lastTouched?.mindId === mindId && time - lastTouched.time < EDIT_TOUCH_INTERVAL_MS) {
        return false;
      }
      db.run("UPDATE minds SET updated_at = ? WHERE id = ? AND deleted_at IS NULL", [at, mindId]);
      touched(mindId, at);
      return true;
    },
  };
}
