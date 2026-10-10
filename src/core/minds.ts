import { randomUUID } from "node:crypto";
import { type CreateMindInput, DEFAULT_MIND_KIND, type Mind, type MovedItem } from "./api";
import { InvalidInputError, isRecord, NotFoundError } from "./errors";
import { parseKind } from "./mindAccess";
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
  kind: string;
  created_at: string;
  updated_at: string;
  folder_id: string | null;
}

const COLUMNS = "id, title, kind, created_at, updated_at, folder_id";

const toMind = (row: MindRow): Mind => ({
  id: row.id,
  title: row.title,
  kind: row.kind,
  createdAt: row.created_at,
  updatedAt: row.updated_at,
  folderId: row.folder_id,
});

function parseTitle(title: unknown): string {
  if (typeof title !== "string") throw new InvalidInputError("A Mind's title must be text.");
  return title.trim();
}

/** A Folder's id, or null for Not in a Folder. Whether it exists is the caller's to check. */
export function parseFolderChoice(folderId: unknown): string | null {
  if (folderId === null || folderId === undefined) return null;
  if (typeof folderId !== "string" || folderId === "") {
    throw new InvalidInputError("A Folder id must be a non-empty string, or null for none.");
  }
  return folderId;
}

export function parseCreateMindInput(input: unknown): Required<CreateMindInput> {
  if (input === undefined) return { title: "", kind: DEFAULT_MIND_KIND, folderId: null };
  if (!isRecord(input)) throw new InvalidInputError("createMind expects an object.");
  const { title = "", kind, folderId } = input;
  const parsedKind = parseKind(kind);
  if (parsedKind === null) throw new InvalidInputError("That kind of Mind isn't known.");
  return { title: parseTitle(title), kind: parsedKind, folderId: parseFolderChoice(folderId) };
}

/** A list of ids, each once, in order. */
export function parseIds(ids: unknown, what: string): string[] {
  if (ids === undefined) return [];
  if (!Array.isArray(ids) || !ids.every((id) => typeof id === "string" && id !== "")) {
    throw new InvalidInputError(`${what} must be a list of ids.`);
  }
  return [...new Set(ids as string[])];
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
      `SELECT ${COLUMNS} FROM minds WHERE id = ? AND deleted_at IS NULL`,
      [parseMindId(mindId)],
    );
    if (!row) throw new NotFoundError("That Mind doesn't exist or has been deleted.");
    return toMind(row);
  };

  return {
    get,

    /** Makes a Mind; its Folder, if any, is the caller's to have checked. */
    create(input: Required<CreateMindInput>): Mind {
      const { title, kind, folderId } = input;
      const id = randomUUID();
      const at = now();
      db.run(
        "INSERT INTO minds (id, title, kind, created_at, updated_at, folder_id) VALUES (?, ?, ?, ?, ?, ?)",
        [id, title, kind, at, at, folderId],
      );
      touched(id, at);
      return { id, title, kind, createdAt: at, updatedAt: at, folderId };
    },

    list(): Mind[] {
      return db
        .all<MindRow>(
          `SELECT ${COLUMNS} FROM minds
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

    /**
     * Puts these Minds in a Folder (checked by the caller), or in none, and
     * returns where each was. All must exist, or nothing moves. A Mind that
     * moves counts as changed (`updatedAt`), as a rename does.
     */
    move(mindIds: readonly string[], folderId: string | null): MovedItem[] {
      return db.transaction(() => {
        const moved = mindIds.map((id) => ({ id, from: get(id).folderId }));
        const at = now();
        for (const item of moved) {
          if (item.from === folderId) continue;
          db.run("UPDATE minds SET folder_id = ?, updated_at = ? WHERE id = ?", [
            folderId,
            at,
            item.id,
          ]);
        }
        return moved;
      });
    },

    /** A Folder was deleted: its Minds go to Not in a Folder. Returns whether there were any. */
    leaveFolder(folderId: string): boolean {
      const inIt = db.get<{ count: number }>(
        "SELECT count(*) AS count FROM minds WHERE folder_id = ? AND deleted_at IS NULL",
        [folderId],
      );
      db.run("UPDATE minds SET folder_id = NULL, updated_at = ? WHERE folder_id = ?", [
        now(),
        folderId,
      ]);
      return (inIt?.count ?? 0) > 0;
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
