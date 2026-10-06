import { randomUUID } from "node:crypto";
import type { Folder } from "./api";
import { InvalidInputError, isRecord, NotFoundError } from "./errors";
import type { Database } from "./storage";

const MAX_NAME_LENGTH = 500;

interface FolderRow {
  id: string;
  parent_id: string | null;
  name: string;
  created_at: string;
  updated_at: string;
}

const COLUMNS = "id, parent_id, name, created_at, updated_at";

const toFolder = (row: FolderRow): Folder => ({
  id: row.id,
  name: row.name,
  parentId: row.parent_id,
  createdAt: row.created_at,
  updatedAt: row.updated_at,
});

/**
 * A Folder and every Folder below it, at any depth. UNION rather than UNION ALL,
 * so a cycle (which moves refuse, but a later sync could bring) still ends.
 */
const SUBTREE = `
  WITH RECURSIVE subtree (id) AS (
    SELECT id FROM folders WHERE id = ? AND deleted_at IS NULL
    UNION
    SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
    WHERE folders.deleted_at IS NULL
  )
  SELECT id FROM subtree`;

export function parseFolderId(id: unknown): string {
  if (typeof id !== "string" || id === "") {
    throw new InvalidInputError("A Folder id must be a non-empty string.");
  }
  return id;
}

/** A parent Folder: an id, or null for the top level. */
function parseParentId(parentId: unknown): string | null {
  return parentId === null || parentId === undefined ? null : parseFolderId(parentId);
}

function parseName(name: unknown): string {
  if (typeof name !== "string") throw new InvalidInputError("A Folder's name must be text.");
  const trimmed = name.trim();
  if (!trimmed) throw new InvalidInputError("A Folder's name can't be empty.");
  if (trimmed.length > MAX_NAME_LENGTH) {
    throw new InvalidInputError(
      `A Folder's name can't be longer than ${MAX_NAME_LENGTH} characters.`,
    );
  }
  return trimmed;
}

function parseCreateInput(input: unknown): { name: string; parentId: string | null } {
  if (!isRecord(input)) throw new InvalidInputError("createFolder expects an object.");
  return { name: parseName(input.name), parentId: parseParentId(input.parentId) };
}

export function createFolders(db: Database, now: () => string) {
  const get = (folderId: unknown): Folder => {
    const row = db.get<FolderRow>(
      `SELECT ${COLUMNS} FROM folders WHERE id = ? AND deleted_at IS NULL`,
      [parseFolderId(folderId)],
    );
    if (!row) throw new NotFoundError("That Folder doesn't exist or has been deleted.");
    return toFolder(row);
  };

  /** The ids of a live Folder and all its live sub-Folders, at any depth. */
  const subtree = (folderId: string): string[] =>
    db.all<{ id: string }>(SUBTREE, [folderId]).map((row) => row.id);

  return {
    get,
    subtree,

    create(input: unknown): Folder {
      const { name, parentId } = parseCreateInput(input);
      return db.transaction(() => {
        if (parentId !== null) get(parentId);
        const id = randomUUID();
        const at = now();
        db.run(
          "INSERT INTO folders (id, parent_id, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
          [id, parentId, name, at, at],
        );
        return { id, name, parentId, createdAt: at, updatedAt: at };
      });
    },

    list(): Folder[] {
      return db
        .all<FolderRow>(
          `SELECT ${COLUMNS} FROM folders WHERE deleted_at IS NULL
           ORDER BY name COLLATE NOCASE, created_at, rowid`,
        )
        .map(toFolder);
    },

    rename(folderId: unknown, nameInput: unknown): Folder {
      const folder = get(folderId);
      const name = parseName(nameInput);
      const at = now();
      db.run("UPDATE folders SET name = ?, updated_at = ? WHERE id = ?", [name, at, folder.id]);
      return { ...folder, name, updatedAt: at };
    },

    /** Reparents a Folder. Refuses a move into itself or below itself, which would make a cycle. */
    move(folderId: unknown, parentInput: unknown): Folder {
      const parentId = parseParentId(parentInput);
      return db.transaction(() => {
        const folder = get(folderId);
        if (parentId !== null) {
          get(parentId);
          if (subtree(folder.id).includes(parentId)) {
            throw new InvalidInputError(
              "A Folder can't be moved into itself or into one of its own sub-Folders.",
            );
          }
        }
        if (parentId === folder.parentId) return folder;
        const at = now();
        db.run("UPDATE folders SET parent_id = ?, updated_at = ? WHERE id = ?", [
          parentId,
          at,
          folder.id,
        ]);
        return { ...folder, parentId, updatedAt: at };
      });
    },

    /**
     * Marks a Folder and all its sub-Folders deleted at `at`, and returns their
     * ids. The Documents filed in them are the caller's to unfile.
     */
    delete(folderId: unknown, at: string): string[] {
      return db.transaction(() => {
        const ids = subtree(get(folderId).id);
        db.run(
          `UPDATE folders SET deleted_at = ?, updated_at = ?
           WHERE id IN (SELECT value FROM json_each(?))`,
          [at, at, JSON.stringify(ids)],
        );
        return ids;
      });
    },
  };
}
