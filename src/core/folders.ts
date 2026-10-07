/**
 * Folders (CONTEXT.md): the folders inside Linked folders, as they are on
 * disk. Each Linked folder has a Folder for itself, and one for each folder
 * inside it that holds a Document at some depth. The User doesn't create,
 * rename or move them: they follow the disk, at each scan. A file added on
 * its own is in no Folder.
 *
 * A Folder's id is derived from its Linked folder's id and its path relative
 * to it, so a scan always gives the same folder the same id, and a Search
 * scope naming it keeps working. A folder renamed on disk is a new Folder.
 */
import { createHash } from "node:crypto";
import { basename } from "node:path";
import type { Folder } from "./api";
import { InvalidInputError, NotFoundError } from "./errors";
import type { Database } from "./storage";

interface FolderRow {
  id: string;
  parent_id: string | null;
  name: string;
  linked_folder_id: string;
  relative_path: string;
  created_at: string;
  updated_at: string;
  deleted_at?: string | null;
}

const COLUMNS = "id, parent_id, name, linked_folder_id, relative_path, created_at, updated_at";

const toFolder = (row: FolderRow): Folder => ({
  id: row.id,
  name: row.name,
  parentId: row.parent_id,
  linkedFolderId: row.linked_folder_id,
  relativePath: row.relative_path,
  createdAt: row.created_at,
  updatedAt: row.updated_at,
});

/**
 * A Folder and every Folder below it, at any depth. UNION rather than UNION ALL,
 * so a cycle (which a later sync could bring) still ends.
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

/**
 * The id of the Folder at `relative` ("" for the Linked folder itself, "/"
 * between names) in a Linked folder: the same at every scan. Shaped like a
 * UUID (version 8, RFC 9562: custom), from the SHA-256 of both.
 */
export function folderIdFor(linkedFolderId: string, relative: string): string {
  const hex = createHash("sha256").update(`${linkedFolderId}\n${relative}`).digest("hex");
  const variant = ((Number.parseInt(hex[16] as string, 16) & 0x3) | 0x8).toString(16);
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    `8${hex.slice(13, 16)}`,
    `${variant}${hex.slice(17, 20)}`,
    hex.slice(20, 32),
  ].join("-");
}

/** The relative paths of `relative`'s ancestors and itself: "a/b" gives "", "a", "a/b". */
function withAncestors(relative: string): string[] {
  const paths = [""];
  if (relative === "") return paths;
  const names = relative.split("/");
  for (let at = 1; at <= names.length; at++) paths.push(names.slice(0, at).join("/"));
  return paths;
}

export type Folders = ReturnType<typeof createFolders>;

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

    list(): Folder[] {
      return db
        .all<FolderRow>(
          `SELECT ${COLUMNS} FROM folders WHERE deleted_at IS NULL
           ORDER BY name COLLATE NOCASE, created_at, rowid`,
        )
        .map(toFolder);
    },

    /**
     * Makes a Linked folder's Folders those at `relatives` (folders holding a
     * Document, relative to the Linked folder) and their ancestors, plus the
     * Linked folder's own Folder: missing ones are created (or brought back),
     * the others marked deleted. Returns whether anything changed.
     */
    sync(linkedFolderId: string, rootPath: string, relatives: Iterable<string>): boolean {
      const wanted = new Map<string, string>(); // id -> relative path
      for (const relative of relatives) {
        for (const path of withAncestors(relative)) {
          wanted.set(folderIdFor(linkedFolderId, path), path);
        }
      }
      wanted.set(folderIdFor(linkedFolderId, ""), "");
      return db.transaction(() => {
        const at = now();
        const rows = new Map(
          db
            .all<FolderRow>(
              `SELECT ${COLUMNS}, deleted_at FROM folders WHERE linked_folder_id = ?`,
              [linkedFolderId],
            )
            .map((row) => [row.id, row]),
        );
        let changed = false;
        for (const [id, relative] of wanted) {
          const name = relative === "" ? basename(rootPath) || rootPath : basename(relative);
          const parentId =
            relative === ""
              ? null
              : folderIdFor(
                  linkedFolderId,
                  relative.includes("/") ? relative.slice(0, relative.lastIndexOf("/")) : "",
                );
          const row = rows.get(id);
          if (!row) {
            db.run(
              `INSERT INTO folders (id, parent_id, name, linked_folder_id, relative_path, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)`,
              [id, parentId, name, linkedFolderId, relative, at, at],
            );
            changed = true;
          } else if (row.deleted_at || row.name !== name || row.parent_id !== parentId) {
            db.run(
              `UPDATE folders SET deleted_at = NULL, name = ?, parent_id = ?, updated_at = ? WHERE id = ?`,
              [name, parentId, at, id],
            );
            changed = true;
          }
        }
        for (const row of rows.values()) {
          if (row.deleted_at || wanted.has(row.id)) continue;
          db.run("UPDATE folders SET deleted_at = ?, updated_at = ? WHERE id = ?", [
            at,
            at,
            row.id,
          ]);
          changed = true;
        }
        return changed;
      });
    },

    /** Marks every Folder of a Linked folder deleted at `at`, e.g. when it is removed. */
    removeAll(linkedFolderId: string, at: string): boolean {
      const before = db.get<{ count: number }>(
        "SELECT count(*) AS count FROM folders WHERE linked_folder_id = ? AND deleted_at IS NULL",
        [linkedFolderId],
      );
      db.run(
        `UPDATE folders SET deleted_at = ?, updated_at = ?
         WHERE linked_folder_id = ? AND deleted_at IS NULL`,
        [at, at, linkedFolderId],
      );
      return (before?.count ?? 0) > 0;
    },
  };
}
