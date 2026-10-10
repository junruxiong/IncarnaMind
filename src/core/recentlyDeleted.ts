/**
 * Recently deleted (CONTEXT.md), and putting back what was deleted, merged or untagged.
 *
 * Deleting marks rows deleted (ADR-0003), a Mind, Document, Folder or Tag together with the
 * rows that go with it, all at the same moment: restoring one brings back the rows whose
 * `deleted_at` is its own. Deleting permanently, by the User or `RECENTLY_DELETED_DAYS` after
 * at start-up, drops that content and keeps the row as a tombstone (`purged_at`), which can't
 * be restored. Nothing here reads or writes the User's files.
 */
import type { MessageKey, MessageParams } from "../shared/i18n";
import {
  type DeletedItem,
  type DeletedKind,
  type DeletedRef,
  type DocumentKind,
  RECENTLY_DELETED_DAYS,
  type RestoreRef,
  type TagLinkState,
} from "./api";
import type { createDocuments } from "./documents";
import { InvalidInputError, isRecord, NotFoundError } from "./errors";
import type { createLibrary } from "./library";
import type { createMindContent } from "./mindContent";
import type { createMinds } from "./minds";
import type { Database } from "./storage";
import { parseTagLinkStates, sameName, type TagsStore } from "./tags";

type Stored = "mind" | "document" | "folder" | "tag";

const TABLES: Readonly<Record<Stored, string>> = {
  mind: "minds",
  document: "documents",
  folder: "library_groups",
  tag: "tags",
};

const KINDS: readonly DeletedKind[] = ["mind", "chat", "document", "folder", "tag"];

const stored = (kind: DeletedKind): Stored => (kind === "chat" ? "mind" : kind);

const earliestRestorable = (now: string) =>
  new Date(Date.parse(now) - RECENTLY_DELETED_DAYS * 24 * 60 * 60 * 1000).toISOString();

function parseRefs(input: unknown, what: string): RestoreRef[] {
  if (!Array.isArray(input)) throw new InvalidInputError(`${what} expects a list of { kind, id }.`);
  return input.map((each): RestoreRef => {
    if (!isRecord(each) || typeof each.id !== "string" || each.id === "")
      throw new InvalidInputError(`${what} expects a list of { kind, id }.`);
    const kind = KINDS.find((known) => known === each.kind);
    if (!kind) throw new InvalidInputError(`${what} can't take a ${String(each.kind)}.`);
    if (each.links === undefined) return { kind, id: each.id };
    return { kind, id: each.id, links: parseTagLinkStates(each.links) };
  });
}

/** Drops deleted rows' content and marks them deleted permanently. Run in a transaction. */
function purge(db: Database, kind: Stored, ids: readonly string[], at: string): void {
  if (ids.length === 0) return;
  const list = JSON.stringify(ids);
  const IN = "IN (SELECT value FROM json_each(?))";
  if (kind === "mind") db.run(`DELETE FROM mind_updates WHERE mind_id ${IN}`, [list]);
  if (kind === "document") {
    db.run(`DELETE FROM passages WHERE document_id ${IN} AND deleted_at IS NOT NULL`, [list]);
    // Live text of a deleted Document is what its Linked folder kept for Citations: it stays.
    db.run(`DELETE FROM document_pages WHERE document_id ${IN} AND deleted_at IS NOT NULL`, [list]);
    db.run(`DELETE FROM document_tags WHERE document_id ${IN}`, [list]);
  }
  if (kind === "tag") db.run(`DELETE FROM document_tags WHERE tag_id ${IN}`, [list]);
  if (kind === "folder") db.run(`DELETE FROM deleted_folder_items WHERE folder_id ${IN}`, [list]);
  db.run(
    `UPDATE ${TABLES[kind]} SET purged_at = ?, updated_at = ?
     WHERE id ${IN} AND deleted_at IS NOT NULL AND purged_at IS NULL`,
    [at, at, list],
  );
}

export function purgeExpired(db: Database, now: string): void {
  const before = earliestRestorable(now);
  db.transaction(() => {
    for (const kind of Object.keys(TABLES) as Stored[]) {
      const ids = db.all<{ id: string }>(
        `SELECT id FROM ${TABLES[kind]} WHERE deleted_at < ? AND purged_at IS NULL`,
        [before],
      );
      purge(
        db,
        kind,
        ids.map((row) => row.id),
        now,
      );
    }
  });
}

interface DeletedRow {
  name: string;
  deleted_at: string | null;
  purged_at: string | null;
}

export function createRecentlyDeleted(options: {
  db: Database;
  now(): string;
  /** A message in the interface language. */
  say(key: MessageKey, params?: MessageParams): string;
  minds: ReturnType<typeof createMinds>;
  content: ReturnType<typeof createMindContent>;
  documents: ReturnType<typeof createDocuments>;
  tags: TagsStore;
  library: ReturnType<typeof createLibrary>;
  mindsChanged(): void;
  tagsChanged(): void;
  tagged(documentIds: string[]): void;
}) {
  const { db, now, say, minds, content, documents, tags, library } = options;

  const rowOf = (kind: Stored, id: string) =>
    db.get<DeletedRow>(
      `SELECT ${kind === "mind" ? "title" : "name"} AS name, deleted_at, purged_at
       FROM ${TABLES[kind]} WHERE id = ?`,
      [id],
    );

  const nameOf = (kind: Stored, row: DeletedRow) =>
    kind === "mind" && row.name === "" ? say("mind.untitled") : row.name;

  function checkDocument(id: string, name: string): void {
    const row = db.get<{ path: string; linked_folder_id: string | null }>(
      "SELECT path, linked_folder_id FROM documents WHERE id = ?",
      [id],
    );
    if (!row) return;
    if (
      row.linked_folder_id !== null &&
      !db.get("SELECT 1 FROM linked_folders WHERE id = ? AND deleted_at IS NULL", [
        row.linked_folder_id,
      ])
    )
      throw new NotFoundError(say("restore.error.unlinked", { name }));
    if (db.get("SELECT 1 FROM documents WHERE path = ? AND deleted_at IS NULL", [row.path]))
      throw new InvalidInputError(say("restore.error.documentBack", { name }));
  }

  function checkLinks(links: readonly TagLinkState[]): void {
    for (const { documentId, tagId } of links) {
      if (!db.get("SELECT 1 FROM documents WHERE id = ? AND deleted_at IS NULL", [documentId]))
        throw new NotFoundError(say("restore.error.documentGone"));
      const tag = rowOf("tag", tagId);
      if (!tag || tag.deleted_at !== null)
        throw new NotFoundError(say("restore.error.tagGone", { name: tag?.name ?? "" }));
    }
  }

  return {
    list(): DeletedItem[] {
      const since = earliestRestorable(now());
      const item = (
        kind: DeletedKind,
        row: { id: string; name: string; deleted_at: string },
        document?: { kind: DocumentKind; groupId: string | null; linkedFolderId: string | null },
      ): DeletedItem => ({
        kind,
        id: row.id,
        name: row.name,
        deletedAt: row.deleted_at,
        documentKind: document?.kind ?? null,
        groupId: document?.groupId ?? null,
        linkedFolderId: document?.linkedFolderId ?? null,
      });
      const items = [
        ...db
          .all<{ id: string; name: string; kind: string; deleted_at: string }>(
            "SELECT id, title AS name, kind, deleted_at FROM minds WHERE deleted_at >= ? AND purged_at IS NULL",
            [since],
          )
          .map((row) => item(row.kind === "chat" ? "chat" : "mind", row)),
        ...db
          .all<{
            id: string;
            name: string;
            kind: DocumentKind;
            deleted_at: string;
            group_id: string | null;
            linked_folder_id: string | null;
          }>(
            // Not those unlinked with their Linked folder, nor those whose file is in the Library again.
            `SELECT d.id, d.name, d.kind, d.deleted_at, g.group_id, d.linked_folder_id
             FROM documents d
             LEFT JOIN document_groups g ON g.document_id = d.id AND g.deleted_at IS NULL
             WHERE d.deleted_at >= ? AND d.purged_at IS NULL
               AND (d.linked_folder_id IS NULL OR EXISTS (
                 SELECT 1 FROM linked_folders l WHERE l.id = d.linked_folder_id AND l.deleted_at IS NULL))
               AND NOT EXISTS (SELECT 1 FROM documents o WHERE o.path = d.path AND o.deleted_at IS NULL)`,
            [since],
          )
          .map((row) =>
            item("document", row, {
              kind: row.kind,
              groupId: row.group_id,
              linkedFolderId: row.linked_folder_id,
            }),
          ),
        ...db
          .all<{ id: string; name: string; deleted_at: string }>(
            "SELECT id, name, deleted_at FROM library_groups WHERE deleted_at >= ? AND purged_at IS NULL",
            [since],
          )
          .map((row) => item("folder", row)),
        ...db
          .all<{ id: string; name: string; deleted_at: string }>(
            "SELECT id, name, deleted_at FROM tags WHERE deleted_at >= ? AND purged_at IS NULL",
            [since],
          )
          .map((row) => item("tag", row)),
      ];
      return items.sort((a, b) => b.deletedAt.localeCompare(a.deletedAt));
    },

    /** See `CoreApi.restore`. */
    restore(input: unknown): void {
      const refs = parseRefs(input, "restore");
      const since = earliestRestorable(now());
      const done = db.transaction(() => {
        const restored = {
          minds: false,
          documents: [] as string[],
          folders: false,
          tags: false,
          tagged: new Set<string>(),
        };
        for (const ref of refs) {
          const kind = stored(ref.kind);
          const row = rowOf(kind, ref.id);
          if (!row) throw new NotFoundError(say("restore.error.unknown"));
          const name = nameOf(kind, row);
          if (row.deleted_at !== null) {
            if (row.purged_at !== null)
              throw new NotFoundError(say("restore.error.deletedPermanently", { name }));
            if (row.deleted_at < since)
              throw new NotFoundError(
                say("restore.error.expired", { name, days: RECENTLY_DELETED_DAYS }),
              );
            if (kind === "mind") {
              minds.restore(ref.id);
              content.restore(ref.id, row.deleted_at);
              restored.minds = true;
            } else if (kind === "document") {
              checkDocument(ref.id, name);
              documents.restore([ref.id]);
              restored.documents.push(ref.id);
            } else if (kind === "folder") {
              library.restore(ref.id);
              restored.folders = true;
            } else {
              if (tags.list().some((tag) => sameName(tag.name, row.name)))
                throw new InvalidInputError(say("restore.error.tagNameTaken", { name }));
              for (const id of tags.restore(ref.id, row.deleted_at)) restored.tagged.add(id);
              restored.tags = true;
            }
          }
          if (ref.links) {
            checkLinks(ref.links);
            for (const id of tags.revert(ref.links)) restored.tagged.add(id);
          }
        }
        return restored;
      });
      if (done.minds) options.mindsChanged();
      if (done.documents.length > 0) {
        documents.restored(done.documents);
        library.announce(done.documents);
      }
      if (done.folders) library.restored();
      if (done.tags) options.tagsChanged();
      if (done.tagged.size > 0) options.tagged([...done.tagged]);
    },

    /** See `CoreApi.revertDocumentTags`. Returns the Documents whose Tags changed. */
    revertTags(input: unknown): string[] {
      const links = parseTagLinkStates(input);
      return db.transaction(() => {
        checkLinks(links);
        return tags.revert(links);
      });
    },

    /** See `CoreApi.deletePermanently`. */
    deletePermanently(input: unknown): void {
      const refs: DeletedRef[] = parseRefs(input, "deletePermanently");
      const at = now();
      db.transaction(() => {
        for (const ref of refs) {
          const kind = stored(ref.kind);
          const row = rowOf(kind, ref.id);
          if (!row) throw new NotFoundError(say("restore.error.unknown"));
          if (row.deleted_at === null)
            throw new InvalidInputError(
              say("deletePermanently.error.notDeleted", { name: nameOf(kind, row) }),
            );
          purge(db, kind, [ref.id], at);
        }
      });
    },
  };
}
