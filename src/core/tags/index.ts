/**
 * Tags and the Tags on each Document (CONTEXT.md: a Tag is a label with a
 * short description; a Document can have many). This module only stores
 * them: automatic tagging lives in ./tagger.
 *
 * The rule that matters: a Tag the User added to or removed from a Document
 * is the User's from then on. Their removal is kept as a deleted link with
 * source "user" (see migration 12), and automatic tagging skips every
 * Document and Tag that has one.
 */
import { randomUUID } from "node:crypto";
import type { DocumentTag, Tag, TagSource } from "../api";
import { InvalidInputError, isRecord, NotFoundError } from "../errors";
import type { Language } from "../language";
import type { Database } from "../storage";
import type { TagDecision } from "./classify";
import { PRESET_TAGS } from "./presets";

const MAX_NAME_LENGTH = 100;
const MAX_DESCRIPTION_LENGTH = 500;

interface TagRow {
  id: string;
  name: string;
  description: string;
  preset: string | null;
  created_at: string;
  updated_at: string;
}

interface LinkRow {
  id: string;
  tag_id: string;
  source: TagSource;
  confidence: number | null;
  needs_review: number;
}

const COLUMNS = "id, name, description, preset, created_at, updated_at";

const toTag = (row: TagRow): Tag => ({
  id: row.id,
  name: row.name,
  description: row.description,
  preset: row.preset !== null,
  createdAt: row.created_at,
  updatedAt: row.updated_at,
});

/** How names are compared: two names that differ only in case are the same Tag. */
export const sameName = (a: string, b: string) => a.toLowerCase() === b.toLowerCase();

function parseTagId(id: unknown): string {
  if (typeof id !== "string" || id === "") {
    throw new InvalidInputError("A Tag id must be a non-empty string.");
  }
  return id;
}

function parseName(name: unknown): string {
  if (typeof name !== "string") throw new InvalidInputError("A Tag's name must be text.");
  const trimmed = name.trim();
  if (!trimmed) throw new InvalidInputError("A Tag's name can't be empty.");
  if (trimmed.length > MAX_NAME_LENGTH) {
    throw new InvalidInputError(`A Tag's name can't be longer than ${MAX_NAME_LENGTH} characters.`);
  }
  return trimmed;
}

function parseDescription(description: unknown): string {
  if (typeof description !== "string") {
    throw new InvalidInputError("A Tag's description must be text.");
  }
  const trimmed = description.trim();
  if (trimmed.length > MAX_DESCRIPTION_LENGTH) {
    throw new InvalidInputError(
      `A Tag's description can't be longer than ${MAX_DESCRIPTION_LENGTH} characters.`,
    );
  }
  return trimmed;
}

/**
 * The live Tags on a Document, in Tag name order. Links to a Tag that is
 * deleted don't count, whatever their own state.
 */
export function tagsOfDocument(db: Database, documentId: string): DocumentTag[] {
  return db
    .all<{ tag_id: string; source: TagSource; confidence: number | null; needs_review: number }>(
      `SELECT l.tag_id, l.source, l.confidence, l.needs_review
       FROM document_tags l JOIN tags t ON t.id = l.tag_id
       WHERE l.document_id = ? AND l.deleted_at IS NULL AND t.deleted_at IS NULL
       ORDER BY t.name COLLATE NOCASE, t.created_at, t.rowid`,
      [documentId],
    )
    .map((row) => ({
      tagId: row.tag_id,
      source: row.source,
      confidence: row.confidence,
      needsReview: row.needs_review !== 0,
    }));
}

export type TagsStore = ReturnType<typeof createTags>;

export function createTags(db: Database, now: () => string) {
  const get = (tagId: unknown): Tag => {
    const row = db.get<TagRow>(`SELECT ${COLUMNS} FROM tags WHERE id = ? AND deleted_at IS NULL`, [
      parseTagId(tagId),
    ]);
    if (!row) throw new NotFoundError("That Tag doesn't exist or has been deleted.");
    return toTag(row);
  };

  const list = (): Tag[] =>
    db
      .all<TagRow>(
        `SELECT ${COLUMNS} FROM tags WHERE deleted_at IS NULL
         ORDER BY name COLLATE NOCASE, created_at, rowid`,
      )
      .map(toTag);

  /** Refuses a name another live Tag has, ignoring case. */
  const checkUnique = (name: string, exceptId?: string) => {
    const taken = list().find((tag) => tag.id !== exceptId && sameName(tag.name, name));
    if (taken) throw new InvalidInputError(`There is already a Tag named “${taken.name}”.`);
  };

  const liveLink = (documentId: string, tagId: string) =>
    db.get<LinkRow>(
      `SELECT id, tag_id, source, confidence, needs_review FROM document_tags
       WHERE document_id = ? AND tag_id = ? AND deleted_at IS NULL`,
      [documentId, tagId],
    );

  const insertLink = (
    documentId: string,
    tagId: string,
    source: TagSource,
    at: string,
    deletedAt: string | null = null,
    decision: Pick<TagDecision, "confidence" | "needsReview"> = {
      confidence: null,
      needsReview: false,
    },
  ) =>
    db.run(
      `INSERT INTO document_tags
         (id, document_id, tag_id, source, confidence, needs_review, created_at, updated_at, deleted_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        randomUUID(),
        documentId,
        tagId,
        source,
        decision.confidence,
        decision.needsReview ? 1 : 0,
        at,
        at,
        deletedAt,
      ],
    );

  return {
    get,
    list,

    /**
     * Creates the preset Tags if no Tag has ever existed here (deleted ones
     * count, so deleting every preset doesn't bring them back). Returns
     * whether it did.
     */
    seedPresets(language: Language): boolean {
      return db.transaction(() => {
        if (db.get("SELECT 1 FROM tags LIMIT 1")) return false;
        const at = now();
        for (const preset of PRESET_TAGS) {
          const { name, description } = preset.text[language];
          db.run(
            `INSERT INTO tags (id, name, description, preset, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?)`,
            [randomUUID(), name, description, preset.key, at, at],
          );
        }
        return true;
      });
    },

    create(input: unknown): Tag {
      if (!isRecord(input)) throw new InvalidInputError("createTag expects an object.");
      const name = parseName(input.name);
      const description =
        input.description === undefined ? "" : parseDescription(input.description);
      return db.transaction(() => {
        checkUnique(name);
        const id = randomUUID();
        const at = now();
        db.run(
          `INSERT INTO tags (id, name, description, preset, created_at, updated_at)
           VALUES (?, ?, ?, NULL, ?, ?)`,
          [id, name, description, at, at],
        );
        return { id, name, description, preset: false, createdAt: at, updatedAt: at };
      });
    },

    update(tagId: unknown, patch: unknown): Tag {
      if (!isRecord(patch)) throw new InvalidInputError("updateTag expects an object.");
      for (const key of Object.keys(patch)) {
        if (key !== "name" && key !== "description") {
          throw new InvalidInputError(`A Tag has no field "${key}".`);
        }
      }
      return db.transaction(() => {
        const tag = get(tagId);
        const name = patch.name === undefined ? tag.name : parseName(patch.name);
        const description =
          patch.description === undefined ? tag.description : parseDescription(patch.description);
        if (name === tag.name && description === tag.description) return tag;
        checkUnique(name, tag.id);
        const at = now();
        db.run("UPDATE tags SET name = ?, description = ?, updated_at = ? WHERE id = ?", [
          name,
          description,
          at,
          tag.id,
        ]);
        return { ...tag, name, description, updatedAt: at };
      });
    },

    /**
     * Marks a Tag deleted at `at`, and takes it off every Document. Returns the
     * ids of the Documents that carried it.
     */
    delete(tagId: unknown, at: string): string[] {
      return db.transaction(() => {
        const tag = get(tagId);
        const documentIds = db
          .all<{ document_id: string }>(
            `SELECT document_id FROM document_tags WHERE tag_id = ? AND deleted_at IS NULL
             ORDER BY created_at, rowid`,
            [tag.id],
          )
          .map((row) => row.document_id);
        db.run("UPDATE tags SET deleted_at = ?, updated_at = ? WHERE id = ?", [at, at, tag.id]);
        db.run(
          `UPDATE document_tags SET deleted_at = ?, updated_at = ?
           WHERE tag_id = ? AND deleted_at IS NULL`,
          [at, at, tag.id],
        );
        return documentIds;
      });
    },

    /**
     * The User puts a Tag on a Document (the caller checks the Document).
     * An automatic link becomes theirs: that is how the User confirms a Tag
     * marked "needs review". Returns whether anything changed.
     */
    addToDocument(documentId: string, tagIdInput: unknown): boolean {
      return db.transaction(() => {
        const tag = get(tagIdInput);
        const at = now();
        const link = liveLink(documentId, tag.id);
        if (link?.source === "user") return false;
        if (link) {
          db.run(
            `UPDATE document_tags SET source = 'user', confidence = NULL, needs_review = 0,
               updated_at = ?
             WHERE id = ?`,
            [at, link.id],
          );
        } else {
          insertLink(documentId, tag.id, "user", at);
        }
        return true;
      });
    },

    /**
     * The User takes a Tag off a Document (the caller checks the Document).
     * The link is marked deleted as theirs, so automatic tagging leaves it off;
     * if the Tag wasn't on the Document, that decision is still recorded.
     * Returns whether the Document's Tags changed.
     */
    removeFromDocument(documentId: string, tagIdInput: unknown): boolean {
      return db.transaction(() => {
        const tag = get(tagIdInput);
        const at = now();
        const link = liveLink(documentId, tag.id);
        if (link) {
          db.run(
            `UPDATE document_tags SET source = 'user', deleted_at = ?, updated_at = ? WHERE id = ?`,
            [at, at, link.id],
          );
          return true;
        }
        const decided = db.get(
          `SELECT 1 FROM document_tags WHERE document_id = ? AND tag_id = ? AND source = 'user'`,
          [documentId, tag.id],
        );
        if (!decided) insertLink(documentId, tag.id, "user", at, at);
        return false;
      });
    },

    /**
     * Merges Tag `fromInput` into Tag `intoInput` at `at`: each Document's
     * link moves over, keeping who chose it (the User's wins where a Document
     * has both, and an automatic link keeps its confidence), and a removal the
     * User made carries over where they haven't decided on the other Tag. The
     * merged Tag is then deleted. Returns the ids of the Documents whose Tags
     * changed: those that carried the merged Tag.
     */
    merge(fromInput: unknown, intoInput: unknown, at: string): string[] {
      return db.transaction(() => {
        const from = get(fromInput);
        const into = get(intoInput);
        if (from.id === into.id) throw new InvalidInputError("A Tag can't be merged into itself.");
        const rows = db.all<LinkRow & { document_id: string; deleted: number }>(
          `SELECT id, document_id, tag_id, source, confidence, needs_review,
             deleted_at IS NOT NULL AS deleted
           FROM document_tags WHERE tag_id = ? AND (deleted_at IS NULL OR source = 'user')
           ORDER BY created_at, rowid`,
          [from.id],
        );
        const changed: string[] = [];
        for (const row of rows) {
          const target = liveLink(row.document_id, into.id);
          const decided = !!db.get(
            "SELECT 1 FROM document_tags WHERE document_id = ? AND tag_id = ? AND source = 'user'",
            [row.document_id, into.id],
          );
          if (row.deleted) {
            // The User's removal: the other Tag stays off too, unless they decided on it.
            if (!target && !decided) insertLink(row.document_id, into.id, "user", at, at);
            continue;
          }
          changed.push(row.document_id);
          if (target) {
            if (row.source === "user" && target.source !== "user")
              db.run(
                `UPDATE document_tags SET source = 'user', confidence = NULL, needs_review = 0,
                   updated_at = ? WHERE id = ?`,
                [at, target.id],
              );
          } else if (row.source === "user" || !decided) {
            // An automatic link doesn't come back where the User took the other Tag off.
            insertLink(row.document_id, into.id, row.source, at, null, {
              confidence: row.confidence,
              needsReview: row.needs_review !== 0,
            });
          }
        }
        db.run("UPDATE tags SET deleted_at = ?, updated_at = ? WHERE id = ?", [at, at, from.id]);
        db.run(
          `UPDATE document_tags SET deleted_at = ?, updated_at = ?
           WHERE tag_id = ? AND deleted_at IS NULL`,
          [at, at, from.id],
        );
        return [...new Set(changed)];
      });
    },

    /**
     * Sets a Document's automatic Tags to those in `decisions`: automatic
     * links not decided on are taken off, decided Tags not on it are added,
     * and those kept take the new confidence and review mark. Tags the User
     * added or removed on the Document, and deleted Tags, are left alone.
     * Returns whether the Document's Tags changed. Run it in the caller's transaction.
     */
    applyAutomatic(documentId: string, decisions: readonly TagDecision[]): boolean {
      return db.transaction(() => {
        const owned = new Set(
          db
            .all<{ tag_id: string }>(
              "SELECT tag_id FROM document_tags WHERE document_id = ? AND source = 'user'",
              [documentId],
            )
            .map((row) => row.tag_id),
        );
        const live = new Set(list().map((tag) => tag.id));
        const wanted = new Map(
          decisions
            .filter((decision) => live.has(decision.tagId) && !owned.has(decision.tagId))
            .map((decision) => [decision.tagId, decision]),
        );
        const links = db.all<LinkRow>(
          `SELECT id, tag_id, source, confidence, needs_review FROM document_tags
           WHERE document_id = ? AND deleted_at IS NULL AND source = 'automatic'`,
          [documentId],
        );
        const at = now();
        let changed = false;
        for (const link of links) {
          const decision = wanted.get(link.tag_id);
          wanted.delete(link.tag_id);
          if (!decision) {
            db.run("UPDATE document_tags SET deleted_at = ?, updated_at = ? WHERE id = ?", [
              at,
              at,
              link.id,
            ]);
            changed = true;
          } else if (
            link.confidence !== decision.confidence ||
            (link.needs_review !== 0) !== decision.needsReview
          ) {
            // Still applies, decided afresh: e.g. Jev's probability moved on a re-tag.
            db.run(
              `UPDATE document_tags SET confidence = ?, needs_review = ?, updated_at = ?
               WHERE id = ?`,
              [decision.confidence, decision.needsReview ? 1 : 0, at, link.id],
            );
            changed = true;
          }
        }
        for (const [tagId, decision] of wanted) {
          insertLink(documentId, tagId, "automatic", at, null, decision);
          changed = true;
        }
        return changed;
      });
    },
  };
}
