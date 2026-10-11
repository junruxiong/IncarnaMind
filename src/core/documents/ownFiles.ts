/**
 * The files IncarnaMind writes (#80): exports and source packs. Each one's
 * content fingerprint is recorded before it is written (migration 50), so
 * when the file, or a copy of it, becomes a Document (saved into a Linked
 * folder, or added by the User), that Document is known as one IncarnaMind
 * wrote (`Document.ownFile`) and checks leave it out of their evidence
 * (./evidence). The schema marks the Document, whichever comes first.
 *
 * Only the fingerprint is kept: no name or path, which belong to one
 * computer. A file the User edits keeps its mark once it has one, since its
 * text still came from a Mind; one that was never seen unedited isn't marked.
 */
import { randomUUID } from "node:crypto";
import type { OwnFileKind } from "../api";
import type { Database } from "../storage";
import { hashBytes } from "./files";

export interface OwnFiles {
  /**
   * Records that IncarnaMind is writing a file with these bytes, and returns
   * the live Documents that this marks: those already indexed with them,
   * whose change the caller announces. Nothing is recorded for an empty
   * file: every empty file has its fingerprint.
   */
  record(data: Uint8Array, kind: OwnFileKind): string[];
}

export function createOwnFiles(db: Database, now: () => string): OwnFiles {
  return {
    record(data, kind) {
      if (data.byteLength === 0) return [];
      const contentHash = hashBytes(data);
      return db.transaction(() => {
        // The same bytes written again are the same file: one row is enough.
        const known = db.get(
          "SELECT 1 FROM own_files WHERE content_hash = ? AND kind = ? AND deleted_at IS NULL",
          [contentHash, kind],
        );
        if (known) return [];
        const marked = db
          .all<{ id: string }>(
            `SELECT id FROM documents
             WHERE content_hash = ? AND own_file IS NULL AND deleted_at IS NULL`,
            [contentHash],
          )
          .map((row) => row.id);
        const at = now();
        db.run(
          `INSERT INTO own_files (id, content_hash, kind, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)`,
          [randomUUID(), contentHash, kind, at, at],
        );
        return marked;
      });
    },
  };
}
