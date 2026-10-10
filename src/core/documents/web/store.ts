/** What is stored of saved web pages, one `web_pages` row per version (migration 37). */
import { randomUUID } from "node:crypto";
import type { WebPageInfo } from "../../api";
import type { Database } from "../../storage";

interface WebPageRow {
  content_hash: string;
  address: string;
  site_name: string;
  author: string | null;
  published: string | null;
  fetched_at: string;
}

export interface WebPageVersion {
  documentId: string;
  contentHash: string;
  address: string;
  siteName: string;
  author: string | null;
  published: string | null;
  fetchedAt: string;
}

/** A web page Document's address and metadata, as last fetched; null if it has none. */
export function webPageInfo(db: Database, documentId: string): WebPageInfo | null {
  const rows = db.all<WebPageRow>(
    `SELECT content_hash, address, site_name, author, published, fetched_at FROM web_pages
     WHERE document_id = ? AND deleted_at IS NULL ORDER BY fetched_at DESC, rowid DESC`,
    [documentId],
  );
  const latest = rows[0];
  if (!latest) return null;
  return {
    address: latest.address,
    siteName: latest.site_name,
    author: latest.author,
    published: latest.published,
    fetchedAt: latest.fetched_at,
    versions: rows.map((row) => ({ contentHash: row.content_hash, fetchedAt: row.fetched_at })),
  };
}

/** The live Document saved from this address, if there is one. */
export function documentAtAddress(db: Database, address: string): string | null {
  return (
    db.get<{ document_id: string }>(
      `SELECT w.document_id FROM web_pages w JOIN documents d ON d.id = w.document_id
       WHERE w.address = ? AND w.deleted_at IS NULL AND d.deleted_at IS NULL
       ORDER BY w.fetched_at DESC LIMIT 1`,
      [address],
    )?.document_id ?? null
  );
}

/** Records a fetch of a version: a new row for a new version, else the same one fetched again. */
export function recordVersion(db: Database, version: WebPageVersion, at: string): void {
  const updated = db.get<{ id: string }>(
    `UPDATE web_pages SET address = ?, site_name = ?, author = ?, published = ?, fetched_at = ?,
       updated_at = ?
     WHERE document_id = ? AND content_hash = ? AND deleted_at IS NULL RETURNING id`,
    [
      version.address,
      version.siteName,
      version.author,
      version.published,
      version.fetchedAt,
      at,
      version.documentId,
      version.contentHash,
    ],
  );
  if (updated) return;
  db.run(
    `INSERT INTO web_pages (id, document_id, content_hash, address, site_name, author, published,
       fetched_at, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      randomUUID(),
      version.documentId,
      version.contentHash,
      version.address,
      version.siteName,
      version.author,
      version.published,
      version.fetchedAt,
      at,
      at,
    ],
  );
}
