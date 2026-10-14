/**
 * One `web_pages` row per version of a web page (migration 47), and one
 * `web_page_pictures` row per picture each version shows (migration 89).
 */
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
  pictures_left_out: number;
}

export interface WebPageVersion {
  documentId: string;
  contentHash: string;
  address: string;
  siteName: string;
  author: string | null;
  published: string | null;
  fetchedAt: string;
  picturesLeftOut: number;
}

/** A picture a version shows, numbered from 1 as the page shows them; its file is named by `pictureHash`. */
export interface WebPagePicture {
  number: number;
  pictureHash: string;
  mediaType: string;
  width: number;
  height: number;
  bytes: number;
  address: string;
}

/** As last fetched; null for a Document that isn't a web page. */
export function webPageInfo(db: Database, documentId: string): WebPageInfo | null {
  const rows = db.all<WebPageRow>(
    `SELECT content_hash, address, site_name, author, published, fetched_at, pictures_left_out
     FROM web_pages
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
    picturesLeftOut: latest.pictures_left_out,
  };
}

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

/** A new row for a new version; the same version fetched again updates its row. */
export function recordVersion(db: Database, version: WebPageVersion, at: string): void {
  const updated = db.get<{ id: string }>(
    `UPDATE web_pages SET address = ?, site_name = ?, author = ?, published = ?, fetched_at = ?,
       pictures_left_out = ?, updated_at = ?
     WHERE document_id = ? AND content_hash = ? AND deleted_at IS NULL RETURNING id`,
    [
      version.address,
      version.siteName,
      version.author,
      version.published,
      version.fetchedAt,
      version.picturesLeftOut,
      at,
      version.documentId,
      version.contentHash,
    ],
  );
  if (updated) return;
  db.run(
    `INSERT INTO web_pages (id, document_id, content_hash, address, site_name, author, published,
       fetched_at, pictures_left_out, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      randomUUID(),
      version.documentId,
      version.contentHash,
      version.address,
      version.siteName,
      version.author,
      version.published,
      version.fetchedAt,
      version.picturesLeftOut,
      at,
      at,
    ],
  );
}

/** Records the pictures a version shows; a version fetched again shows the same ones. */
export function recordPictures(
  db: Database,
  documentId: string,
  contentHash: string,
  pictures: readonly WebPagePicture[],
  at: string,
): void {
  const known = new Set(
    db
      .all<{ number: number }>(
        `SELECT number FROM web_page_pictures
         WHERE document_id = ? AND content_hash = ? AND deleted_at IS NULL`,
        [documentId, contentHash],
      )
      .map((row) => row.number),
  );
  for (const picture of pictures) {
    if (known.has(picture.number)) continue;
    db.run(
      `INSERT INTO web_page_pictures (id, document_id, content_hash, number, picture_hash,
         media_type, width, height, bytes, address, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        randomUUID(),
        documentId,
        contentHash,
        picture.number,
        picture.pictureHash,
        picture.mediaType,
        picture.width,
        picture.height,
        picture.bytes,
        picture.address,
        at,
        at,
      ],
    );
  }
}

/** Whether any version of the Document shows the picture named by this hash. */
export function showsPicture(db: Database, documentId: string, pictureHash: string): boolean {
  return (
    db.get<{ found: number }>(
      `SELECT 1 AS found FROM web_page_pictures
       WHERE document_id = ? AND picture_hash = ? AND deleted_at IS NULL LIMIT 1`,
      [documentId, pictureHash],
    ) !== undefined
  );
}

/** The pictures some version of a web page shows, live or in Recently deleted: their files stay. */
export function keptPictureHashes(db: Database): Set<string> {
  return new Set(
    db
      .all<{ picture_hash: string }>("SELECT DISTINCT picture_hash FROM web_page_pictures")
      .map((row) => row.picture_hash),
  );
}
