/**
 * Found sources in the database (migration 49): where a Document, or a
 * reference record with no Document, that Find sources added came from;
 * and whether a found work is in the Library already, by what was added
 * from it or by its DOI on any reference record (a file the User added
 * themselves counts).
 */
import { randomUUID } from "node:crypto";
import type { DocumentFoundSource, SourceVersion } from "../api";
import type { Database } from "../storage";
import { licenceOf } from "./licences";
import type { WorkIds } from "./types";

interface FoundRow {
  provider: string;
  provider_name: string;
  address: string | null;
  licence: string | null;
  version: string | null;
  version_date: string | null;
  fetched_at: string | null;
}

const VERSIONS: readonly SourceVersion[] = ["published", "accepted", "submitted"];

/** Where a live Document Find sources added came from; null for any other. */
export function foundSourceOf(db: Database, documentId: string): DocumentFoundSource | null {
  const row = db.get<FoundRow>(
    `SELECT provider, provider_name, address, licence, version, version_date, fetched_at
     FROM found_sources
     WHERE document_id = ? AND deleted_at IS NULL`,
    [documentId],
  );
  if (!row || row.fetched_at === null) return null;
  return {
    provider: { id: row.provider, name: row.provider_name },
    address: row.address ?? "",
    licence: licenceOf(row.licence),
    version: VERSIONS.find((version) => version === row.version) ?? null,
    versionDate: row.version_date,
    fetchedAt: row.fetched_at,
  };
}

export interface FoundSourceRow {
  documentId: string | null;
  recordId: string | null;
  provider: { id: string; name: string };
  workKey: string;
  address: string | null;
  licence: string | null;
  version: SourceVersion | null;
  versionDate: string | null;
  fetchedAt: string | null;
}

export function recordFoundSource(db: Database, row: FoundSourceRow, at: string): void {
  db.run(
    `INSERT INTO found_sources (id, document_id, reference_record_id, provider, provider_name,
       work_key, address, licence, version, version_date, fetched_at, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      randomUUID(),
      row.documentId,
      row.recordId,
      row.provider.id,
      row.provider.name,
      row.workKey,
      row.address,
      row.licence,
      row.version,
      row.versionDate,
      row.fetchedAt,
      at,
      at,
    ],
  );
}

/** The keys a work could have been added under: one per identifier. */
const keysOf = (ids: WorkIds): string[] => [
  ...(ids.doi ? [`doi:${ids.doi}`] : []),
  ...(ids.arxiv ? [`arxiv:${ids.arxiv}`] : []),
  ...(ids.pmid ? [`pmid:${ids.pmid}`] : []),
  ...(ids.openalex ? [`openalex:${ids.openalex}`] : []),
];

/** The DOIs a work's records could carry: its own, and arXiv's for its preprint. */
const doisOf = (ids: WorkIds): string[] => [
  ...(ids.doi ? [ids.doi.toLowerCase()] : []),
  ...(ids.arxiv ? [`10.48550/arxiv.${ids.arxiv.toLowerCase()}`] : []),
];

const LIVE_DOCUMENT =
  "EXISTS (SELECT 1 FROM documents d WHERE d.id = r.document_id AND d.deleted_at IS NULL)";

/**
 * Whether a work is in the Library: a live Document of it (its record's DOI,
 * or added by Find sources), else a reference record of it with no
 * Document. Null if neither.
 */
export function inLibrary(
  db: Database,
  ids: WorkIds,
): { documentId: string | null; recordId: string } | null {
  const keys = keysOf(ids);
  const dois = doisOf(ids);
  if (keys.length === 0 && dois.length === 0) return null;
  const rows = db.all<{ id: string; document_id: string | null }>(
    `SELECT r.id, r.document_id FROM reference_records r
     WHERE r.deleted_at IS NULL
       AND (r.document_id IS NULL OR ${LIVE_DOCUMENT})
       AND (lower(json_extract(r.fields, '$.doi.value')) IN (${dois.map(() => "?").join(",") || "NULL"})
         OR r.id IN (SELECT f.reference_record_id FROM found_sources f
                     WHERE f.deleted_at IS NULL AND f.work_key IN (${keys.map(() => "?").join(",") || "NULL"}))
         OR r.document_id IN (SELECT f.document_id FROM found_sources f
                     WHERE f.deleted_at IS NULL AND f.work_key IN (${keys.map(() => "?").join(",") || "NULL"})))
     ORDER BY r.document_id IS NULL, r.created_at`,
    [...dois, ...keys, ...keys],
  );
  const row = rows[0];
  return row ? { documentId: row.document_id, recordId: row.id } : null;
}

/**
 * The copies of Documents Find sources added that have since been deleted
 * permanently, not yet removed: their files can go, unless a Document that
 * isn't purged is at the same path (a later copy given the freed name).
 */
export function purgedFoundFiles(db: Database): { id: string; path: string }[] {
  return db.all<{ id: string; path: string }>(
    `SELECT f.id, d.path FROM documents d JOIN found_sources f ON f.document_id = d.id
     WHERE d.purged_at IS NOT NULL AND f.deleted_at IS NULL
       AND NOT EXISTS (SELECT 1 FROM documents o WHERE o.path = d.path AND o.purged_at IS NULL)`,
  );
}

/** A copy removed: its row goes too, so it isn't removed again. */
export function forgetFoundFile(db: Database, id: string, at: string): void {
  db.run("UPDATE found_sources SET deleted_at = ?, updated_at = ? WHERE id = ?", [at, at, id]);
}
