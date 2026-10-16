/**
 * Reference records in the database (#286, migration 42): one row per
 * Document, whose `fields` hold each field's value and where it came from.
 * The rules for who may change a field live here:
 * - reading the file replaces only what an earlier reading gave ("file");
 * - the User's edits ("user") are kept through every later reading;
 * - Look up ("lookup") fills only empty fields, and the type unless the User chose it;
 * - an import ("import", #287) fills empty fields and replaces what the file or an
 *   earlier import gave, never what the User edited or Look up found.
 */
import { randomUUID } from "node:crypto";
import { emptyReferenceFields, fieldsToCheck, isEmptyValue } from "../../shared/references";
import {
  type ReferenceField,
  type ReferenceFields,
  type ReferenceName,
  type ReferenceRecord,
  type ReferenceSource,
  referenceTypes,
} from "../api";
import { isRecord } from "../errors";
import type { Database } from "../storage";

export type FieldValue = ReferenceFields[ReferenceField];

/** A field as stored: its value, and where it came from. */
export interface StoredField {
  value: FieldValue;
  source: ReferenceSource;
}

export type StoredFields = Partial<Record<ReferenceField, StoredField>>;

export interface RecordRow {
  id: string;
  fields: string;
  looked_up_at: string | null;
  updated_at: string;
}

const SOURCES: readonly ReferenceSource[] = ["file", "lookup", "user", "import"];

export const isReferenceSource = (value: unknown): value is ReferenceSource =>
  SOURCES.some((source) => source === value);
const FIELDS = new Set<string>(Object.keys(emptyReferenceFields()));

export const isReferenceField = (value: unknown): value is ReferenceField =>
  typeof value === "string" && FIELDS.has(value);

const isName = (value: unknown): value is ReferenceName =>
  isRecord(value) &&
  (typeof value.literal === "string" ||
    (typeof value.family === "string" && typeof value.given === "string"));

/** Whether `value` is a value `field` can hold. */
export function isFieldValue(field: ReferenceField, value: unknown): value is FieldValue {
  if (field === "type") return referenceTypes.some((type) => type === value);
  if (field === "authors") return Array.isArray(value) && value.every(isName);
  return typeof value === "string";
}

/** The stored fields, skipping any this version can't read (a newer one's). */
export function parseStored(json: string): StoredFields {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return {};
  }
  if (!isRecord(parsed)) return {};
  const stored: StoredFields = {};
  for (const [field, entry] of Object.entries(parsed)) {
    if (!isReferenceField(field) || !isRecord(entry)) continue;
    const { source, value } = entry;
    if (isReferenceSource(source) && isFieldValue(field, value)) stored[field] = { value, source };
  }
  return stored;
}

export function toRecord(row: RecordRow): ReferenceRecord {
  const stored = parseStored(row.fields);
  const fields = emptyReferenceFields();
  const sources: ReferenceRecord["sources"] = {};
  for (const [field, entry] of Object.entries(stored) as [ReferenceField, StoredField][]) {
    (fields as Record<ReferenceField, FieldValue>)[field] = entry.value;
    sources[field] = entry.source;
  }
  return { id: row.id, fields, sources, lookedUpAt: row.looked_up_at, updatedAt: row.updated_at };
}

const findRow = (db: Database, documentId: string) =>
  db.get<RecordRow>(
    `SELECT id, fields, looked_up_at, updated_at FROM reference_records
     WHERE document_id = ? AND deleted_at IS NULL`,
    [documentId],
  );

/** A Document's reference record, or null until its file has been read. */
export function referenceOf(db: Database, documentId: string): ReferenceRecord | null {
  const row = findRow(db, documentId);
  return row ? toRecord(row) : null;
}

export const same = (a: FieldValue | undefined, b: FieldValue | undefined) =>
  JSON.stringify(a) === JSON.stringify(b);

/**
 * Changes a Document's stored fields with `change`, which edits them in
 * place and says whether anything changed; creates the record first if it
 * has none. Returns whether it wrote.
 */
function change(
  db: Database,
  documentId: string,
  at: string,
  edit: (stored: StoredFields) => boolean,
  lookedUp = false,
): boolean {
  const row = findRow(db, documentId);
  const stored = row ? parseStored(row.fields) : {};
  if (!edit(stored) && row) return false;
  save(db, row?.id ?? null, documentId, stored, at, lookedUp);
  return true;
}

/** Writes a record's fields: updates the row, or inserts one (for a Document, or none yet). */
export function save(
  db: Database,
  id: string | null,
  documentId: string | null,
  stored: StoredFields,
  at: string,
  lookedUp = false,
): string {
  const json = JSON.stringify(stored);
  if (id) {
    db.run(
      `UPDATE reference_records SET fields = ?, looked_up_at = coalesce(?, looked_up_at),
         updated_at = ? WHERE id = ?`,
      [json, lookedUp ? at : null, at, id],
    );
    return id;
  }
  const created = randomUUID();
  db.run(
    `INSERT INTO reference_records (id, document_id, fields, looked_up_at, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?)`,
    [created, documentId, json, lookedUp ? at : null, at, at],
  );
  return created;
}

/**
 * What a reading of the file gave: it replaces what an earlier reading gave,
 * and fills fields nobody has, but never a field the User edited or Look up
 * filled. A field the file no longer gives goes, if the file gave it.
 */
export function writeFileReference(
  db: Database,
  documentId: string,
  read: Partial<ReferenceFields>,
  at: string,
): void {
  change(db, documentId, at, (stored) => {
    let changed = false;
    for (const field of FIELDS as Set<ReferenceField>) {
      const current = stored[field];
      if (current && current.source !== "file") continue;
      const value = read[field];
      const next = value === undefined || isEmptyValue(field, value) ? undefined : value;
      if (next === undefined && field === "type" && read.type !== undefined) {
        // Other, as the file says: kept as the file's, so Look up may still set it.
        if (!current || !same(current.value, read.type)) {
          stored.type = { value: read.type, source: "file" };
          changed = true;
        }
        continue;
      }
      if (next === undefined) {
        if (current) {
          delete stored[field];
          changed = true;
        }
      } else if (!current || !same(current.value, next)) {
        stored[field] = { value: next, source: "file" };
        changed = true;
      }
    }
    return changed;
  });
}

/**
 * The User's edits: each field given takes its value, and becomes theirs.
 * One given as it is changes nothing. Returns the fields that changed.
 */
export function editReference(
  db: Database,
  documentId: string,
  patch: Partial<ReferenceFields>,
  at: string,
): ReferenceField[] {
  const changed: ReferenceField[] = [];
  change(db, documentId, at, (stored) => {
    for (const [field, value] of Object.entries(patch) as [ReferenceField, FieldValue][]) {
      const current = stored[field];
      const now = current?.value ?? emptyReferenceFields()[field];
      if (same(now, value)) continue;
      stored[field] = { value, source: "user" };
      changed.push(field);
    }
    return changed.length > 0;
  });
  return changed;
}

/**
 * The User checked what reading the file gave (`fieldsToCheck`): those fields become theirs,
 * with their values as they are. Returns the fields that did.
 */
export function confirmReference(db: Database, documentId: string, at: string): ReferenceField[] {
  const record = referenceOf(db, documentId);
  if (!record) return [];
  const fields = fieldsToCheck(record);
  change(db, documentId, at, (stored) => {
    for (const field of fields) {
      const current = stored[field];
      if (current) stored[field] = { value: current.value, source: "user" };
    }
    return fields.length > 0;
  });
  return fields;
}

/**
 * What Look up found: it fills each field that is empty, and the type
 * unless the User chose it (a type read from the file is a guess from a DOI
 * or an arXiv id; the registry knows). Returns the fields it filled.
 */
export function fillReference(
  db: Database,
  documentId: string,
  found: Partial<ReferenceFields>,
  at: string,
): ReferenceField[] {
  const filled: ReferenceField[] = [];
  change(
    db,
    documentId,
    at,
    (stored) => {
      for (const [field, value] of Object.entries(found) as [ReferenceField, FieldValue][]) {
        if (isEmptyValue(field, value)) continue;
        const current = stored[field];
        // The registry agrees with the type the file suggested: it is no longer a guess to check.
        if (field === "type" && current?.source === "file" && same(current.value, value)) {
          stored.type = { value, source: "lookup" };
          continue;
        }
        const open =
          field === "type"
            ? current?.source !== "user" && !same(current?.value, value)
            : !current || isEmptyValue(field, current.value);
        if (!open) continue;
        stored[field] = { value, source: "lookup" };
        filled.push(field);
      }
      return true; // When it was looked up is recorded, even if nothing was new.
    },
    true,
  );
  return filled;
}

/**
 * A record with no Document (#296): a found work whose full text isn't open,
 * its fields as a lookup gives them ("lookup"), so a file added for it later
 * fills only what they leave empty. Returns its id.
 */
export function writeUnlinkedReference(
  db: Database,
  found: Partial<ReferenceFields>,
  at: string,
): string {
  const stored: StoredFields = {};
  for (const [field, value] of Object.entries(found) as [ReferenceField, FieldValue][]) {
    if (isReferenceField(field) && !isEmptyValue(field, value)) {
      stored[field] = { value, source: "lookup" };
    }
  }
  const id = randomUUID();
  db.run(
    `INSERT INTO reference_records (id, document_id, fields, looked_up_at, created_at, updated_at)
     VALUES (?, NULL, ?, ?, ?, ?)`,
    [id, JSON.stringify(stored), at, at, at],
  );
  return id;
}

/**
 * Gives a record with no Document its Document, the file just added for
 * it, before that file is read: so reading it fills only the empty fields.
 * False when the Document has a record of its own already.
 */
export function linkReference(
  db: Database,
  recordId: string,
  documentId: string,
  at: string,
): boolean {
  if (findRow(db, documentId)) return false;
  db.run(
    `UPDATE reference_records SET document_id = ?, updated_at = ?
     WHERE id = ? AND document_id IS NULL AND deleted_at IS NULL`,
    [documentId, at, recordId],
  );
  return findRow(db, documentId) !== undefined;
}

/** Puts fields back as they are in `before`, each with its value and source: Undo. */
export function restoreReference(
  db: Database,
  documentId: string,
  before: ReferenceRecord,
  fields: readonly ReferenceField[],
  at: string,
): void {
  change(db, documentId, at, (stored) => {
    let changed = false;
    for (const field of fields) {
      const source = before.sources[field];
      const next = source ? { value: before.fields[field], source } : undefined;
      if (JSON.stringify(stored[field]) === JSON.stringify(next)) continue;
      if (next) stored[field] = next;
      else delete stored[field];
      changed = true;
    }
    return changed;
  });
}
