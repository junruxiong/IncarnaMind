/**
 * Importing references (#287) in the database: the rules for which source
 * wins a field, and what Undo puts back. A record can have no Document yet
 * (an imported reference not matched to one); it gets one when the User adds
 * its PDF, or when a later import matches it.
 *
 * Who wins a field: the User's edit, then Look up's, then an import's, then
 * the file's. An import fills a field nobody has given, and replaces one the
 * file or an earlier import gave, so importing again updates; it never takes
 * a field from the User or from Look up, and a field the User cleared stays clear.
 */
import { isEmptyValue } from "../../shared/references";
import type { ReferenceField, ReferenceFields, ReferenceRecord, ReferenceSource } from "../api";
import type { Database } from "../storage";
import {
  type FieldValue,
  parseStored,
  type RecordRow,
  type StoredFields,
  same,
  save,
  toRecord,
} from "./store";

const RANK: Readonly<Record<ReferenceSource, number>> = { file: 1, import: 2, lookup: 3, user: 4 };

/** What Undo does for one change an import made. */
export type UndoStep =
  | {
      kind: "fields";
      recordId: string;
      created: boolean;
      /** The fields it changed: as they were (absent if the record had no value), and as it wrote them. */
      before: StoredFields;
      wrote: StoredFields;
    }
  | { kind: "link"; recordId: string; documentId: string }
  | { kind: "undelete"; recordId: string };

/** Where an import writes: a Document's record, a record with no Document, or a new one with none. */
export type ImportTarget = { documentId: string } | { recordId: string } | { unlinked: true };

const SELECT = "SELECT id, fields, looked_up_at, updated_at FROM reference_records";

function rowOf(db: Database, target: ImportTarget): RecordRow | undefined {
  if ("documentId" in target) {
    return db.get<RecordRow>(`${SELECT} WHERE document_id = ? AND deleted_at IS NULL`, [
      target.documentId,
    ]);
  }
  if ("recordId" in target) {
    return db.get<RecordRow>(`${SELECT} WHERE id = ? AND deleted_at IS NULL`, [target.recordId]);
  }
  return undefined;
}

/** The records with no Document, oldest first. */
export function unlinkedRecords(db: Database): ReferenceRecord[] {
  return db
    .all<RecordRow>(
      `${SELECT} WHERE document_id IS NULL AND deleted_at IS NULL ORDER BY created_at, id`,
    )
    .map(toRecord);
}

/**
 * Puts `incoming` fields into `stored` where their source wins; returns what
 * it changed as `before` and `wrote`. A value already there is left as it is,
 * whoever gave it; an import replaces an earlier import's value.
 */
function merge(
  stored: StoredFields,
  incoming: StoredFields,
): { before: StoredFields; wrote: StoredFields } {
  const before: StoredFields = {};
  const wrote: StoredFields = {};
  for (const [field, entry] of Object.entries(incoming) as [
    ReferenceField,
    NonNullable<StoredFields[ReferenceField]>,
  ][]) {
    if (isEmptyValue(field, entry.value)) continue;
    const current = stored[field];
    const wins =
      !current ||
      (current.source !== "user" && isEmptyValue(field, current.value)) ||
      RANK[current.source] < RANK[entry.source] ||
      (RANK[current.source] === RANK[entry.source] && entry.source === "import");
    if (!wins || (current && same(current.value, entry.value))) continue;
    if (current) before[field] = current;
    stored[field] = entry;
    wrote[field] = entry;
  }
  return { before, wrote };
}

const fromImport = (fields: Partial<ReferenceFields>): StoredFields => {
  const stored: StoredFields = {};
  for (const [field, value] of Object.entries(fields) as [ReferenceField, FieldValue][]) {
    stored[field] = { value, source: "import" };
  }
  return stored;
};

/** What `writeImported` and `linkImported` did, to report and to undo. */
export interface ImportWrite {
  /** The record that holds the reference now. */
  recordId: string;
  /** Whether anything was made or changed. */
  changed: boolean;
  steps: UndoStep[];
}

/** Writes an imported reference into the record at `target`, making the record if there is none. */
export function writeImported(
  db: Database,
  target: ImportTarget,
  fields: Partial<ReferenceFields>,
  at: string,
): ImportWrite {
  const row = rowOf(db, target);
  const stored = row ? parseStored(row.fields) : {};
  const { before, wrote } = merge(stored, fromImport(fields));
  if (row && Object.keys(wrote).length === 0)
    return { recordId: row.id, changed: false, steps: [] };
  const recordId = save(
    db,
    row?.id ?? null,
    "documentId" in target ? target.documentId : null,
    stored,
    at,
  );
  return {
    recordId,
    changed: true,
    steps: [{ kind: "fields", recordId, created: !row, before, wrote }],
  };
}

/**
 * Gives a record with no Document its Document: the record becomes the
 * Document's, or, if the Document has a record already, its fields are
 * merged into that one (the stronger source wins each) and the loose record
 * is removed.
 */
export function linkImported(
  db: Database,
  recordId: string,
  documentId: string,
  at: string,
): ImportWrite | null {
  const loose = rowOf(db, { recordId });
  if (!loose) return null;
  const own = rowOf(db, { documentId });
  if (!own) {
    db.run(`UPDATE reference_records SET document_id = ?, updated_at = ? WHERE id = ?`, [
      documentId,
      at,
      recordId,
    ]);
    return { recordId, changed: true, steps: [{ kind: "link", recordId, documentId }] };
  }
  const stored = parseStored(own.fields);
  const { before, wrote } = merge(stored, parseStored(loose.fields));
  const steps: UndoStep[] = [];
  if (Object.keys(wrote).length > 0) {
    save(db, own.id, documentId, stored, at);
    steps.push({ kind: "fields", recordId: own.id, created: false, before, wrote });
  }
  db.run(`UPDATE reference_records SET deleted_at = ?, updated_at = ? WHERE id = ?`, [
    at,
    at,
    recordId,
  ]);
  steps.push({ kind: "undelete", recordId });
  return { recordId: own.id, changed: true, steps };
}

/**
 * Takes an import back, last change first. A field the User edited since
 * stays as it is; a record the import made stays if the User has made it
 * their own.
 */
export function undoImported(db: Database, steps: readonly UndoStep[], at: string): void {
  db.transaction(() => {
    for (const step of [...steps].reverse()) {
      if (step.kind === "undelete") {
        db.run(`UPDATE reference_records SET deleted_at = NULL, updated_at = ? WHERE id = ?`, [
          at,
          step.recordId,
        ]);
      } else if (step.kind === "link") {
        db.run(
          `UPDATE reference_records SET document_id = NULL, updated_at = ?
           WHERE id = ? AND document_id = ?`,
          [at, step.recordId, step.documentId],
        );
      } else {
        const row = db.get<RecordRow>(`${SELECT} WHERE id = ? AND deleted_at IS NULL`, [
          step.recordId,
        ]);
        if (!row) continue;
        const stored = parseStored(row.fields);
        for (const field of Object.keys(step.wrote) as ReferenceField[]) {
          const now = stored[field];
          const wrote = step.wrote[field];
          if (!now || !wrote || now.source !== wrote.source || !same(now.value, wrote.value))
            continue;
          const previous = step.before[field];
          if (previous) stored[field] = previous;
          else delete stored[field];
        }
        const theirs = Object.values(stored).some((entry) => entry.source !== "import");
        if (step.created && !theirs) {
          db.run(`UPDATE reference_records SET deleted_at = ?, updated_at = ? WHERE id = ?`, [
            at,
            at,
            row.id,
          ]);
        } else {
          save(db, row.id, null, stored, at);
        }
      }
    }
  });
}
