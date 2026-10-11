import type { Database } from "../../../src/core/storage";
import type { UnitKind, UnitLabel } from "../../../src/shared/units";
import type { StoredText } from "./questions";

/**
 * What the app stored for each Document, read back (and kept) to check each
 * Question against: undefined for a Document that isn't ready.
 */
export function storedTextReader(
  db: Database,
  ids: ReadonlyMap<string, string>,
  readyKeys: ReadonlySet<string>,
): (key: string) => StoredText | undefined {
  const cache = new Map<string, StoredText>();
  return (key) => {
    if (!readyKeys.has(key)) return undefined;
    let text = cache.get(key);
    if (!text) {
      const id = ids.get(key) ?? "";
      text = {
        units: db
          .all<{ page: number | null; kind: UnitKind; label: string | null; text: string }>(
            "SELECT page, kind, label, text FROM document_pages WHERE document_id = ? AND deleted_at IS NULL ORDER BY page",
            [id],
          )
          .map((unit) => ({
            ...unit,
            label: unit.label ? (JSON.parse(unit.label) as UnitLabel) : null,
          })),
        passages: db.all<{ pageFrom: number | null; pageTo: number | null; text: string }>(
          "SELECT page_from AS pageFrom, page_to AS pageTo, text FROM passages WHERE document_id = ? AND deleted_at IS NULL",
          [id],
        ),
      };
      cache.set(key, text);
    }
    return text;
  };
}
