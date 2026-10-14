/**
 * Each style table is stored as one whole value rather than nested Y types:
 * styles never change once read, and two devices reading the same package
 * write the same values.
 */
import type * as Y from "yjs";
import { MIND_WORD_LOOK_FIELD, type WordLook } from "../core/api";

const lookMap = (doc: Y.Doc) => doc.getMap<unknown>(MIND_WORD_LOOK_FIELD);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const number = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

export function setWordLook(doc: Y.Doc, look: WordLook): void {
  const map = lookMap(doc);
  map.set("paragraphs", look.paragraphs);
  map.set("runs", look.runs);
  map.set("tables", look.tables);
  map.set("cells", look.cells);
  map.set("body", look.body);
  map.set("width", look.width);
  // Last, so a reader that sees a version sees the whole look.
  map.set("version", look.version);
}

export function wordLookOf(doc: Y.Doc): WordLook | null {
  const map = lookMap(doc);
  const version = number(map.get("version"));
  if (version === null) return null;
  const styles = <T>(key: string): Record<string, T> => {
    const value = map.get(key);
    return isRecord(value) ? (value as Record<string, T>) : {};
  };
  return {
    version,
    body: number(map.get("body")),
    width: number(map.get("width")),
    paragraphs: styles("paragraphs"),
    runs: styles("runs"),
    tables: styles("tables"),
    cells: styles("cells"),
  };
}
