/**
 * A Word file's look as a Mind imported from it keeps it (#746, `WordLook`):
 * the styles, in a map beside the Blocks (`MIND_WORD_LOOK_FIELD`), that the
 * Blocks' `wordLook` attribute and the text's `wordRun` mark name. Stored as
 * a few values, each whole: the styles don't change once read, and two
 * devices reading the same package write the same ones. Shared by the core,
 * the editor and the tests.
 */
import type * as Y from "yjs";
import { MIND_WORD_LOOK_FIELD, type WordLook } from "../core/api";

const lookMap = (doc: Y.Doc) => doc.getMap<unknown>(MIND_WORD_LOOK_FIELD);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const number = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

/** Writes the look. Inside a transaction. */
export function setWordLook(doc: Y.Doc, look: WordLook): void {
  const map = lookMap(doc);
  map.set("paragraphs", look.paragraphs);
  map.set("runs", look.runs);
  map.set("tables", look.tables);
  map.set("cells", look.cells);
  map.set("body", look.body);
  map.set("width", look.width);
  // Last: a look with its version is whole.
  map.set("version", look.version);
}

/** The Mind's Word look, or null for a Mind that has none (yet). */
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
