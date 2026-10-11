import type { ReferenceFields } from "../../api";

/** One reference as an export file gave it, in the record's own fields. */
export interface ReadReference {
  /** What names it in a problem: its cite key, its id, or where it is in the file. */
  label: string;
  fields: Partial<ReferenceFields>;
  /** Attached files the export points to, as written (a path, absolute or relative to the export). */
  files: string[];
}

/** An entry that couldn't be read; the rest of the file still is. */
export interface ReadProblem {
  /** The entry's cite key or id, else its position in the file. */
  label: string;
  /** What is wrong, in plain English; the window says it in the User's language. */
  reason: "broken" | "no-title" | "not-a-reference";
}

export interface ReadFile {
  format: "csl-json" | "bibtex" | "ris";
  references: ReadReference[];
  problems: ReadProblem[];
}
