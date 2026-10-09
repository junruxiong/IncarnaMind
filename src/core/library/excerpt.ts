/**
 * What Organize reads of a Document (ADR-0012): its name and type, the
 * beginning of its text, and whether a PDF needs page images. The Library and
 * `npm run eval:organize` both build it here, from the Document's first
 * Passages and the Units of its indexed version, so the measurement reads
 * exactly what the app sends.
 */
import type { UnitKind, UnitLabel } from "../../shared/units";
import type { DocumentKind } from "../api";
import { type DocumentExcerpt, excerptFromPassages } from "../tags/classify";
import { pdfNeedsPageImages } from "./routing";

/** How many Passages, from the start, Organize reads. */
export const ORGANIZE_PASSAGES = 6;

/** How many of a PDF's first pages decide whether it needs page images, and how much of each. */
export const ROUTING_PAGES = 12;
export const ROUTING_PAGE_CHARACTERS = 4000;

/** A Unit of the indexed version, as stored (see src/shared/units.ts). */
export interface OrganizeUnit {
  page: number;
  kind: UnitKind;
  label: UnitLabel | null;
  text: string;
}

export interface OrganizeSource {
  /** The Document's name, as the Library shows it (the file name without its extension). */
  name: string;
  kind: DocumentKind;
  pageCount: number | null;
  /** The first Passages' text, in order (at most `ORGANIZE_PASSAGES` are read). */
  passages: readonly string[];
  /** The Units in order. Only the first `ROUTING_PAGES` need their text. */
  units: readonly OrganizeUnit[];
}

/** The excerpt a classifier is given. */
export function organizeExcerpt(source: OrganizeSource): DocumentExcerpt {
  return {
    name: source.name,
    kind: source.kind,
    pageCount: source.pageCount,
    text: excerptFromPassages(source.passages.slice(0, ORGANIZE_PASSAGES)),
  };
}

/** A PDF whose first pages hold too little text to classify from text alone (see ./routing). */
export function organizeNeedsPageImages(
  source: Pick<OrganizeSource, "kind" | "pageCount" | "units">,
): boolean {
  if (source.kind !== "pdf") return false;
  return pdfNeedsPageImages(
    source.units
      .filter((unit) => unit.page >= 1 && unit.page <= ROUTING_PAGES)
      .map((unit) => ({ page: unit.page, text: unit.text.slice(0, ROUTING_PAGE_CHARACTERS) })),
    source.pageCount,
  );
}
