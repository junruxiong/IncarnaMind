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

/**
 * The excerpt a classifier is given: the name, the kind and page count, a
 * deck's or a workbook's outline (see `documentOutline`), and the beginning
 * of the text.
 */
export function organizeExcerpt(source: OrganizeSource): DocumentExcerpt {
  const outline = documentOutline(source);
  return {
    name: source.name,
    kind: source.kind,
    pageCount: source.pageCount,
    ...(outline ? { outline } : {}),
    text: excerptFromPassages(source.passages.slice(0, ORGANIZE_PASSAGES)),
  };
}

/** At most this many entries in an outline, each cut to this many characters. */
const OUTLINE_ENTRIES = 16;
const OUTLINE_ENTRY_CHARACTERS = 60;

const clip = (text: string) => {
  const line = text.replace(/\s+/g, " ").trim();
  return line.length > OUTLINE_ENTRY_CHARACTERS
    ? `${line.slice(0, OUTLINE_ENTRY_CHARACTERS - 1).trimEnd()}…`
    : line;
};

const firstLine = (text: string) => text.split("\n").find((line) => line.trim()) ?? "";

/** "a; b; c", and "(+N more)" past the limit. */
function list(entries: readonly string[]): string {
  const shown = entries.slice(0, OUTLINE_ENTRIES).join("; ");
  const more = entries.length - OUTLINE_ENTRIES;
  return more > 0 ? `${shown} (+${more} more)` : shown;
}

/**
 * The shape of a deck or a workbook, which its text doesn't show: a deck's
 * slide titles, a workbook's sheet names. It tells a deck or a spreadsheet
 * for what it is, and shows what lies past the excerpt. Null for other
 * formats: a Word or Markdown file's headings are in its text already, and
 * listing them again made Tev1 4B less accurate on the tuning half of the
 * Organize set (eval/organize).
 */
export function documentOutline(source: Pick<OrganizeSource, "kind" | "units">): string | null {
  const { kind, units } = source;
  if (kind === "pptx") {
    const slides = units.filter((unit) => unit.kind === "slide");
    if (slides.length === 0) return null;
    const titles = slides.map(
      (unit, index) => `${index + 1}. ${clip(unit.label?.title || firstLine(unit.text)) || "–"}`,
    );
    return `${slides.length} ${slides.length === 1 ? "slide" : "slides"}: ${list(titles)}`;
  }
  if (kind === "xlsx" || kind === "csv") {
    const sheets = [
      ...new Set(units.flatMap((unit) => (unit.label?.sheet ? [clip(unit.label.sheet)] : []))),
    ];
    if (sheets.length === 0) return null;
    return `${sheets.length} ${sheets.length === 1 ? "sheet" : "sheets"}: ${list(sheets)}`;
  }
  return null;
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
