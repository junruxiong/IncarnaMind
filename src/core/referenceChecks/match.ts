/**
 * Whether a reference as written (an entry of a reference list) is the work a
 * record describes, and how they differ (#711). Pure: the records come from
 * Crossref (./crossref).
 *
 * A reference is the work when most of the record's title is in it, word for
 * word once case, accents and punctuation are set aside. Then its year must
 * be one of the record's (issued, in print or online, which can differ by a
 * year), and its first author's family name must be in it. A title that is
 * only partly there is a work by another title: by DOI, that is a mismatch
 * (the DOI leads to a record, but not to this work); found by searching, it
 * is no match at all.
 */
import { normalizeDoi } from "../../shared/references";
import type { ReferenceDifference, ReferenceFields, ReferenceName } from "../api";

/** Of the record's title words, the share the reference must hold for it to be that work. */
export const SAME_TITLE = 0.75;
/** Below this share, a DOI's record is another work's. */
export const OTHER_TITLE = 0.5;

/** A reference as written, with what can be read from it without a service. */
export interface CitedReference {
  text: string;
  doi: string | null;
  /** The year it names: the first four-digit year from 1500 to 2099. */
  year: number | null;
}

/** A record's fields as a check reads them, with every year it gives. */
export interface WorkRecord {
  fields: Partial<ReferenceFields>;
  /** The years it was issued, printed and put online. */
  years: number[];
}

const DOI_IN_TEXT = /(?:doi\.org\/|doi\s*[:：]?\s*)?(10\.\d{4,9}\/[^\s"<>]+)/i;
const YEAR = /(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)/;

/** Folds case and accents, so "Müller" is "muller". */
const folded = (text: string) => text.normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase();

/** Common English words, which say nothing about which work a title is. */
const STOP_WORDS = new Set(
  "a an and are as at be by for from in into is it its of on or the their to with".split(" "),
);

/** A text's words: letters and digits, folded; a CJK run as its characters, two by two. */
export function titleWords(text: string): string[] {
  const words: string[] = [];
  for (const [run] of folded(text).matchAll(/[\p{L}\p{N}]+/gu)) {
    for (const [part] of run.matchAll(/\p{Script=Han}+|[^\p{Script=Han}]+/gu)) {
      if (/^\p{Script=Han}/u.test(part)) {
        const chars = [...part];
        if (chars.length === 1) words.push(part);
        for (let at = 0; at + 1 < chars.length; at++) words.push(`${chars[at]}${chars[at + 1]}`);
      } else if (!STOP_WORDS.has(part)) {
        words.push(part);
      }
    }
  }
  return words;
}

/** What can be read from a reference as written: its DOI and year. */
export function citedReference(text: string): CitedReference {
  const doi = DOI_IN_TEXT.exec(text)?.[1];
  const year = YEAR.exec(text.replace(DOI_IN_TEXT, " "))?.[1];
  return {
    text: text.replace(/\s+/g, " ").trim(),
    doi: doi ? normalizeDoi(doi) : null,
    year: year ? Number(year) : null,
  };
}

/** The share of a title's words that a reference holds, from 0 to 1. */
export function titleCoverage(reference: string, title: string): number {
  const wanted = new Set(titleWords(title));
  if (wanted.size === 0) return 0;
  const held = new Set(titleWords(reference));
  let shared = 0;
  for (const word of wanted) if (held.has(word)) shared++;
  return shared / wanted.size;
}

/** A name's family name, or a literal name whole. */
const familyOf = (name: ReferenceName) => ("family" in name ? name.family : name.literal);

/** The authors as a reader would list them: "Hochreiter, Schmidhuber". */
const authorList = (authors: readonly ReferenceName[]) =>
  authors.slice(0, 3).map(familyOf).join(", ") + (authors.length > 3 ? " et al." : "");

/**
 * How `cited` differs from the work `record` describes: its title (when
 * little of it is in the reference), its year, its first author.
 */
export function differences(cited: CitedReference, record: WorkRecord): ReferenceDifference[] {
  const found: ReferenceDifference[] = [];
  const title = record.fields.title ?? "";
  if (title && titleCoverage(cited.text, title) < SAME_TITLE) {
    found.push({ field: "title", found: title, cited: null });
  }
  if (cited.year !== null && record.years.length > 0 && !record.years.includes(cited.year)) {
    found.push({
      field: "year",
      found: String(Math.min(...record.years)),
      cited: String(cited.year),
    });
  }
  const first = record.fields.authors?.[0];
  if (first && !titleWords(cited.text).some((word) => titleWords(familyOf(first)).includes(word))) {
    found.push({ field: "authors", found: authorList(record.fields.authors ?? []), cited: null });
  }
  return found;
}
