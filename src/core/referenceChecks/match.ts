/**
 * A reference is the work a record describes when most of the record's title
 * is in it, word for word once case, accents and punctuation are set aside.
 * Then its year must be one of the record's (issued, in print or online), or
 * a year off when one of them is a preprint, whose year its proceedings'
 * often follows; and its first author's family name must be in it. A title
 * only partly there is another work: by DOI, a mismatch; by search, no match.
 */
import {
  ARXIV_IN_TEXT,
  arxivOf,
  DOI_IN_TEXT,
  doiOf,
  joinBrokenDoi,
  titleWords,
  yearOf,
} from "../../shared/referenceList";
import type { ReferenceDifference, ReferenceFields, ReferenceName } from "../api";

/** Of the record's title words, the share the reference must hold for it to be that work. */
export const SAME_TITLE = 0.75;
/** A title of fewer words than this ("Deep learning", "Introduction") must stand alone in the reference. */
const SHORT_TITLE = 3;

export interface CitedReference {
  text: string;
  doi: string | null;
  arxiv: string | null;
  year: number | null;
  /** It cites a preprint: an arXiv number, or "preprint" in so many words. */
  preprint: boolean;
}

export interface WorkRecord {
  fields: Partial<ReferenceFields>;
  /** The years it was issued, printed and put online. */
  years: number[];
  preprint?: boolean;
}

export function citedReference(text: string): CitedReference {
  const joined = joinBrokenDoi(text);
  const arxiv = arxivOf(joined);
  return {
    text: text.replace(/\s+/g, " ").trim(),
    doi: doiOf(joined),
    arxiv,
    // A DOI's or arXiv number's digits aren't its year: "arXiv:1706.03762".
    year: yearOf(joined.replace(DOI_IN_TEXT, " ").replace(ARXIV_IN_TEXT, " ")),
    preprint: arxiv !== null || /\bpreprint\b/i.test(text),
  };
}

const LIST_NUMBER = /^\s*(?:\[\d{1,3}\]|\d{1,3}\.)\s+/;
/** What an author list goes on with after an initial's dot: "Vaswani, N", "Shazeer", not a title. */
const AUTHORS_GO_ON =
  /^(?:\p{Lu}[\p{L}'’-]*(?:,\s*\p{Lu}\b\.?|$)|\p{Lu}[\p{L}'’-]*,\s*(?:and\s+)?\p{Lu}\b)/u;
/** A title in quotation marks, as numbered styles write it: “Attention is all you need,”. */
const QUOTED_TITLE = /[“"]([^”"]{3,}?)[,.]?[”"]/;
/** A title before its type, as GB/T 7714 writes it: 机器学习[M]. */
const TYPED_TITLE = /([^.。]+?)\s*\[(?:J|M|C|D|R|S|P|N|EB\/OL|Z)\]/;
/** A title after the year, as APA and Harvard write it: (2017). Attention is all you need. */
const TITLE_AFTER_YEAR =
  /(?:\((?:1[5-9]|20)\d\d[a-z]?\)\.?|(?<!\d)(?:1[5-9]|20)\d\d[a-z]?\.)\s*([^.?!。]+[?!]?)/;

/** Quoted, before its type, after the year, or else the first part after the authors. */
export function titleOf(reference: string): string | null {
  for (const pattern of [QUOTED_TITLE, TYPED_TITLE, TITLE_AFTER_YEAR]) {
    const title = pattern.exec(reference)?.[1]?.trim();
    if (title && titleWords(title).length > 0) return title;
  }
  const [, ...parts] = reference.replace(LIST_NUMBER, "").split(/[.。]\s+/);
  const afterAuthors = parts.findIndex((part) => !AUTHORS_GO_ON.test(part));
  return (
    parts
      .slice(Math.max(afterAuthors, 0))
      .find((part) => titleWords(part).length >= 2)
      ?.trim() ?? null
  );
}

/**
 * The share of a title's words that a reference holds, from 0 to 1. Two
 * words a PDF joined at a line's hyphen ("attentionbased") count as held.
 */
export function titleCoverage(reference: string, title: string): number {
  const wanted = titleWords(title);
  const held = new Set(titleWords(reference));
  const shared = new Set<string>();
  for (const [at, word] of wanted.entries()) {
    const next = wanted[at + 1];
    if (held.has(word)) shared.add(word);
    if (next !== undefined && held.has(`${word}${next}`)) shared.add(word).add(next);
  }
  const distinct = new Set(wanted).size;
  return distinct === 0 ? 0 : shared.size / distinct;
}

/** Where a reference's parts end: its authors, year, title and venue are set apart by these. */
const PARTS = /[.?!;:"“”。；：]+/;

/** A short title must be a part of its own: "Introduction" isn't "Introduction to quantum widgets". */
export function holdsTitle(reference: string, title: string): boolean {
  const wanted = titleWords(title);
  if (wanted.length === 0) return false;
  if (wanted.length >= SHORT_TITLE) return titleCoverage(reference, title) >= SAME_TITLE;
  return reference.split(PARTS).some((part) => {
    const words = titleWords(part);
    return words.length === wanted.length && words.every((word, at) => word === wanted[at]);
  });
}

const familyOf = (name: ReferenceName) => ("family" in name ? name.family : name.literal);

/** The authors as a reader would list them, in any language: "Hochreiter, Schmidhuber", "A, B, C, …". */
const authorList = (authors: readonly ReferenceName[]) =>
  [...authors.slice(0, 3).map(familyOf), ...(authors.length > 3 ? ["…"] : [])].join(", ");

export function differences(cited: CitedReference, record: WorkRecord): ReferenceDifference[] {
  const found: ReferenceDifference[] = [];
  const title = record.fields.title ?? "";
  if (title && !holdsTitle(cited.text, title)) {
    found.push({ field: "title", found: title, cited: null });
  }
  const { year } = cited;
  const slack = cited.preprint || record.preprint ? 1 : 0;
  if (
    year !== null &&
    record.years.length > 0 &&
    !record.years.some((each) => Math.abs(each - year) <= slack)
  ) {
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
