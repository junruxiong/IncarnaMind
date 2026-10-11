/**
 * Reading a work's record from CSL JSON (#286), the Citation Style
 * Language's data format, which Crossref's JSON for a work follows too (with
 * lists where CSL has strings): Look up reads Crossref's answers with it, and
 * importing CSL JSON from Zotero and others (#287) can too.
 */
import { isCjkName, normalizeDoi, parseIssued } from "../../shared/references";
import type { ReferenceFields, ReferenceName, ReferenceType } from "../api";
import { decodeEntities } from "../documents/formats/xml";
import { isRecord } from "../errors";

/** Markup some titles carry ("<i>In vivo</i>", MathML), and runs of white space, taken out. */
export const plain = (text: string) =>
  decodeEntities(text.replace(/<[^>]*>/g, ""))
    .replace(/\s+/g, " ")
    .trim();

/** CSL's types, and Crossref's own names for theirs, as the record's types. */
const TYPES: Readonly<Record<string, ReferenceType>> = {
  "article-journal": "article-journal",
  "journal-article": "article-journal",
  "paper-conference": "paper-conference",
  "proceedings-article": "paper-conference",
  chapter: "chapter",
  "book-chapter": "chapter",
  "book-section": "chapter",
  "book-part": "chapter",
  book: "book",
  monograph: "book",
  "edited-book": "book",
  "reference-book": "book",
  thesis: "thesis",
  dissertation: "thesis",
  report: "report",
  "report-component": "report",
  webpage: "webpage",
  article: "article",
  "posted-content": "article",
};

/** A string field: CSL writes a string, Crossref a list of them. */
function text(value: unknown): string {
  const first = Array.isArray(value) ? value[0] : value;
  return typeof first === "string" ? plain(first) : typeof first === "number" ? String(first) : "";
}

function names(value: unknown): ReferenceName[] {
  if (!Array.isArray(value)) return [];
  const found: ReferenceName[] = [];
  for (const entry of value) {
    if (!isRecord(entry)) continue;
    const family = text(entry.family);
    const given = text(entry.given);
    const whole = text(entry.literal) || text(entry.name);
    if (family) {
      // A name in Chinese, Japanese or Korean is kept whole, family name first.
      found.push(
        isCjkName(family + given) || !given ? { literal: `${family}${given}` } : { family, given },
      );
    } else if (whole) {
      found.push({ literal: whole });
    }
  }
  return found;
}

/** A CSL date ({ "date-parts": [[2010, 8, 4]] }, or a raw one), as the record keeps it. */
function date(value: unknown): string {
  if (!isRecord(value)) return "";
  const parts = Array.isArray(value["date-parts"]) ? value["date-parts"][0] : undefined;
  if (Array.isArray(parts) && typeof parts[0] === "number") {
    return parseIssued(parts.filter((part) => typeof part === "number").join("-")) ?? "";
  }
  return parseIssued(text(value.raw) || text(value.literal)) ?? "";
}

/**
 * A work's record from CSL JSON, or from Crossref's JSON for a work, which
 * uses CSL's names: the fields it gives. Fields it doesn't are left out.
 */
export function fromCsl(item: unknown): Partial<ReferenceFields> {
  if (!isRecord(item)) return {};
  const found: Partial<ReferenceFields> = {};
  const type = TYPES[text(item.type)];
  if (type) found.type = type;
  const title = text(item.title);
  if (title) found.title = title;
  const authors = names(item.author);
  if (authors.length > 0) found.authors = authors;
  const issued =
    date(item.issued) ||
    date(item["published-print"]) ||
    date(item["published-online"]) ||
    date(item.published);
  if (issued) found.issued = issued;
  const container = text(item["container-title"]);
  if (container) found.containerTitle = container;
  for (const [from, to] of [
    ["volume", "volume"],
    ["issue", "issue"],
    ["page", "pages"],
    ["publisher", "publisher"],
  ] as const) {
    const value = text(item[from]);
    if (value) found[to] = to === "pages" ? value.replace(/(\d)\s*-\s*(\d)/, "$1–$2") : value;
  }
  const doi = normalizeDoi(text(item.DOI));
  if (doi) found.doi = doi;
  // A DOI's own address says nothing the DOI doesn't.
  const url = text(item.URL);
  if (url && !/^https?:\/\/(?:dx\.)?doi\.org\//i.test(url)) found.url = url;
  const language = text(item.language);
  if (/^[a-z]{2,3}(?:[-_][a-z0-9]{2,8})*$/i.test(language))
    found.language = language.replace(/_/g, "-");
  return found;
}
