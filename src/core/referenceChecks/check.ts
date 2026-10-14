/**
 * One reference checked against Crossref (#711): by its DOI when it gives
 * one, else by Crossref's matching of the reference as written. It is found
 * (the work exists, as written), doesn't match (a record exists, but its
 * title, year or first author differ), or isn't found (no record is this
 * work, or its DOI leads nowhere).
 */
import type { ReferenceCheck } from "../api";
import type { CrossrefClient } from "./crossref";
import {
  type CitedReference,
  differences,
  SAME_TITLE,
  titleCoverage,
  titleWords,
  type WorkRecord,
} from "./match";

/** Fewer title words than this, and a reference says too little to look up. */
const MIN_WORDS = 4;
const URL = /https?:\/\/\S+/i;

/**
 * Whether a reference names a scholarly work a check can look up: one with a
 * DOI, or enough words that isn't only a web address (a web page, which
 * Crossref doesn't hold, would read as "not found").
 */
export function isCheckable(cited: CitedReference): boolean {
  if (cited.doi) return true;
  if (URL.test(cited.text)) return false;
  return titleWords(cited.text).length >= MIN_WORDS;
}

/** The record as the check shows it. */
function shown(record: WorkRecord): NonNullable<ReferenceCheck["record"]> {
  const { fields } = record;
  return {
    title: fields.title ?? "",
    authors: (fields.authors ?? []).map((name) =>
      "family" in name
        ? name.given
          ? `${name.family}, ${name.given}`
          : name.family
        : name.literal,
    ),
    year: record.years.length > 0 ? Math.min(...record.years) : null,
    doi: fields.doi || null,
  };
}

/** The check of a reference with a record: found as written, or how it differs. */
function compared(cited: CitedReference, record: WorkRecord): ReferenceCheck {
  const found = differences(cited, record);
  return {
    reference: cited.text,
    mark: found.length === 0 ? "found" : "mismatch",
    record: shown(record),
    differences: found,
    reason: null,
  };
}

/** Checks one reference; throws what the client throws when Crossref can't be asked. */
export async function checkReference(
  cited: CitedReference,
  crossref: CrossrefClient,
  signal: AbortSignal,
): Promise<ReferenceCheck> {
  const notFound: ReferenceCheck = {
    reference: cited.text,
    mark: "not-found",
    record: null,
    differences: [],
    reason: null,
  };
  if (cited.doi) {
    const record = await crossref.byDoi(cited.doi, signal);
    return record ? compared(cited, record) : notFound;
  }
  let best: { record: WorkRecord; coverage: number } | null = null;
  for (const record of await crossref.search(cited.text, signal)) {
    const coverage = titleCoverage(cited.text, record.fields.title ?? "");
    if (!best || coverage > best.coverage) best = { record, coverage };
  }
  return best && best.coverage >= SAME_TITLE ? compared(cited, best.record) : notFound;
}
