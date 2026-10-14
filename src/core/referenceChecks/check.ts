/**
 * One reference checked against Crossref (#711): by its DOI when it gives
 * one, else by Crossref's matching of the reference as written. It is found
 * (the work exists, as written), doesn't match (a record exists, but its
 * title, year or first author differ), or isn't found (no record is this
 * work, or its DOI leads nowhere). A DOI Crossref doesn't hold but another
 * agency registered, as DataCite does arXiv's, names a work that exists
 * whose details Crossref can't give: it isn't checked.
 */
import type { ReferenceCheck } from "../api";
import type { CrossrefClient } from "./crossref";
import {
  type CitedReference,
  differences,
  SAME_TITLE,
  titleCoverage,
  type WorkRecord,
} from "./match";

/** The record as the check shows it. */
function shown(record: WorkRecord): NonNullable<ReferenceCheck["record"]> {
  const { fields } = record;
  return {
    title: fields.title ?? "",
    authors: fields.authors ?? [],
    year: record.years.length > 0 ? Math.min(...record.years) : null,
    doi: fields.doi || null,
  };
}

/** The check of a reference with a record: found as written, or how it differs. */
function compared(
  cited: CitedReference,
  record: WorkRecord,
  by: ReferenceCheck["by"],
): ReferenceCheck {
  const found = differences(cited, record);
  return {
    reference: cited.text,
    mark: found.length === 0 ? "found" : "mismatch",
    by,
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
  const unmatched = (by: ReferenceCheck["by"]): ReferenceCheck => ({
    reference: cited.text,
    mark: "not-found",
    by,
    record: null,
    differences: [],
    reason: null,
  });
  if (cited.doi) {
    const record = await crossref.byDoi(cited.doi, signal);
    if (record) return compared(cited, record, "doi");
    if ((await crossref.agency(cited.doi, signal)) === null) return unmatched("doi");
    return { ...unmatched("doi"), mark: "not-checked", reason: "registered-elsewhere" };
  }
  // A work can have several records (a reprint, a later edition): the one that differs least is it.
  let best: { check: ReferenceCheck; coverage: number } | null = null;
  for (const record of await crossref.search(cited.text, signal)) {
    const coverage = titleCoverage(cited.text, record.fields.title ?? "");
    if (coverage < SAME_TITLE) continue;
    const check = compared(cited, record, "search");
    const fewer = best ? best.check.differences.length - check.differences.length : 1;
    if (fewer > 0 || (fewer === 0 && best && coverage > best.coverage)) best = { check, coverage };
  }
  return best?.check ?? unmatched("search");
}
