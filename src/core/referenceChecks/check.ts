/**
 * One reference checked (#711). By its DOI (or arXiv number) in Crossref,
 * then OpenAlex and, for a preprint, arXiv, until a record is the work as
 * cited, else keeping the record that differs least. Without either, by
 * Crossref's matching of the reference as written, and, when Crossref has no
 * record of it or only one that differs from it, by its title in OpenAlex,
 * keeping the record that differs least (Crossref's on a tie).
 *
 * It is found (the work exists, as written), doesn't match (a record exists,
 * but its title, year or first author differ), isn't found (its DOI or arXiv
 * number leads nowhere), or isn't listed (neither Crossref nor OpenAlex has
 * a record like it; the work may still exist). A DOI another agency
 * registered that OpenAlex has no record of either names a work that exists
 * whose details can't be compared: it isn't checked.
 */
import type { ReferenceCheck, ReferenceCheckReason, ReferenceService } from "../api";
import { SourceError } from "../sources/http";
import type { Lookups } from "./lookups";
import {
  type CitedReference,
  differences,
  holdsTitle,
  titleCoverage,
  titleOf,
  type WorkRecord,
} from "./match";

/** A service couldn't be asked about a reference, and why. */
export class ServiceFailure extends Error {
  override name = "ServiceFailure";
  constructor(
    readonly service: ReferenceService,
    readonly reason: ReferenceCheckReason,
    cause?: unknown,
  ) {
    super(`${service}: ${reason}`, { cause });
  }
}

/** One check of a list: the services, and those it stopped asking (offline, or asked to slow down). */
export interface CheckRun {
  readonly lookups: Lookups;
  readonly stopped: Map<ReferenceService, ReferenceCheckReason>;
}

const reasonOf = (error: unknown): ReferenceCheckReason =>
  error instanceof SourceError &&
  (error.reason === "rate-limited" || error.reason === "unreachable")
    ? error.reason
    : "service-error";

/**
 * Asks `service`, unless the check stopped asking it. Offline, or asked to
 * slow down, it stops asking it for the rest of the list rather than failing
 * once a reference.
 */
async function ask<T>(
  run: CheckRun,
  service: ReferenceService,
  signal: AbortSignal,
  call: () => Promise<T>,
): Promise<T> {
  const stopped = run.stopped.get(service);
  if (stopped) throw new ServiceFailure(service, stopped);
  try {
    return await call();
  } catch (error) {
    if (signal.aborted) throw error;
    const reason = reasonOf(error);
    if (reason !== "service-error") run.stopped.set(service, reason);
    throw new ServiceFailure(service, reason, error);
  }
}

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
  service: ReferenceService,
): ReferenceCheck {
  const found = differences(cited, record);
  return {
    reference: cited.text,
    mark: found.length === 0 ? "found" : "mismatch",
    service,
    record: shown(record),
    differences: found,
    reason: null,
  };
}

/**
 * Of a search's records, the one that is the work: it holds the reference's
 * title and, among several (a reprint, a later edition), differs least.
 * `sameAuthor`: only a record whose first author is the reference's, as a
 * search by title alone needs.
 */
function bestOf(
  cited: CitedReference,
  records: readonly WorkRecord[],
  service: ReferenceService,
  sameAuthor: boolean,
): ReferenceCheck | null {
  let best: { check: ReferenceCheck; coverage: number } | null = null;
  for (const record of records) {
    const title = record.fields.title ?? "";
    if (!holdsTitle(cited.text, title)) continue;
    const check = compared(cited, record, service);
    if (sameAuthor && check.differences.some((each) => each.field === "authors")) continue;
    const coverage = titleCoverage(cited.text, title);
    const fewer = best ? best.check.differences.length - check.differences.length : 1;
    if (fewer > 0 || (fewer === 0 && best && coverage > best.coverage)) best = { check, coverage };
  }
  return best?.check ?? null;
}

/** Of several checks of a reference, the one that differs least; the first on a tie. */
const fewestDifferences = (checks: readonly ReferenceCheck[]): ReferenceCheck | null =>
  checks.reduce<ReferenceCheck | null>(
    (best, check) => (!best || check.differences.length < best.differences.length ? check : best),
    null,
  );

/** Checks one reference; throws `ServiceFailure` when a service it needs can't be asked. */
export async function checkReference(
  cited: CitedReference,
  run: CheckRun,
  signal: AbortSignal,
): Promise<ReferenceCheck> {
  const { crossref, openalex, arxiv } = run.lookups;
  const unmatched = (
    mark: ReferenceCheck["mark"],
    service: ReferenceService | null,
  ): ReferenceCheck => ({
    reference: cited.text,
    mark,
    service,
    record: null,
    differences: [],
    reason: null,
  });

  // arXiv's own DOI, which DataCite registers for every preprint, so Crossref holds none.
  const doi = cited.doi ?? (cited.arxiv ? `10.48550/arXiv.${cited.arxiv}` : null);
  if (doi) {
    // Each record the identifier leads to, until one is the work as cited; else the closest.
    const checks: ReferenceCheck[] = [];
    const lookUp = async (
      service: ReferenceService,
      call: () => Promise<WorkRecord | null>,
    ): Promise<ReferenceCheck | null> => {
      let record: WorkRecord | null;
      try {
        record = await ask(run, service, signal, call);
      } catch (error) {
        if (checks.length > 0 && error instanceof ServiceFailure) return null;
        throw error;
      }
      if (!record) return null;
      const check = compared(cited, record, service);
      checks.push(check);
      return check.mark === "found" ? check : null;
    };
    const found = cited.doi && (await lookUp("crossref", () => crossref.byDoi(doi, signal)));
    if (found) return found;
    if (checks.length === 0) {
      // Crossref says which agency registered a DOI, any agency's: none, and it leads nowhere.
      const agency = await ask(run, "crossref", signal, () => crossref.agency(doi, signal));
      if (agency === null) return unmatched("not-found", "crossref");
    }
    const inOpenAlex = await lookUp("openalex", () => openalex.byDoi(doi, signal));
    if (inOpenAlex) return inOpenAlex;
    const { arxiv: id } = cited;
    // Asked only about a number that exists: arXiv's API can hang on one that doesn't.
    const onArxiv = id && (await lookUp("arxiv", () => arxiv.byId(id, signal)));
    if (onArxiv) return onArxiv;
    return (
      fewestDifferences(checks) ?? {
        ...unmatched("not-checked", null),
        reason: "registered-elsewhere",
      }
    );
  }

  const inCrossref = bestOf(
    cited,
    await ask(run, "crossref", signal, () => crossref.search(cited.text, signal)),
    "crossref",
    false,
  );
  if (inCrossref?.mark === "found") return inCrossref;
  // Crossref's record may be another edition's (a reprint, a later year): OpenAlex may hold the one cited.
  const title = titleOf(cited.text) ?? cited.text;
  let searched: WorkRecord[];
  try {
    searched = await ask(run, "openalex", signal, () => openalex.search(title, signal));
  } catch (error) {
    if (inCrossref && error instanceof ServiceFailure) return inCrossref;
    throw error;
  }
  const inOpenAlex = bestOf(cited, searched, "openalex", true);
  if (
    inCrossref &&
    (!inOpenAlex || inOpenAlex.differences.length >= inCrossref.differences.length)
  ) {
    return inCrossref;
  }
  return inOpenAlex ?? unmatched("not-listed", null);
}
