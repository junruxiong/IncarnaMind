/**
 * Every answer a service gave, kept for good (migration 96): checking a list
 * again asks no service twice, so it never spends OpenAlex's free daily
 * allowance, which everyone on a network shares. Kept by DOI or arXiv number,
 * else by the reference's normalised title (and year, for Crossref's matching,
 * which reads the whole reference). A failure isn't kept, so it is asked again.
 */
import { randomUUID } from "node:crypto";
import { titleWords, yearOf } from "../../shared/referenceList";
import type { ReferenceName, ReferenceService } from "../api";
import { isRecord } from "../errors";
import type { Database } from "../storage";
import type { Lookups } from "./lookups";
import { titleOf, type WorkRecord } from "./match";

/** What a check reads of a record, which is all the cache keeps. */
const kept = (record: WorkRecord) => ({
  title: record.fields.title ?? "",
  authors: record.fields.authors ?? [],
  doi: record.fields.doi ?? null,
  years: record.years,
  preprint: record.preprint === true,
});

const isName = (value: unknown): value is ReferenceName =>
  isRecord(value) &&
  (typeof value.literal === "string" ||
    (typeof value.family === "string" && typeof value.given === "string"));

function recordOf(value: unknown): WorkRecord | null {
  if (!isRecord(value) || typeof value.title !== "string") return null;
  const { authors, doi, years } = value;
  return {
    fields: {
      title: value.title,
      authors: Array.isArray(authors) ? authors.filter(isName) : [],
      ...(typeof doi === "string" && { doi }),
    },
    years: Array.isArray(years)
      ? years.filter((year): year is number => typeof year === "number")
      : [],
    preprint: value.preprint === true,
  };
}

const recordsOf = (value: unknown): WorkRecord[] =>
  Array.isArray(value) ? value.map(recordOf).filter((each) => each !== null) : [];

const titleKey = (title: string) => titleWords(title).join(" ");

export function cachedLookups(lookups: Lookups, db: Database, now: () => string): Lookups {
  const lookUp = (service: ReferenceService, kind: string, key: string) =>
    db.get<{ answer: string }>(
      `SELECT answer FROM reference_lookups
       WHERE service = ? AND kind = ? AND lookup_key = ? AND deleted_at IS NULL`,
      [service, kind, key],
    )?.answer;

  const remember = async <T>(
    service: ReferenceService,
    kind: string,
    key: string,
    ask: () => Promise<T>,
    write: (answer: T) => unknown,
    read: (stored: unknown) => T,
  ): Promise<T> => {
    const stored = lookUp(service, kind, key);
    if (stored !== undefined) {
      try {
        return read(JSON.parse(stored));
      } catch (error) {
        console.error(`A kept ${service} answer couldn't be read; asking again.`, error);
      }
    }
    const answer = await ask();
    const at = now();
    db.run(
      `INSERT INTO reference_lookups (id, service, kind, lookup_key, answer, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT (service, kind, lookup_key) WHERE deleted_at IS NULL
       DO UPDATE SET answer = excluded.answer, updated_at = excluded.updated_at`,
      [randomUUID(), service, kind, key, JSON.stringify(write(answer)), at, at],
    );
    return answer;
  };

  const record = (
    service: ReferenceService,
    kind: string,
    key: string,
    ask: () => Promise<WorkRecord | null>,
  ) => remember(service, kind, key, ask, (answer) => answer && kept(answer), recordOf);
  const records = (
    service: ReferenceService,
    kind: string,
    key: string,
    ask: () => Promise<WorkRecord[]>,
  ) => remember(service, kind, key, ask, (answer) => answer.map(kept), recordsOf);

  const { crossref, openalex, arxiv } = lookups;
  return {
    crossref: {
      byDoi: (doi, signal) =>
        record("crossref", "doi", doi.toLowerCase(), () => crossref.byDoi(doi, signal)),
      agency: (doi, signal) =>
        remember(
          "crossref",
          "agency",
          doi.toLowerCase(),
          () => crossref.agency(doi, signal),
          (agency) => agency,
          (stored) => (typeof stored === "string" ? stored : null),
        ),
      search: (reference, signal) =>
        records(
          "crossref",
          "search",
          `${titleKey(titleOf(reference) ?? reference)}|${yearOf(reference) ?? ""}`,
          () => crossref.search(reference, signal),
        ),
    },
    openalex: {
      byDoi: (doi, signal) =>
        record("openalex", "doi", doi.toLowerCase(), () => openalex.byDoi(doi, signal)),
      search: (title, signal) =>
        records("openalex", "title", titleKey(title), () => openalex.search(title, signal)),
    },
    arxiv: {
      byId: (id, signal) => record("arxiv", "id", id, () => arxiv.byId(id, signal)),
    },
  };
}
