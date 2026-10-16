/**
 * The services a reference is looked up in (#711): Crossref first; OpenAlex
 * when Crossref has no record of it or one that differs, which holds many conference papers and
 * preprints Crossref doesn't; arXiv for a reference given by its arXiv
 * number. Each on the lane Find sources uses for it, at the pace its terms
 * ask, with no key and no contact address (ADR-0018).
 */
import type { ExternalService, ReferenceService } from "../api";
import { ARXIV, CROSSREF } from "../referenceRecords/lookup";
import type { SourceHttp } from "../sources/http";
import { ARXIV_URLS, createArxiv } from "../sources/providers/arxiv";
import { createOpenAlex, OPENALEX_URL } from "../sources/providers/openalex";
import type { ProviderWork } from "../sources/types";
import { CROSSREF_URL, type CrossrefClient, createCrossrefClient } from "./crossref";
import type { WorkRecord } from "./match";

/** How many of OpenAlex's best matches a search reads. */
const ROWS = 5;

export const OPENALEX: ExternalService = { id: OPENALEX_URL, name: "OpenAlex" };

/** Each service as consent lists it. */
export const REFERENCE_SERVICES: Readonly<Record<ReferenceService, ExternalService>> = {
  crossref: CROSSREF,
  openalex: OPENALEX,
  arxiv: ARXIV,
};

export interface Lookups {
  crossref: CrossrefClient;
  openalex: {
    /** The work a DOI names; null when OpenAlex has none. */
    byDoi(doi: string, signal: AbortSignal): Promise<WorkRecord | null>;
    /** The works whose records best match a title, best first. */
    search(title: string, signal: AbortSignal): Promise<WorkRecord[]>;
  };
  arxiv: {
    /** The preprint an arXiv number names; null when arXiv has none. */
    byId(id: string, signal: AbortSignal): Promise<WorkRecord | null>;
  };
}

const yearOf = (date: string | null | undefined) => {
  const year = Number(date?.slice(0, 4));
  return Number.isInteger(year) && year > 0 ? year : null;
};

/** A work Find sources' providers give, as a check reads it: every year it gives. */
function recordOf(work: ProviderWork | null): WorkRecord | null {
  if (!work?.fields.title) return null;
  const years = [yearOf(work.fields.issued), yearOf(work.versionDate)].filter(
    (year): year is number => year !== null,
  );
  return { fields: work.fields, years: [...new Set(years)] };
}

export function createLookups(
  http: SourceHttp,
  urls: Partial<Record<ReferenceService, string>> = {},
): Lookups {
  const openalex = createOpenAlex(urls.openalex ?? OPENALEX_URL);
  const arxiv = createArxiv({
    api: urls.arxiv ?? ARXIV_URLS.api,
    files: urls.arxiv ?? ARXIV_URLS.files,
  });
  return {
    crossref: createCrossrefClient(http, urls.crossref ?? CROSSREF_URL),
    openalex: {
      async byDoi(doi, signal) {
        return recordOf(await openalex.resolve({ kind: "doi", id: doi }, { http, signal }));
      },
      async search(title, signal) {
        const works = await openalex.search(
          { text: title, fromYear: null, toYear: null, openAccessOnly: false, limit: ROWS },
          { http, signal },
        );
        return works.map(recordOf).filter((record): record is WorkRecord => record !== null);
      },
    },
    arxiv: {
      async byId(id, signal) {
        return recordOf(await arxiv.resolve({ kind: "arxiv", id }, { http, signal }));
      },
    },
  };
}
