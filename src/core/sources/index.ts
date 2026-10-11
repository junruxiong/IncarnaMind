/**
 * Find sources (#296, ADR-0018): searching public Source providers for works
 * and adding them to the Library. Searches go to each provider asked, with
 * the words and filters only, after the User has allowed the "find-sources"
 * flow to it (../consent), never in local mode; their results are merged
 * (./merge) and kept for this session, so the work the User adds is the one
 * they saw. Adding downloads an open copy (only open-access ones, never
 * from behind a paywall) into the data folder's "Sources" and adds it as a
 * Document, its reference record filled from the providers' records and
 * where it came from recorded (./store); with no open copy, the reference
 * record alone is kept. The Answer's Tool (./tool) runs the same search.
 */
import { mkdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { arxivIdOf, normalizeDoi } from "../../shared/references";
import type {
  AddFoundSourceResult,
  DataKind,
  Document,
  ExternalService,
  FindSourcesResult,
  FoundSource,
  SourceAccess,
  SourceFailureReason,
  SourceProviderInfo,
  SourceProviderOutcome,
} from "../api";
import type { Consent } from "../consent";
import { isInside } from "../documents/files";
import { ConsentDeclinedError, InvalidInputError, isRecord } from "../errors";
import { safeFileName, writeNewFile } from "../fileNames";
import { fillReference, linkReference, writeUnlinkedReference } from "../referenceRecords/store";
import type { Database } from "../storage";
import type { Tool } from "../tools";
import { downloadOpenPdf, fileOf } from "./fullText";
import { SourceError, type SourceHttp } from "./http";
import { licenceOf } from "./licences";
import { type MergedWork, mergeWorks } from "./merge";
import { inLibrary, purgedFoundFiles, recordFoundSource } from "./store";
import { sourcesTool } from "./tool";
import type {
  FullText,
  FullTextLocation,
  ProviderWork,
  SourceProvider,
  WorkIdentifier,
} from "./types";

export type { SourceProvider } from "./types";

/** What the "find-sources" flow sends: the search words and filters, and the IDs of works added. */
export const FIND_SOURCES_SENDS: readonly DataKind[] = ["search-terms", "work-ids"];

/** For a row whose provider is unknown: never expected, as every work was found by one. */
const UNKNOWN = { id: "unknown", name: "" };

/** How many works each provider is asked for. */
const RESULTS_PER_PROVIDER = 10;

/** How long a search's results answer the same search again, without asking the providers. */
const SEARCH_TTL_MS = 30 * 60 * 1000;

/** The most searches and works kept for the session. */
const MAX_SEARCHES = 50;
const MAX_WORKS = 1000;

/** The longest query sent. */
const MAX_QUERY_LENGTH = 500;

export interface SourcesOptions {
  db: Database;
  now: () => string;
  /** For how long results are kept. */
  clock: () => number;
  consent: Consent;
  /** Local mode is on: nothing is sent. */
  localOnly(): boolean;
  providers: readonly SourceProvider[];
  http: SourceHttp;
  /** Where downloaded copies are kept: the data folder's "Sources". */
  folder: string;
  /** Throws for a Folder that doesn't exist; null is Not in a Folder. */
  checkFolder(folderId: string | null): void;
  /**
   * Adds a file as a Document in a Folder, as Add Documents does; `added`
   * hears its id as it is stored, before it is processed. Null if the file
   * couldn't be added.
   */
  addFile(
    path: string,
    folderId: string | null,
    added: (documentId: string) => void,
  ): Promise<Document | null>;
  /** The live Document. */
  document(id: string): Document;
  /** Aborted when the core closes. */
  signal: AbortSignal;
  reportError(error: unknown): void;
}

interface ParsedInput {
  query: string;
  fromYear: number | null;
  toYear: number | null;
  openAccessOnly: boolean;
  providers: string[] | null;
}

function parseYear(value: unknown, what: string): number | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1000 || value > 9999) {
    throw new InvalidInputError(`${what} must be a year.`);
  }
  return value;
}

export function parseFindInput(input: unknown): ParsedInput {
  if (!isRecord(input) || typeof input.query !== "string") {
    throw new InvalidInputError("findSources expects a query.");
  }
  const query = input.query.replace(/\s+/g, " ").trim().slice(0, MAX_QUERY_LENGTH);
  if (query === "") throw new InvalidInputError("Type what to search for.");
  const fromYear = parseYear(input.fromYear, "fromYear");
  const toYear = parseYear(input.toYear, "toYear");
  if (fromYear !== null && toYear !== null && fromYear > toYear) {
    throw new InvalidInputError("The first year comes after the last.");
  }
  if (input.openAccessOnly !== undefined && typeof input.openAccessOnly !== "boolean") {
    throw new InvalidInputError("openAccessOnly must be true or false.");
  }
  const providers = input.providers;
  if (
    providers !== undefined &&
    (!Array.isArray(providers) || !providers.every((id) => typeof id === "string"))
  ) {
    throw new InvalidInputError("providers must be a list of provider ids.");
  }
  return {
    query,
    fromYear,
    toYear,
    openAccessOnly: input.openAccessOnly === true,
    providers: providers === undefined ? null : [...new Set(providers as string[])],
  };
}

/** A DOI, an arXiv ID or a PubMed ID the User typed, which is looked up rather than searched for. */
export function identifierOf(query: string): WorkIdentifier | null {
  const doi = normalizeDoi(query);
  if (doi) return { kind: "doi", id: doi.toLowerCase() };
  const arxiv =
    arxivIdOf(query) ?? /^(?:arxiv:\s*)?(\d{4}\.\d{4,5})(?:v\d+)?$/i.exec(query)?.[1] ?? null;
  if (arxiv) return { kind: "arxiv", id: arxiv };
  const pmid = /^(?:pmid:?\s*(\d{1,9})|(\d{7,9}))$/i.exec(query);
  const id = pmid?.[1] ?? pmid?.[2];
  return id ? { kind: "pmid", id } : null;
}

const FAILURES: Record<SourceError["reason"], SourceFailureReason> = {
  unreachable: "unreachable",
  "rate-limited": "rate-limited",
  "service-error": "service-error",
  "not-found": "service-error",
  "not-open": "service-error",
};

/**
 * A copy adding can download now. A web page ("html") is the seam for #261:
 * once `saveWebPage` is in IncarnaMind, it saves one as a snapshot Document,
 * and this takes it too.
 */
const downloadable = (copy: FullTextLocation) => copy.format !== "html";

/** How a found work's full text can be had: its first copy that adding can download, else a web page. */
function accessOf(work: MergedWork): SourceAccess {
  const first = work.fullText.find(downloadable) ?? work.fullText[0];
  if (!first) return { kind: "closed" };
  return {
    kind: "open",
    format: first.format === "pdf" ? "pdf" : first.format === "html" ? "web-page" : "text",
    version: first.version,
    licence: licenceOf(first.licence),
  };
}

export function createSources(options: SourcesOptions) {
  const { db, now, consent, providers, http } = options;
  const byId = new Map(providers.map((provider) => [provider.id, provider]));
  const nameOf = (id: string) => byId.get(id)?.name ?? id;
  /** Each work seen this session, by key: what `add` adds. */
  const works = new Map<string, MergedWork>();
  /** Recent searches, by their input: answered again without asking. */
  const searches = new Map<
    string,
    { at: number; works: string[]; providers: SourceProviderOutcome[] }
  >();
  /** Adding the same work twice at once adds it once. */
  const adding = new Map<string, Promise<AddFoundSourceResult>>();

  const keep = <V>(map: Map<string, V>, key: string, value: V, max: number) => {
    map.delete(key);
    map.set(key, value);
    while (map.size > max) {
      const oldest = map.keys().next().value;
      if (oldest === undefined) break;
      map.delete(oldest);
    }
  };

  const foundSource = (work: MergedWork): FoundSource => ({
    key: work.key,
    fields: work.fields,
    foundBy: work.foundBy.map((id) => ({ id, name: nameOf(id) })),
    citedBy: work.citedBy,
    versionDate: work.versionDate,
    access: accessOf(work),
    landingUrl: work.landingUrl,
    inLibrary: inLibrary(db, work.ids),
  });

  const context = (signal: AbortSignal) => ({
    http,
    signal: AbortSignal.any([signal, options.signal]),
  });

  /** Asks the User once per provider; false if they declined. */
  const allowed = async (service: ExternalService): Promise<boolean> => {
    try {
      await consent.ensure("find-sources", service);
      return true;
    } catch (error) {
      if (error instanceof ConsentDeclinedError) return false;
      throw error;
    }
  };

  async function ask(
    provider: SourceProvider,
    input: ParsedInput,
    identifier: WorkIdentifier | null,
    signal: AbortSignal,
  ): Promise<{ works: ProviderWork[]; outcome: SourceProviderOutcome }> {
    const outcome = (
      count: number,
      failure: SourceProviderOutcome["failure"] = null,
    ): SourceProviderOutcome => ({ id: provider.id, name: provider.name, count, failure });
    if (!(await allowed(provider.service))) {
      return {
        works: [],
        outcome: outcome(0, {
          reason: "declined",
          message: `Searching ${provider.name} isn't allowed.`,
        }),
      };
    }
    // Local mode may have come on while the User was being asked.
    if (options.localOnly()) return { works: [], outcome: outcome(0) };
    try {
      const found = identifier
        ? [await provider.resolve?.(identifier, context(signal))].filter(
            (work): work is ProviderWork => !!work,
          )
        : ((await provider.search?.(
            {
              text: input.query,
              fromYear: input.fromYear,
              toYear: input.toYear,
              openAccessOnly: input.openAccessOnly,
              limit: RESULTS_PER_PROVIDER,
            },
            context(signal),
          )) ?? []);
      return { works: found, outcome: outcome(found.length) };
    } catch (error) {
      if (!(error instanceof SourceError)) throw error;
      return {
        works: [],
        outcome: outcome(0, { reason: FAILURES[error.reason], message: error.message }),
      };
    }
  }

  /** The providers a search goes to: an identifier to those that look it up, words to those that search. */
  function targets(input: ParsedInput, identifier: WorkIdentifier | null): SourceProvider[] {
    const wanted = (provider: SourceProvider) =>
      input.providers === null ||
      input.providers.includes(provider.id) ||
      // Crossref looks DOIs up, whichever providers are searched.
      (identifier !== null && provider.search === undefined);
    if (!identifier) return providers.filter((each) => each.search && wanted(each));
    return providers.filter(
      (each) =>
        each.resolve &&
        wanted(each) &&
        // Each looks up the identifiers it holds.
        (identifier.kind === "doi"
          ? each.id !== "arxiv" && each.id !== "pubmed"
          : identifier.kind === "arxiv"
            ? each.id === "arxiv" || each.id === "openalex"
            : each.id === "pubmed" || each.id === "openalex"),
    );
  }

  async function search(input: unknown, signal: AbortSignal): Promise<FindSourcesResult> {
    const parsed = parseFindInput(input);
    if (options.localOnly()) return { outcome: "local-mode" };
    const identifier = identifierOf(parsed.query);
    const asked = targets(parsed, identifier);
    // The same search from the Library or an Answer, however its providers were named, while
    // the User's decision on each stays as it was.
    const cacheKey = () =>
      JSON.stringify([
        parsed.query,
        parsed.fromYear,
        parsed.toYear,
        parsed.openAccessOnly,
        asked.map((each) => `${each.id}:${consent.status("find-sources", each.service)}`),
      ]);
    const cached = searches.get(cacheKey());
    if (cached && options.clock() - cached.at < SEARCH_TTL_MS) {
      const kept = cached.works.flatMap((key) => {
        const work = works.get(key);
        return work ? [work] : [];
      });
      if (kept.length === cached.works.length) {
        return { outcome: "found", works: kept.map(foundSource), providers: cached.providers };
      }
    }
    const answers = await Promise.all(
      asked.map((provider) => ask(provider, parsed, identifier, signal)),
    );
    if (options.localOnly()) return { outcome: "local-mode" };
    const merged = mergeWorks(
      answers.map((answer, index) => ({ provider: asked[index]?.id ?? "", works: answer.works })),
    ).filter((work) => !parsed.openAccessOnly || work.fullText.length > 0);
    for (const work of merged) keep(works, work.key, work, MAX_WORKS);
    const outcomes = answers.map((answer) => answer.outcome);
    // A search a provider failed is asked again next time; one the User declined stays declined.
    if (outcomes.every((outcome) => !outcome.failure || outcome.failure.reason === "declined")) {
      // Keyed by the decisions made while it was asked.
      keep(
        searches,
        cacheKey(),
        { at: options.clock(), works: merged.map((work) => work.key), providers: outcomes },
        MAX_SEARCHES,
      );
    }
    return { outcome: "found", works: merged.map(foundSource), providers: outcomes };
  }

  /** Writes a fetched copy into the Sources folder, under the work's title; returns its path. */
  async function keepCopy(text: FullText, work: MergedWork): Promise<string> {
    const { data, extension } = fileOf(text);
    await mkdir(options.folder, { recursive: true });
    const name = await writeNewFile(
      options.folder,
      `${safeFileName(work.fields.title, "Untitled")}${extension}`,
      data,
    );
    return join(options.folder, name);
  }

  async function addWork(work: MergedWork, folderId: string | null): Promise<AddFoundSourceResult> {
    const existing = inLibrary(db, work.ids);
    if (existing?.documentId) {
      return {
        outcome: "document",
        document: options.document(existing.documentId),
        existing: true,
      };
    }
    let declined: string | null = null;
    const fetchable = work.fullText.filter(downloadable);
    for (const copy of fetchable) {
      const provider = copy.provider === null ? null : byId.get(copy.provider);
      // A copy only its provider can fetch, from a provider not here (or one that fetches none).
      if (provider === undefined || (provider && !provider.fetchFullText)) continue;
      if (provider && !(await allowed(provider.service))) {
        declined = provider.name;
        continue;
      }
      if (options.localOnly()) {
        return { outcome: "failed", reason: "local-mode", message: "Local mode is on." };
      }
      let path: string;
      let text: FullText;
      try {
        text = provider?.fetchFullText
          ? await provider.fetchFullText(copy, context(options.signal))
          : await downloadOpenPdf(http, copy.url, options.signal);
        path = await keepCopy(text, work);
      } catch (error) {
        if (!(error instanceof SourceError)) throw error;
        // That copy can't be had: the next one may be.
        continue;
      }
      const document = await options.addFile(path, folderId, (documentId) => {
        // Before the file is read: what its reading gives fills only what the providers left empty.
        db.transaction(() => {
          const at = now();
          const linked =
            existing && linkReference(db, existing.recordId, documentId, at)
              ? existing.recordId
              : null;
          fillReference(db, documentId, work.fields, at);
          recordFoundSource(
            db,
            {
              documentId,
              recordId: linked,
              provider: provider ?? byId.get(work.foundBy[0] ?? "") ?? UNKNOWN,
              workKey: work.key,
              address: text.url,
              licence: (text.format === "sections" ? text.licence : null) ?? copy.licence,
              version: copy.version,
              versionDate:
                (text.format === "sections" ? text.versionDate : null) ??
                copy.versionDate ??
                work.versionDate,
              fetchedAt: at,
            },
            at,
          );
        });
      });
      if (document) return { outcome: "document", document, existing: false };
      await rm(path, { force: true });
    }
    if (declined && !existing) {
      return {
        outcome: "failed",
        reason: "declined",
        message: `Downloading from ${declined} isn't allowed.`,
      };
    }
    const page = work.fullText.find((copy) => copy.format === "html");
    const reason = fetchable.length > 0 ? "download-failed" : page ? "web-page" : "closed";
    // Its open text, when that is a web page; else its page at its publisher.
    const landingUrl = (reason === "web-page" ? page?.url : null) ?? work.landingUrl;
    if (existing) {
      return {
        outcome: "reference",
        recordId: existing.recordId,
        reason,
        landingUrl,
        existing: true,
      };
    }
    const recordId = db.transaction(() => {
      const at = now();
      const id = writeUnlinkedReference(db, work.fields, at);
      recordFoundSource(
        db,
        {
          documentId: null,
          recordId: id,
          provider: byId.get(work.foundBy[0] ?? "") ?? UNKNOWN,
          workKey: work.key,
          address: work.landingUrl,
          licence: null,
          version: null,
          versionDate: work.versionDate,
          fetchedAt: null,
        },
        at,
      );
      return id;
    });
    return { outcome: "reference", recordId, reason, landingUrl, existing: false };
  }

  return {
    list(): SourceProviderInfo[] {
      return providers.map((provider) => ({
        id: provider.id,
        name: provider.name,
        service: provider.service,
        searches: provider.search !== undefined,
        consent: consent.status("find-sources", provider.service),
        termsUrl: provider.terms.url,
      }));
    },

    /** See `CoreApi.findSources`. */
    search: (input: unknown, signal: AbortSignal = options.signal) => search(input, signal),

    /**
     * The Tool an Answer can call: none in local mode, or when the User
     * declined every provider that searches.
     */
    tools(): Tool[] {
      if (options.localOnly()) return [];
      const searching = providers.filter(
        (each) => each.search && consent.status("find-sources", each.service) !== "declined",
      );
      return searching.length > 0 ? [sourcesTool(search, searching)] : [];
    },

    /** See `CoreApi.addFoundSource`. */
    async add(input: unknown): Promise<AddFoundSourceResult> {
      if (!isRecord(input) || typeof input.key !== "string" || input.key === "") {
        throw new InvalidInputError("addFoundSource expects a found work's key.");
      }
      const folderId = input.folderId ?? null;
      if (folderId !== null && typeof folderId !== "string") {
        throw new InvalidInputError("A Folder id must be text, or null.");
      }
      if (options.localOnly()) {
        return { outcome: "failed", reason: "local-mode", message: "Local mode is on." };
      }
      const work = works.get(input.key);
      if (!work) {
        return {
          outcome: "failed",
          reason: "not-found",
          message: "That work isn't in this session's results: search again to add it.",
        };
      }
      options.checkFolder(folderId);
      const running = adding.get(work.key);
      if (running) return running;
      const started = addWork(work, folderId).finally(() => adding.delete(work.key));
      adding.set(work.key, started);
      return started;
    },

    /** Removes the copies of Documents deleted permanently: their text goes with them. */
    async dropPurged(): Promise<void> {
      await Promise.all(
        purgedFoundFiles(db)
          .filter((path) => isInside(options.folder, path))
          .map((path) => rm(path, { force: true })),
      ).catch(options.reportError);
    },
  };
}

export type Sources = ReturnType<typeof createSources>;
