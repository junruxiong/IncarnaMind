/**
 * OpenAlex (https://openalex.org): search and metadata for works in every
 * field, with their open-access copies and citation counts. Its data is CC0
 * ("all data is CC0", https://help.openalex.org/how-to-use-the-api/api-overview).
 * Basic use needs no key: a search costs $0.001 of a free daily $0.10 (about
 * 100 searches a day), a lookup of one work is free, and more than 100
 * requests a second is refused with a 429
 * (https://help.openalex.org/how-to-use-the-api/rate-limits-and-authentication,
 * https://help.openalex.org/access/example-costs/). Its cached PDFs
 * (content.openalex.org) cost $0.01 each, so open copies are fetched from
 * where OpenAlex says they are instead.
 */
import { arxivIdOf, parseName } from "../../../shared/references";
import type { ReferenceFields, ReferenceName, ReferenceType, SourceVersion } from "../../api";
import { isRecord } from "../../errors";
import { SourceError } from "../http";
import type {
  FullTextLocation,
  ProviderContext,
  ProviderWork,
  SourceProvider,
  SourceQuery,
  WorkIdentifier,
  WorkIds,
} from "../types";
import { optionalNumber, plainTitle, text } from "./common";

export const OPENALEX_URL = "https://api.openalex.org";

/** What a search asks for of each work: the fields mapped below, nothing more. */
const SELECT = [
  "id",
  "doi",
  "display_name",
  "publication_date",
  "authorships",
  "primary_location",
  "locations",
  "best_oa_location",
  "cited_by_count",
  "type",
  "language",
  "ids",
  "biblio",
].join(",");

const VERSIONS: Record<string, SourceVersion> = {
  publishedVersion: "published",
  acceptedVersion: "accepted",
  submittedVersion: "submitted",
};

const tail = (url: string, prefix: RegExp): string | undefined =>
  prefix.test(url) ? url.replace(prefix, "") : undefined;

/** The PMC id in a PubMed Central address, e.g. "https://www.ncbi.nlm.nih.gov/pmc/articles/10172116". */
function pmcidOf(url: string): string | undefined {
  const match =
    /(?:pmc\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov\/pmc)\/articles\/(?:PMC)?(\d+)/i.exec(url);
  return match?.[1] ? `PMC${match[1]}` : undefined;
}

function typeOf(work: Record<string, unknown>, sourceType: string): ReferenceType {
  switch (text(work.type)) {
    case "article":
    case "review":
    case "letter":
    case "editorial":
      return sourceType === "conference" ? "paper-conference" : "article-journal";
    case "preprint":
      return "article";
    case "book-chapter":
      return "chapter";
    case "book":
      return "book";
    case "dissertation":
      return "thesis";
    case "report":
      return "report";
    default:
      return "document";
  }
}

const locationsOf = (work: Record<string, unknown>): Record<string, unknown>[] => {
  const best = isRecord(work.best_oa_location) ? [work.best_oa_location] : [];
  const all = Array.isArray(work.locations) ? work.locations.filter(isRecord) : [];
  return [...best, ...all];
};

/**
 * Its open copies, best first: PDFs where OpenAlex knows one, an arXiv copy
 * through arXiv (so arXiv's pace and consent apply), and a PubMed Central
 * copy through PubMed (PMC's site is closed to automated downloads; its
 * E-utilities are the way it allows). The best open location with no PDF
 * is the publisher's web page, a copy to save as a web page (#261).
 */
function openCopies(work: Record<string, unknown>, ids: WorkIds, date: string | null) {
  const copies: FullTextLocation[] = [];
  const seen = new Set<string>();
  const add = (copy: FullTextLocation) => {
    if (seen.has(copy.url)) return;
    seen.add(copy.url);
    copies.push(copy);
  };
  locationsOf(work).forEach((location, index) => {
    if (location.is_oa !== true) return;
    const best = index === 0 && location === work.best_oa_location;
    const version = VERSIONS[text(location.version)] ?? null;
    const licence = text(location.license) || null;
    const pdf = text(location.pdf_url);
    const landing = text(location.landing_page_url);
    const pmcid = pmcidOf(pdf) ?? pmcidOf(landing);
    const arxiv = arxivIdOf(pdf) ?? arxivIdOf(landing);
    if (pmcid) {
      add({
        format: "sections",
        url: `https://pmc.ncbi.nlm.nih.gov/articles/${pmcid}/`,
        provider: "pubmed",
        version,
        licence,
        versionDate: null,
      });
    } else if (arxiv) {
      add({
        format: "pdf",
        url: `https://arxiv.org/pdf/${arxiv}`,
        provider: "arxiv",
        version: "submitted",
        licence,
        versionDate: null,
      });
    } else if (pdf && !/^https?:\/\/content\.openalex\.org\//.test(pdf)) {
      add({
        format: "pdf",
        url: pdf,
        provider: null,
        version,
        licence,
        versionDate: version === "published" ? date : null,
      });
    } else if (best && landing && !/^https?:\/\/(?:www\.)?doaj\.org\//.test(landing)) {
      add({
        format: "html",
        url: landing,
        provider: null,
        version,
        licence,
        versionDate: version === "published" ? date : null,
      });
    }
  });
  if (ids.pmcid && !copies.some((copy) => copy.provider === "pubmed")) {
    // Every PMC article is free to read; whether its full text may be fetched, PMC says when asked.
    add({
      format: "sections",
      url: `https://pmc.ncbi.nlm.nih.gov/articles/${ids.pmcid}/`,
      provider: "pubmed",
      version: null,
      licence: null,
      versionDate: null,
    });
  }
  return copies;
}

/** One OpenAlex work as a provider's work. */
export function fromOpenAlex(work: Record<string, unknown>): ProviderWork | null {
  const title = plainTitle(text(work.display_name));
  if (!title) return null;
  const doi = tail(text(work.doi), /^https?:\/\/(?:dx\.)?doi\.org\//i)?.toLowerCase();
  const rawIds = isRecord(work.ids) ? work.ids : {};
  const ids: WorkIds = {
    ...(doi && { doi }),
    ...(tail(text(work.id), /^https:\/\/openalex\.org\//) && {
      openalex: tail(text(work.id), /^https:\/\/openalex\.org\//),
    }),
    ...(tail(text(rawIds.pmid), /^https?:\/\/pubmed\.ncbi\.nlm\.nih\.gov\//) && {
      pmid: tail(text(rawIds.pmid), /^https?:\/\/pubmed\.ncbi\.nlm\.nih\.gov\//)?.replace(
        /\/$/,
        "",
      ),
    }),
    ...(pmcidOf(text(rawIds.pmcid)) && { pmcid: pmcidOf(text(rawIds.pmcid)) }),
  };
  const arxivDoi = /^10\.48550\/arxiv\.(.+)$/i.exec(doi ?? "");
  if (arxivDoi?.[1]) ids.arxiv = arxivDoi[1];
  // A copy in PubMed Central names the article's PMC id, which OpenAlex's ids may not.
  for (const location of locationsOf(work)) {
    ids.pmcid ??= pmcidOf(text(location.pdf_url)) ?? pmcidOf(text(location.landing_page_url));
  }
  const primary = isRecord(work.primary_location) ? work.primary_location : {};
  const source = isRecord(primary.source) ? primary.source : {};
  const biblio = isRecord(work.biblio) ? work.biblio : {};
  const first = text(biblio.first_page);
  const last = text(biblio.last_page);
  const authors = (Array.isArray(work.authorships) ? work.authorships : [])
    .map((authorship) =>
      isRecord(authorship) && isRecord(authorship.author)
        ? parseName(text(authorship.author.display_name))
        : null,
    )
    .filter((name): name is ReferenceName => name !== null);
  const date = text(work.publication_date) || null;
  const type = typeOf(work, text(source.type));
  const fields: Partial<ReferenceFields> = {
    type,
    title,
    authors,
    ...(date && { issued: date }),
    ...(text(source.display_name) && { containerTitle: text(source.display_name) }),
    ...(text(biblio.volume) && { volume: text(biblio.volume) }),
    ...(text(biblio.issue) && { issue: text(biblio.issue) }),
    ...(first && { pages: last && last !== first ? `${first}–${last}` : first }),
    ...(text(source.host_organization_name) && {
      publisher: text(source.host_organization_name),
    }),
    ...(doi && { doi }),
    ...(ids.arxiv && { url: `https://arxiv.org/abs/${ids.arxiv}` }),
    ...(text(work.language) && { language: text(work.language) }),
  };
  return {
    ids,
    fields,
    citedBy: optionalNumber(work.cited_by_count),
    versionDate: date,
    fullText: openCopies(work, ids, date),
    landingUrl: text(primary.landing_page_url) || (doi ? `https://doi.org/${doi}` : null),
  };
}

function worksOf(body: string): Record<string, unknown>[] {
  let answer: unknown;
  try {
    answer = JSON.parse(body);
  } catch {
    throw new SourceError("service-error", "OpenAlex's answer isn't JSON.");
  }
  if (!isRecord(answer)) throw new SourceError("service-error", "OpenAlex's answer is empty.");
  if (Array.isArray(answer.results)) return answer.results.filter(isRecord);
  return [answer];
}

export function createOpenAlex(baseUrl: string = OPENALEX_URL): SourceProvider {
  const lane = "openalex";
  const terms = {
    url: "https://help.openalex.org/how-to-use-the-api/rate-limits-and-authentication",
    // Far under its 100 a second: one person's searches.
    intervalMs: 100,
  };
  const get = (url: string, context: ProviderContext) =>
    context.http.text(url, {
      lane,
      intervalMs: terms.intervalMs,
      accept: "application/json",
      signal: context.signal,
    });
  return {
    id: "openalex",
    name: "OpenAlex",
    service: { id: "https://api.openalex.org", name: "OpenAlex" },
    terms,
    async search(query: SourceQuery, context) {
      const filters = [
        ...(query.fromYear !== null ? [`from_publication_date:${query.fromYear}-01-01`] : []),
        ...(query.toYear !== null ? [`to_publication_date:${query.toYear}-12-31`] : []),
        ...(query.openAccessOnly ? ["is_oa:true"] : []),
      ];
      const params = new URLSearchParams({
        search: query.text,
        per_page: String(query.limit),
        select: SELECT,
      });
      if (filters.length > 0) params.set("filter", filters.join(","));
      const body = await get(`${baseUrl}/works?${params}`, context);
      return worksOf(body)
        .map(fromOpenAlex)
        .filter((work): work is ProviderWork => work !== null);
    },
    async resolve(identifier: WorkIdentifier, context) {
      const path =
        identifier.kind === "doi"
          ? `doi:${identifier.id}`
          : identifier.kind === "pmid"
            ? `pmid:${identifier.id}`
            : `doi:10.48550/arXiv.${identifier.id}`;
      try {
        // As OpenAlex writes its own ids ("doi:10.7717/peerj.4375"): the rest of a DOI escaped.
        const escaped = encodeURIComponent(path).replace(/%3A/gi, ":").replace(/%2F/gi, "/");
        const body = await get(`${baseUrl}/works/${escaped}?select=${SELECT}`, context);
        const [work] = worksOf(body);
        return work ? fromOpenAlex(work) : null;
      } catch (error) {
        if (error instanceof SourceError && error.reason === "not-found") return null;
        throw error;
      }
    },
  };
}
