/**
 * The Source provider interface (ADR-0018): what one public service gives
 * Find sources. A provider searches by words, looks a work up by an
 * identifier, and says where its open full text is and fetches it. Adding
 * a provider (a law, filing or drug-label service) is a module that
 * implements this and is registered in ./providers; the framework around it
 * (consent, pace, merging, adding to the Library, the Answer's Tool and the
 * Library's results) doesn't change.
 */
import type { ExternalService, ReferenceFields, SourceVersion } from "../api";
import type { SourceHttp } from "./http";

/** What to search for, as each provider is asked it. */
export interface SourceQuery {
  /** The words, as the User or the model wrote them. */
  text: string;
  /** Inclusive years; null for any. */
  fromYear: number | null;
  toYear: number | null;
  /** Only works whose full text is open. */
  openAccessOnly: boolean;
  /** The most results to ask for. */
  limit: number;
}

/** The identifiers a work can carry; two works that share one are the same work. */
export interface WorkIds {
  /** Lower case, without "https://doi.org/". */
  doi?: string;
  /** Without its version: "1706.03762". */
  arxiv?: string;
  pmid?: string;
  /** With its prefix: "PMC10034092". */
  pmcid?: string;
  /** "W4362665266". */
  openalex?: string;
  /** A drug label's set id, which DailyMed and openFDA share: lower case. */
  setid?: string;
  /** A trial's registry number: "NCT04368728". */
  nct?: string;
}

/** An identifier the User typed, which providers look up instead of searching. */
export type WorkIdentifier =
  | { kind: "doi"; id: string }
  | { kind: "arxiv"; id: string }
  | { kind: "pmid"; id: string }
  | { kind: "setid"; id: string }
  | { kind: "nct"; id: string };

/**
 * Where a work's full text can be had openly. "pdf": a PDF file at `url`.
 * "sections": text the provider gives by section, through its
 * `fetchFullText` (PMC's articles now; provisions of a law later). "html":
 * the text is a web page at `url`, a publisher's or a repository's, saved as
 * `saveWebPage` saves one (#261), whatever its `provider`.
 */
export interface FullTextLocation {
  format: "pdf" | "sections" | "html";
  url: string;
  /**
   * The provider that fetches it, whose consent is asked first: its own
   * service. Null for an open-access address anywhere on the web (a
   * publisher's or a repository's), downloaded once, as a web page is added.
   */
  provider: string | null;
  version: SourceVersion | null;
  /** As written: "cc-by", or a licence's address. */
  licence: string | null;
  /** The date of this copy's version, when the provider says: a preprint's, or the published one's. */
  versionDate: string | null;
}

/** A work as one provider describes it. */
export interface ProviderWork {
  ids: WorkIds;
  fields: Partial<ReferenceFields>;
  citedBy: number | null;
  /** The date of the version described, ISO 8601 at the precision known. */
  versionDate: string | null;
  /** Its open copies, best first. Empty: none is known. */
  fullText: FullTextLocation[];
  /** Its page at its publisher or registry. */
  landingUrl: string | null;
  /** What the source itself says about relying on it (openFDA's disclaimer), shown with the result. */
  notice?: string;
}

/** One section of a text a provider gives by section: a heading, as the work cites it, and its paragraphs. */
export interface TextSection {
  heading: string;
  /** 1 under the title, 2 under that, … */
  level: number;
  paragraphs: string[];
}

/** A full text, fetched. */
export type FullText =
  | { format: "pdf"; bytes: Uint8Array; url: string }
  | {
      format: "sections";
      title: string;
      sections: TextSection[];
      url: string;
      /** What the text itself says it is licensed under, when it says. */
      licence: string | null;
      /** The date of the version, when the text says. */
      versionDate: string | null;
    };

/** What a provider is given for one call. */
export interface ProviderContext {
  http: SourceHttp;
  signal: AbortSignal;
}

/** How a provider's operator asks to be treated, cited in ADR-0018. */
export interface SourceTerms {
  /** Where the operator states its terms and limits. */
  url: string;
  /** The least time between two requests to it, from the end of one to the start of the next. */
  intervalMs: number;
}

export interface SourceProvider {
  /** Unique and stable: it is stored with what it added. */
  readonly id: string;
  /** As its operator names it. */
  readonly name: string;
  /** Where its requests go: consent is asked per service (the "find-sources" flow). */
  readonly service: ExternalService;
  readonly terms: SourceTerms;
  /** The identifiers `resolve` looks up. */
  readonly resolves?: readonly WorkIdentifier["kind"][];
  /** False: a search asks it only when it is named (drug labels, trials), not with every search for papers. */
  readonly searchedByDefault?: boolean;
  /** It takes an optional key of the User's, kept through `secrets.ts` and sent in a header. */
  readonly takesKey?: boolean;
  /** Searches by words; absent for one that only looks works up (Crossref). */
  search?(query: SourceQuery, context: ProviderContext): Promise<ProviderWork[]>;
  /** Looks a work up by an identifier it knows; null for one it doesn't hold. */
  resolve?(identifier: WorkIdentifier, context: ProviderContext): Promise<ProviderWork | null>;
  /** Fetches an open copy it offered with itself as `provider`. Throws `SourceError` when it can't. */
  fetchFullText?(location: FullTextLocation, context: ProviderContext): Promise<FullText>;
}
