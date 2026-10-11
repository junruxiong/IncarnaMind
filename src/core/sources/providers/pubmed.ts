/**
 * PubMed and PubMed Central, through NCBI's E-utilities
 * (https://www.ncbi.nlm.nih.gov/books/NBK25497/): ESearch finds PubMed ids,
 * ESummary describes them, and EFetch gives a PubMed Central article's full
 * text as JATS XML when its publisher allows it. NCBI asks for no more than
 * three requests a second from one address without a key, and a `tool`
 * name; its `email` must be a developer's, never a user's, so none is sent.
 * PMC allows automated retrieval only through its own services, E-utilities
 * among them (https://pmc.ncbi.nlm.nih.gov/tools/amdataset), so a PMC copy is
 * fetched with EFetch, never from PMC's pages. Each article carries its own
 * licence, which the Document keeps.
 */
import { parseIssued } from "../../../shared/references";
import type { ReferenceFields, ReferenceName } from "../../api";
import { isRecord } from "../../errors";
import { SourceError } from "../http";
import { readJats } from "../jats";
import type {
  ProviderContext,
  ProviderWork,
  SourceProvider,
  SourceQuery,
  WorkIdentifier,
} from "../types";
import { jsonObject, plainTitle, text } from "./common";

export const PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov";

/** Three requests a second, NCBI's limit without a key, with room to spare. */
const INTERVAL_MS = 400;

/** Who is asking, as NCBI asks callers to say. */
const TOOL = "IncarnaMind";

/** PubMed's three-letter languages, as the record keeps languages (BCP 47). */
const LANGUAGES: Readonly<Record<string, string>> = {
  eng: "en",
  chi: "zh",
  jpn: "ja",
  kor: "ko",
  ger: "de",
  fre: "fr",
  spa: "es",
  ita: "it",
  por: "pt",
  rus: "ru",
  dut: "nl",
  pol: "pl",
  tur: "tr",
};

/** "Guo C" (PubMed writes family names, then initials) as a name; a group's name kept whole. */
function nameOf(author: Record<string, unknown>): ReferenceName | null {
  const name = text(author.name);
  if (!name) return null;
  if (text(author.authtype) !== "Author") return { literal: name };
  const match = /^(.+?)\s+([A-Z]{1,4})$/.exec(name);
  return match?.[1] && match[2] ? { family: match[1], given: match[2] } : { literal: name };
}

/** One ESummary record as a provider's work. */
export function fromPubmed(record: Record<string, unknown>): ProviderWork | null {
  const pmid = text(record.uid);
  const title = plainTitle(text(record.title));
  if (!pmid || !title) return null;
  const ids: Record<string, string> = {};
  for (const id of Array.isArray(record.articleids) ? record.articleids : []) {
    if (!isRecord(id)) continue;
    ids[text(id.idtype)] = text(id.value);
  }
  const doi = ids.doi?.toLowerCase();
  const pmcid = /^PMC\d+$/.test(ids.pmc ?? "") ? ids.pmc : undefined;
  const authors = (Array.isArray(record.authors) ? record.authors : [])
    .map((author) => (isRecord(author) ? nameOf(author) : null))
    .filter((name): name is ReferenceName => name !== null);
  const issued = parseIssued(text(record.sortpubdate)) ?? parseIssued(text(record.pubdate));
  const languages = Array.isArray(record.lang) ? record.lang : [];
  const language = LANGUAGES[text(languages[0])];
  const fields: Partial<ReferenceFields> = {
    type: "article-journal",
    title,
    authors,
    ...(issued && { issued }),
    ...(text(record.source) && { containerTitle: text(record.source) }),
    ...(text(record.volume) && { volume: text(record.volume) }),
    ...(text(record.issue) && { issue: text(record.issue) }),
    ...(text(record.pages) && { pages: text(record.pages).replace(/(\d)-(\d)/, "$1–$2") }),
    ...(doi && { doi }),
    ...(language && { language }),
  };
  return {
    ids: { pmid, ...(doi && { doi }), ...(pmcid && { pmcid }) },
    fields,
    citedBy: null,
    versionDate: issued,
    fullText: pmcid
      ? [
          {
            format: "sections",
            url: `https://pmc.ncbi.nlm.nih.gov/articles/${pmcid}/`,
            provider: "pubmed",
            version: null,
            licence: null,
            versionDate: null,
          },
        ]
      : [],
    landingUrl: `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`,
  };
}

/** The words as a PubMed search, with its filters. */
export function pubmedTerm(query: SourceQuery): string {
  // "free full text" is PubMed's subset of articles free to read, PMC's among them.
  return query.openAccessOnly ? `(${query.text}) AND free full text[sb]` : query.text;
}

export function createPubmed(baseUrl: string = PUBMED_URL): SourceProvider {
  const terms = {
    url: "https://www.ncbi.nlm.nih.gov/books/NBK25497/",
    intervalMs: INTERVAL_MS,
  };
  const get = (
    path: string,
    params: Record<string, string>,
    accept: string,
    context: ProviderContext,
  ) =>
    context.http.text(
      `${baseUrl}/entrez/eutils/${path}?${new URLSearchParams({ ...params, tool: TOOL })}`,
      { lane: "ncbi", intervalMs: terms.intervalMs, accept, signal: context.signal },
    );

  async function summaries(ids: readonly string[], context: ProviderContext) {
    if (ids.length === 0) return [];
    const answer = jsonObject(
      await get(
        "esummary.fcgi",
        { db: "pubmed", id: ids.join(","), retmode: "json" },
        "application/json",
        context,
      ),
      "PubMed",
    );
    const result = isRecord(answer.result) ? answer.result : {};
    return ids
      .map((id) => result[id])
      .filter(isRecord)
      .map(fromPubmed)
      .filter((work): work is ProviderWork => work !== null);
  }

  return {
    id: "pubmed",
    resolves: ["pmid"],
    name: "PubMed",
    service: { id: "https://eutils.ncbi.nlm.nih.gov", name: "PubMed" },
    terms,
    async search(query: SourceQuery, context) {
      const params: Record<string, string> = {
        db: "pubmed",
        term: pubmedTerm(query),
        retmode: "json",
        retmax: String(query.limit),
        sort: "relevance",
      };
      if (query.fromYear !== null || query.toYear !== null) {
        params.datetype = "pdat";
        params.mindate = String(query.fromYear ?? 1800);
        params.maxdate = String(query.toYear ?? 3000);
      }
      const answer = jsonObject(
        await get("esearch.fcgi", params, "application/json", context),
        "PubMed",
      );
      const found = isRecord(answer.esearchresult) ? answer.esearchresult.idlist : undefined;
      const ids = Array.isArray(found)
        ? found.filter((id): id is string => typeof id === "string")
        : [];
      return summaries(ids, context);
    },
    async resolve(identifier: WorkIdentifier, context) {
      if (identifier.kind !== "pmid") return null;
      return (await summaries([identifier.id], context))[0] ?? null;
    },
    async fetchFullText(location, context) {
      const pmcid = /PMC\d+/.exec(location.url)?.[0];
      if (!pmcid)
        throw new SourceError("not-open", `${location.url} isn't a PubMed Central article.`);
      const xml = await context.http.text(
        `${baseUrl}/entrez/eutils/efetch.fcgi?${new URLSearchParams({ db: "pmc", id: pmcid, retmode: "xml", tool: TOOL })}`,
        {
          lane: "ncbi",
          intervalMs: terms.intervalMs,
          accept: "application/xml",
          signal: context.signal,
          timeoutMs: 60_000,
          maxBytes: 50 * 1024 * 1024,
        },
      );
      const article = readJats(xml);
      return {
        format: "sections",
        title: article.title,
        sections: article.sections,
        url: location.url,
        licence: article.licence,
        versionDate: article.versionDate,
      };
    },
  };
}
