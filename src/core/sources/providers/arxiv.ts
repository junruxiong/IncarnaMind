/**
 * arXiv (https://arxiv.org): preprints in physics, mathematics, computer
 * science and more, every one open. Its API answers in Atom. Its terms ask
 * for "no more than one request every three seconds, and … a single
 * connection at a time", from all of one's machines together; its metadata
 * is CC0; and the content of e-prints may be retrieved and stored "for your
 * own personal use, or for research purposes", fetched from arXiv.org
 * itself (https://info.arxiv.org/help/api/tou.html). So one PDF is fetched
 * when the User adds a paper, on the same lane as the searches, and never in
 * bulk.
 */
import { ARXIV_ID, arxivUrl, parseIssued, parseName } from "../../../shared/references";
import type { ReferenceName } from "../../api";
import {
  descendants,
  elements,
  parseXml,
  textOf,
  type XmlElement,
} from "../../documents/formats/xml";
import { plain } from "../../referenceRecords/csl";
import { SourceError } from "../http";
import type { ProviderContext, ProviderWork, SourceProvider, SourceQuery } from "../types";
import { checkPdf } from "./common";

export const ARXIV_URLS = { api: "https://export.arxiv.org", files: "https://arxiv.org" };

/** The one lane every request to arXiv takes, Look up's (#286) too. */
export const ARXIV_LANE = "arxiv";

/** arXiv's "one request every three seconds". */
export const ARXIV_INTERVAL_MS = 3000;

const field = (entry: XmlElement, name: string) => {
  const element = elements(entry).find((each) => each.name === name);
  return element ? plain(textOf(element)) : "";
};

/** The works in an arXiv Atom feed; an entry arXiv titles "Error" (an unknown id) is none. */
export function parseArxivFeed(body: string, filesUrl: string = ARXIV_URLS.files): ProviderWork[] {
  let feed: XmlElement;
  try {
    feed = parseXml(body);
  } catch {
    throw new SourceError("service-error", "arXiv's answer isn't readable.");
  }
  const works: ProviderWork[] = [];
  for (const entry of descendants(feed, "entry")) {
    const title = field(entry, "title");
    const match = new RegExp(`abs/${ARXIV_ID.source}$`).exec(field(entry, "id"));
    if (!match?.[1] || title === "" || title === "Error") continue;
    const versioned = match[0].slice("abs/".length);
    const bare = match[1];
    const authors = elements(entry)
      .filter((each) => each.name === "author")
      .map((author) => parseName(plain(textOf(elements(author)[0] ?? author))))
      .filter((name): name is ReferenceName => name !== null);
    const published = parseIssued(field(entry, "published"));
    const updated = parseIssued(field(entry, "updated"));
    const journalDoi = field(entry, "arxiv:doi").toLowerCase();
    works.push({
      // The published version's DOI, when the authors gave one, so the two are one work.
      ids: { arxiv: bare, ...(journalDoi && { doi: journalDoi }) },
      fields: {
        type: "article",
        title,
        ...(authors.length > 0 && { authors }),
        ...(published && { issued: published }),
        publisher: "arXiv",
        // arXiv's own DOI, which DataCite registers for every paper.
        doi: `10.48550/arXiv.${bare}`,
        url: arxivUrl(bare),
      },
      citedBy: null,
      versionDate: updated ?? published,
      fullText: [
        {
          format: "pdf",
          url: `${filesUrl}/pdf/${versioned}`,
          provider: "arxiv",
          version: "submitted",
          licence: null,
          versionDate: updated ?? published,
        },
      ],
      landingUrl: arxivUrl(versioned),
    });
  }
  return works;
}

/** The words as arXiv's query: each one (or each quoted phrase) in any field, all of them. */
export function arxivQuery(query: SourceQuery): string {
  const terms = (query.text.match(/"[^"]+"|[^\s"]+/g) ?? [])
    .map((term) => term.replace(/"/g, "").trim())
    .filter(Boolean)
    .map((term) => `all:"${term}"`);
  if (query.fromYear !== null || query.toYear !== null) {
    terms.push(
      `submittedDate:[${query.fromYear ?? 1991}01010000 TO ${query.toYear ?? 9999}12312359]`,
    );
  }
  return terms.join(" AND ");
}

export function createArxiv(urls: { api: string; files: string } = ARXIV_URLS): SourceProvider {
  const terms = {
    url: "https://info.arxiv.org/help/api/tou.html",
    intervalMs: ARXIV_INTERVAL_MS,
  };
  const request = (accept: string, context: ProviderContext, timeoutMs?: number) => ({
    lane: ARXIV_LANE,
    intervalMs: terms.intervalMs,
    accept,
    signal: context.signal,
    ...(timeoutMs && { timeoutMs }),
  });
  return {
    id: "arxiv",
    resolves: ["arxiv"],
    name: "arXiv",
    service: { id: "https://export.arxiv.org", name: "arXiv" },
    terms,
    async search(query, context) {
      const params = new URLSearchParams({
        search_query: arxivQuery(query),
        start: "0",
        max_results: String(query.limit),
        sortBy: "relevance",
      });
      const body = await context.http.text(
        `${urls.api}/api/query?${params}`,
        request("application/atom+xml", context),
      );
      return parseArxivFeed(body, urls.files);
    },
    async resolve(identifier, context) {
      if (identifier.kind !== "arxiv") return null;
      const params = new URLSearchParams({ id_list: identifier.id, max_results: "1" });
      const body = await context.http.text(
        `${urls.api}/api/query?${params}`,
        request("application/atom+xml", context),
      );
      return parseArxivFeed(body, urls.files)[0] ?? null;
    },
    async fetchFullText(location, context) {
      // Only from arXiv itself, as its terms ask, whatever address another provider gave.
      const id = new RegExp(`(?:abs|pdf)/(${ARXIV_ID.source})`).exec(location.url)?.[1];
      if (!id) throw new SourceError("not-open", `${location.url} isn't an arXiv paper.`);
      const url = `${urls.files}/pdf/${id}`;
      const { bytes } = await context.http.get(url, {
        ...request("application/pdf", context, 60_000),
        maxBytes: 100 * 1024 * 1024,
      });
      checkPdf(bytes, url);
      return { format: "pdf", bytes, url };
    },
  };
}
