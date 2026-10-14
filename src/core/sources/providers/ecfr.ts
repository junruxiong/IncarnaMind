/**
 * The eCFR (https://www.ecfr.gov): the Code of Federal Regulations as amended
 * to date, by the Office of the Federal Register and GPO, from its search and
 * versioner APIs, which need no key. Its operators call it "an unofficial
 * editorial compilation of CFR material and Federal Register amendments … not
 * an official legal edition of the CFR"
 * (https://www.archives.gov/federal-register/cfr/about-ecfr), so its results
 * and its Documents say so, and GovInfo (./govinfo) gives the official annual
 * edition. Its versioner answers 406 to a request that doesn't accept a
 * compressed answer. Its developer pages turn automated readers away, so no
 * limit could be read: IncarnaMind keeps to one request a second.
 *
 * Words search the regulations in force today; a citation ("17 CFR
 * 240.10b-5", "17 CFR Part 243", "… as of 2019-06-30") looks a section or a
 * part up, as in force today or on that day (the eCFR's history starts in
 * January 2017). A section removed since is found with its last text.
 */
import { isRecord } from "../../errors";
import { readCfr } from "../cfr";
import { type CfrCitation, cfrId, parseCfr } from "../citations";
import { SourceError } from "../http";
import type { FullText, ProviderContext, ProviderWork, SourceProvider } from "../types";
import { aboutText, jsonObject, plainTitle, remembered, text } from "./common";

export const ECFR_URL = "https://www.ecfr.gov";

const INTERVAL_MS = 1000;

/** The versioner refuses an answer it may not compress. */
const COMPRESSED = { "Accept-Encoding": "gzip, deflate, br" };

/** How long the titles' "up to date as of" days are kept: they move on once a day. */
const TITLES_TTL_MS = 60 * 60 * 1000;

export const ECFR_UNOFFICIAL =
  "The eCFR is an editorial compilation of CFR material and Federal Register amendments, not an official legal edition of the CFR.";

/** The day before an ISO date. */
function dayBefore(date: string): string {
  return new Date(Date.parse(`${date}T00:00:00Z`) - 86_400_000).toISOString().slice(0, 10);
}

/** Its page on the eCFR, as in force today or on a day. */
export function ecfrPage(citation: CfrCitation, date: string | null): string {
  const at = date ? `on/${date}` : "current";
  const where = citation.section ? `section-${citation.section}` : `part-${citation.part}`;
  return `https://www.ecfr.gov/${at}/title-${citation.title}/${where}`;
}

export function createEcfr(baseUrl: string = ECFR_URL): SourceProvider {
  const terms = {
    url: "https://www.ecfr.gov/developers/documentation/api/v1",
    intervalMs: INTERVAL_MS,
  };
  const fetched = remembered<FullText>();
  let titles: { at: number; days: Promise<Map<number, string>> } | null = null;

  const get = (path: string, context: ProviderContext, large = false) =>
    context.http.text(`${baseUrl}${path}`, {
      lane: "ecfr",
      intervalMs: terms.intervalMs,
      accept: path.includes(".xml") ? "application/xml" : "application/json",
      signal: context.signal,
      headers: COMPRESSED,
      ...(large && { timeoutMs: 60_000, maxBytes: 50 * 1024 * 1024 }),
    });

  /** The day each title is up to date as of: what "in force today" reads. */
  async function upToDate(title: number, context: ProviderContext): Promise<string | null> {
    if (!titles || Date.now() - titles.at > TITLES_TTL_MS) {
      const days = get("/api/versioner/v1/titles.json", context).then((body) => {
        const answer = jsonObject(body, "The eCFR");
        const list = Array.isArray(answer.titles) ? answer.titles.filter(isRecord) : [];
        return new Map(
          list.flatMap((each) =>
            typeof each.number === "number" && each.reserved !== true && text(each.up_to_date_as_of)
              ? [[each.number, text(each.up_to_date_as_of)] as const]
              : [],
          ),
        );
      });
      titles = { at: Date.now(), days };
      days.catch(() => {
        titles = null;
      });
    }
    return (await titles.days).get(title) ?? null;
  }

  /** The versioner's address for the text, kept as the copy's address: "current" is read when it is fetched. */
  const copyUrl = (citation: CfrCitation, date: string | null, removed: string | null = null) => {
    const params = new URLSearchParams({ part: citation.part });
    if (citation.section) params.set("section", citation.section);
    if (removed) params.set("removed", removed);
    return `${baseUrl}/api/versioner/v1/full/${date ?? "current"}/title-${citation.title}.xml?${params}`;
  };

  function work(
    citation: CfrCitation,
    heading: string,
    agency: string,
    date: string | null,
    removed: string | null = null,
  ): ProviderWork {
    const page = ecfrPage(citation, citation.date);
    const title = `${citation.title} CFR ${heading}${removed ? ` (removed ${removed})` : ""}`;
    return {
      ids: { cfr: cfrId(citation) },
      fields: {
        type: "document",
        title,
        authors: agency ? [{ literal: agency }] : [],
        ...(date && { issued: date }),
        containerTitle: "Code of Federal Regulations (eCFR)",
        publisher: "Office of the Federal Register",
        volume: String(citation.title),
        url: page,
        language: "en",
      },
      citedBy: null,
      versionDate: date,
      fullText: [
        {
          format: "sections",
          url: copyUrl(citation, date, removed),
          provider: "ecfr",
          version: "published",
          licence: "public-domain",
          versionDate: date,
        },
      ],
      landingUrl: page,
      unofficial: true,
    };
  }

  /** A section the eCFR no longer has on a day: removed, with the text it had the day before; null if it never had it. */
  async function removedSection(
    citation: CfrCitation,
    date: string,
    context: ProviderContext,
  ): Promise<ProviderWork | null> {
    if (!citation.section) return null;
    const params = new URLSearchParams({ section: citation.section });
    const answer = jsonObject(
      await get(`/api/versioner/v1/versions/title-${citation.title}.json?${params}`, context),
      "The eCFR",
    );
    const versions = (Array.isArray(answer.content_versions) ? answer.content_versions : [])
      .filter(isRecord)
      .filter((each) => text(each.date) !== "" && text(each.date) <= date)
      .sort((a, b) => text(a.date).localeCompare(text(b.date)));
    const last = versions.at(-1);
    if (last?.removed !== true || versions.length < 2) return null;
    const removed = text(last.date);
    const inForce = dayBefore(removed);
    const heading = plainTitle(text(last.name).replace(/^§+\s*/, ""));
    return work({ ...citation, date: inForce }, heading, "", inForce, removed);
  }

  return {
    id: "ecfr",
    name: "eCFR",
    service: { id: "https://www.ecfr.gov", name: "eCFR" },
    terms,
    resolves: ["cfr"],
    searchedByDefault: false,
    async search(query, context) {
      const params = new URLSearchParams({
        query: query.text,
        per_page: String(query.limit),
        date: "current",
      });
      const answer = jsonObject(await get(`/api/search/v1/results?${params}`, context), "The eCFR");
      return (Array.isArray(answer.results) ? answer.results : [])
        .filter(isRecord)
        .flatMap((result): ProviderWork[] => {
          const where = isRecord(result.hierarchy) ? result.hierarchy : {};
          const headings = isRecord(result.headings) ? result.headings : {};
          const title = Number(text(where.title));
          const part = text(where.part);
          const section = text(where.section);
          if (
            result.type !== "Section" ||
            result.removed === true ||
            result.reserved === true ||
            !Number.isInteger(title) ||
            !part ||
            !section
          ) {
            return [];
          }
          const subject = plainTitle(text(headings.section));
          return [
            work(
              { title, part, section, date: null },
              `${section} ${subject}`.trim(),
              plainTitle(text(headings.chapter)),
              null,
            ),
          ];
        });
    },
    async resolve(identifier, context) {
      if (identifier.kind !== "cfr") return null;
      const asked = parseCfr(identifier.id);
      if (!asked) return null;
      const today = await upToDate(asked.title, context);
      if (!today) return null;
      // A day the eCFR hasn't reached yet reads as today.
      const citation = asked.date && asked.date < today ? asked : { ...asked, date: null };
      const date = citation.date ?? today;
      const params = new URLSearchParams({ part: citation.part });
      if (citation.section) params.set("section", citation.section);
      let body: string;
      try {
        body = await get(
          `/api/versioner/v1/ancestry/${date}/title-${citation.title}.json?${params}`,
          context,
        );
      } catch (error) {
        if (error instanceof SourceError && error.reason === "not-found") {
          return removedSection(citation, date, context);
        }
        throw error;
      }
      const ancestors = jsonObject(body, "The eCFR").ancestors;
      const nodes = (Array.isArray(ancestors) ? ancestors : []).filter(isRecord);
      const target = nodes.at(-1);
      if (!target || target.reserved === true) return null;
      const agency = text(nodes.find((each) => each.type === "chapter")?.label_description);
      const heading = plainTitle(text(target.label).replace(/^§+\s*/, ""));
      return work(citation, heading, agency, date);
    },
    async fetchFullText(location, context): Promise<FullText> {
      const url = URL.parse(location.url);
      const match = /\/full\/([^/]+)\/title-(\d+)\.xml$/.exec(url?.pathname ?? "");
      const part = url?.searchParams.get("part");
      if (!url || !match?.[1] || !match[2] || !part) {
        throw new SourceError("not-open", `${location.url} isn't an eCFR text.`);
      }
      const title = Number(match[2]);
      const section = url.searchParams.get("section");
      const removed = url.searchParams.get("removed");
      return fetched(location.url, async () => {
        const date = match[1] === "current" ? await upToDate(title, context) : (match[1] ?? null);
        if (!date) throw new SourceError("not-found", `The eCFR has no title ${title}.`);
        const params = new URLSearchParams({ part });
        if (section) params.set("section", section);
        const cfr = readCfr(
          await get(`/api/versioner/v1/full/${date}/title-${title}.xml?${params}`, context, true),
        );
        const citation: CfrCitation = { title, part, section, date };
        const cited = `${title} CFR ${section ?? `Part ${part}`}`;
        return {
          format: "sections",
          title: `${title} CFR ${cfr.heading}`,
          sections: [
            aboutText([
              `${cited}, as in force on ${date}, from the eCFR (www.ecfr.gov).`,
              ...(removed
                ? [
                    `This section was removed on ${removed}. This is its text as in force on ${date}.`,
                  ]
                : []),
              ECFR_UNOFFICIAL,
            ]),
            ...cfr.sections,
          ],
          url: ecfrPage(citation, date),
          licence: "public-domain",
          versionDate: date,
        };
      });
    },
  };
}
