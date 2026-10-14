/**
 * GovInfo (https://www.govinfo.gov), the Government Publishing Office's
 * official publications: here a section of the United States Code ("42
 * U.S.C. § 1983") or of the Code of Federal Regulations' annual edition ("17
 * CFR 240.10b-5"), looked up by its citation. Its API
 * (https://api.govinfo.gov/docs/) needs an api.data.gov key, which is the
 * User's own: kept through `secrets.ts`, sent in an `X-Api-Key` header and
 * never in an address, and never sent to its link service. Without one,
 * GovInfo is off. A key allows 1,000 requests an hour; past that the API
 * answers 429 for up to an hour, so a 429 stops the lookup and says so. Its
 * texts are works of the US Government, with no copyright (17 U.S.C. § 105,
 * https://www.govinfo.gov/about/policies).
 *
 * A citation is turned into GovInfo's granule by its link service, whose
 * redirect names it (that request carries no key), then the granule's summary
 * and text come from the API: three requests for a section, the text only
 * when the User adds it.
 */

import { readCfr } from "../cfr";
import { parseCfr, parseUsc } from "../citations";
import { SourceError } from "../http";
import type { FullText, ProviderContext, ProviderWork, SourceProvider } from "../types";
import { readUsCode } from "../uscode";
import { aboutText, jsonObject, plainTitle, remembered, text } from "./common";

export const GOVINFO_URLS = { api: "https://api.govinfo.gov", site: "https://www.govinfo.gov" };

const INTERVAL_MS = 500;

/** Where a User gets the key GovInfo needs: the page its own errors name. */
export const GOVINFO_SIGN_UP = "https://www.govinfo.gov/api-signup";

/** "/content/pkg/USCODE-2024-title42/html/USCODE-2024-title42-chap21-subchapI-sec1983.htm" */
const CONTENT = /\/content\/pkg\/([A-Za-z0-9-]+)\/(?:html|xml|pdf)\/([A-Za-z0-9-]+)\.[a-z]+$/;

/** "/packages/USCODE-2024-title42/granules/USCODE-2024-title42-chap21-subchapI-sec1983/htm" */
const GRANULE = /\/packages\/([A-Za-z0-9-]+)\/granules\/([A-Za-z0-9-]+)\/(htm|xml)$/;

const detailsPage = (pkg: string, granule: string) =>
  `https://www.govinfo.gov/app/details/${pkg}/${granule}`;

/** "USCODE-2024-title42" → 2024 and 42; "CFR-2025-title17-vol4" → 2025 and 17. */
function packageParts(pkg: string): { year: string; title: number } | null {
  const match = /^(?:USCODE|CFR)-(\d{4})-title(\d+)/.exec(pkg);
  return match?.[1] && match[2] ? { year: match[1], title: Number(match[2]) } : null;
}

export function createGovinfo(
  urls: { api: string; site: string } = GOVINFO_URLS,
  key: () => Promise<string | null> = async () => null,
): SourceProvider {
  const terms = { url: "https://api.govinfo.gov/docs/", intervalMs: INTERVAL_MS };
  const fetched = remembered<FullText>();

  /** A request to the API, with the User's key in a header. */
  async function api(path: string, accept: string, context: ProviderContext, large = false) {
    const apiKey = await key();
    if (!apiKey) throw new SourceError("needs-key", "GovInfo needs an api.data.gov key.");
    try {
      return await context.http.text(`${urls.api}${path}`, {
        lane: "govinfo",
        intervalMs: terms.intervalMs,
        accept,
        signal: context.signal,
        headers: { "X-Api-Key": apiKey },
        ...(large && { timeoutMs: 60_000, maxBytes: 50 * 1024 * 1024 }),
      });
    } catch (error) {
      // It answers 401 API_KEY_INVALID to a key it doesn't know; other api.data.gov services answer 403.
      if (error instanceof SourceError && (error.status === 401 || error.status === 403)) {
        throw new SourceError("key-rejected", "GovInfo didn't accept the key.", error.status);
      }
      throw error;
    }
  }

  /** The package and granule the link service redirects a citation to; null for one it doesn't know. */
  async function granuleOf(path: string, context: ProviderContext) {
    try {
      const { url } = await context.http.get(`${urls.site}/link/${path}`, {
        lane: "govinfo",
        intervalMs: terms.intervalMs,
        accept: "text/html",
        signal: context.signal,
        redirect: "manual",
      });
      const match = CONTENT.exec(URL.parse(url)?.pathname ?? "");
      return match?.[1] && match[2] ? { pkg: match[1], granule: match[2] } : null;
    } catch (error) {
      // It answers 400 for a section it doesn't have.
      if (error instanceof SourceError && (error.status === 400 || error.status === 404)) {
        return null;
      }
      throw error;
    }
  }

  return {
    id: "govinfo",
    name: "GovInfo",
    service: { id: "https://www.govinfo.gov", name: "GovInfo" },
    terms,
    resolves: ["usc", "cfr"],
    searchedByDefault: false,
    takesKey: true,
    needsKey: { signUp: GOVINFO_SIGN_UP },
    async resolve(identifier, context) {
      let found: { pkg: string; granule: string } | null = null;
      let cited = "";
      if (identifier.kind === "usc") {
        const usc = parseUsc(identifier.id);
        if (!usc) return null;
        const params = new URLSearchParams({ "link-type": "html", year: "mostrecent" });
        found = await granuleOf(`uscode/${usc.title}/${usc.section}?${params}`, context);
        cited = `${usc.title} U.S.C. § ${usc.section}`;
      } else if (identifier.kind === "cfr") {
        const cfr = parseCfr(identifier.id);
        // The annual edition by section; a whole part is the eCFR's.
        if (!cfr?.section) return null;
        const params = new URLSearchParams({
          sectionnum: cfr.section.slice(cfr.part.length + 1),
          year: cfr.date?.slice(0, 4) ?? "mostrecent",
          "link-type": "xml",
        });
        found = await granuleOf(`cfr/${cfr.title}/${cfr.part}?${params}`, context);
        cited = `${cfr.title} CFR ${cfr.section}`;
      }
      if (!found) return null;
      const { pkg, granule } = found;
      const summary = jsonObject(
        await api(`/packages/${pkg}/granules/${granule}/summary`, "application/json", context),
        "GovInfo",
      );
      const usc = identifier.kind === "usc";
      const edition = packageParts(pkg);
      const issued = text(summary.dateIssued) || null;
      const subject = plainTitle(text(summary.title));
      const page = detailsPage(pkg, granule);
      return {
        ids: { govinfo: granule },
        fields: {
          type: "document",
          title: `${cited} ${subject}`.trim(),
          authors: [],
          ...(issued && { issued }),
          containerTitle: usc
            ? `United States Code${edition ? `, ${edition.year} edition` : ""}`
            : `Code of Federal Regulations${edition ? `, annual edition ${edition.year}` : ""}`,
          publisher: "U.S. Government Publishing Office",
          ...(edition && { volume: String(edition.title) }),
          url: page,
          language: "en",
        },
        citedBy: null,
        versionDate: issued,
        fullText: [
          {
            format: "sections",
            url: `${urls.api}/packages/${pkg}/granules/${granule}/${usc ? "htm" : "xml"}`,
            provider: "govinfo",
            version: "published",
            licence: "public-domain",
            versionDate: issued,
          },
        ],
        landingUrl: page,
      } satisfies ProviderWork;
    },
    async fetchFullText(location, context): Promise<FullText> {
      const match = GRANULE.exec(URL.parse(location.url)?.pathname ?? "");
      const pkg = match?.[1];
      const granule = match?.[2];
      const edition = pkg ? packageParts(pkg) : null;
      if (!match || !pkg || !granule || !edition) {
        throw new SourceError("not-open", `${location.url} isn't a GovInfo text.`);
      }
      return fetched(location.url, async () => {
        const body = await api(
          `/packages/${pkg}/granules/${granule}/${match[3]}`,
          match[3] === "htm" ? "text/html" : "application/xml",
          context,
          true,
        );
        const page = detailsPage(pkg, granule);
        if (match[3] === "htm") {
          const code = readUsCode(body);
          const through = code.currentThrough ? `, current through ${code.currentThrough}` : "";
          return {
            format: "sections",
            title: `${edition.title} U.S.C. § ${code.heading}`,
            sections: [
              aboutText([
                `${edition.title} U.S.C. § ${code.section}, from the United States Code, ${edition.year} edition${through}, published by the U.S. Government Publishing Office on GovInfo.`,
              ]),
              ...code.sections,
            ],
            url: page,
            licence: "public-domain",
            versionDate: code.currentThrough ?? location.versionDate,
          };
        }
        const cfr = readCfr(body);
        const revised = /<DATE>(\d{4}-\d{2}-\d{2})<\/DATE>/.exec(body)?.[1] ?? location.versionDate;
        const cite = cfr.heading.split(" ")[0] ?? "";
        return {
          format: "sections",
          title: `${edition.title} CFR ${cfr.heading}`,
          sections: [
            aboutText([
              `${edition.title} CFR ${cite}, from the Code of Federal Regulations, annual edition${revised ? ` revised as of ${revised}` : ` ${edition.year}`}, published by the U.S. Government Publishing Office on GovInfo.`,
            ]),
            ...cfr.sections,
          ],
          url: page,
          licence: "public-domain",
          versionDate: revised,
        };
      });
    },
  };
}
