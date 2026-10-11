/**
 * DailyMed (https://dailymed.nlm.nih.gov), the National Library of Medicine's
 * database of the labels (SPL) drug makers file with the FDA: its web services
 * (https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm) search
 * by drug name or set id and give one label as XML, with its sections and
 * their numbers as the label has them. Its page gives no limit; NLM's general
 * API terms ask for no more than 20 requests a second, so IncarnaMind waits
 * half a second between requests. A search finds a label, and adding it
 * reads the whole label, one request.
 */
import { isRecord } from "../../errors";
import { SourceError } from "../http";
import { readSpl } from "../spl";
import type { FullText, ProviderContext, ProviderWork, SourceProvider } from "../types";
import { aboutLabel, companyName, jsonObject, sentenceCase, text } from "./common";

export const DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2";

const INTERVAL_MS = 500;

/** DailyMed answers 406 to "Accept: application/xml" alone, though it serves XML for a `.xml` address. */
const XML_ACCEPT = "application/xml, */*;q=0.5";

const SET_ID = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "Oct 08, 2026" as "2026-10-08"; null for anything else. */
export function isoDate(written: string): string | null {
  const match = /^([A-Za-z]{3}) (\d{1,2}), (\d{4})$/.exec(written.trim());
  const month = MONTHS.indexOf(match?.[1] ?? "");
  if (!match || month < 0) return null;
  return `${match[3]}-${String(month + 1).padStart(2, "0")}-${(match[2] ?? "").padStart(2, "0")}`;
}

/** "METFORMIN HYDROCHLORIDE SOLUTION [VISTAPHARM, LLC]": the product, and who files it. */
export function splitTitle(title: string): { product: string; labeler: string } {
  const match = /^(.*?)\s*\[([^\]]*)\]\s*$/s.exec(title.trim());
  return { product: sentenceCase(match?.[1] ?? title), labeler: companyName(match?.[2] ?? "") };
}

export function createDailymed(baseUrl: string = DAILYMED_URL): SourceProvider {
  const terms = {
    url: "https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm",
    intervalMs: INTERVAL_MS,
  };
  const get = (path: string, accept: string, context: ProviderContext, timeoutMs?: number) =>
    context.http.text(`${baseUrl}/${path}`, {
      lane: "dailymed",
      intervalMs: terms.intervalMs,
      accept,
      signal: context.signal,
      ...(timeoutMs && { timeoutMs, maxBytes: 50 * 1024 * 1024 }),
    });

  async function list(params: Record<string, string>, context: ProviderContext) {
    const answer = jsonObject(
      await get(`spls.json?${new URLSearchParams(params)}`, "application/json", context),
      "DailyMed",
    );
    return (Array.isArray(answer.data) ? answer.data : [])
      .filter(isRecord)
      .map((entry): ProviderWork | null => {
        const setId = text(entry.setid).toLowerCase();
        const { product, labeler } = splitTitle(text(entry.title));
        if (!setId || !product) return null;
        const published = isoDate(text(entry.published_date));
        return {
          ids: { setid: setId },
          fields: {
            type: "document",
            title: product,
            authors: labeler ? [{ literal: labeler }] : [],
            ...(published && { issued: published }),
            containerTitle: "DailyMed",
            publisher: "National Library of Medicine",
            language: "en",
          },
          citedBy: null,
          versionDate: published,
          fullText: [
            {
              format: "sections",
              url: `${baseUrl}/spls/${setId}.xml`,
              provider: "dailymed",
              version: "published",
              licence: null,
              versionDate: published,
            },
          ],
          landingUrl: `https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=${setId}`,
        };
      })
      .filter((work): work is ProviderWork => work !== null);
  }

  return {
    id: "dailymed",
    name: "DailyMed",
    service: { id: "https://dailymed.nlm.nih.gov", name: "DailyMed" },
    terms,
    resolves: ["setid"],
    searchedByDefault: false,
    async search(query, context) {
      const name = query.text.replace(/\s+/g, " ").trim();
      return name ? list({ drug_name: name, pagesize: String(query.limit) }, context) : [];
    },
    async resolve(identifier, context) {
      if (identifier.kind !== "setid") return null;
      return (await list({ setid: identifier.id }, context))[0] ?? null;
    },
    async fetchFullText(location, context): Promise<FullText> {
      const setId = SET_ID.exec(location.url)?.[0];
      if (!setId) throw new SourceError("not-open", `${location.url} isn't a DailyMed label.`);
      const label = readSpl(
        await get(`spls/${setId.toLowerCase()}.xml`, XML_ACCEPT, context, 60_000),
      );
      return {
        format: "sections",
        title: label.title,
        sections: [
          aboutLabel({
            source: "Drug label from DailyMed",
            setId: label.setId || setId,
            version: label.version,
            effective: label.effective,
            notice: null,
          }),
          ...label.sections,
        ],
        url: location.url,
        licence: null,
        versionDate: label.effective,
      };
    },
  };
}
