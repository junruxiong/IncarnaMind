/**
 * ClinicalTrials.gov (API v2, https://clinicaltrials.gov/data-api/api): a
 * search by condition, drug or words, and one study's record by its NCT
 * number, which becomes a Document with each part of the record as a section.
 * Its operator's page states no limit that could be read (third-party guides
 * say about 50 requests a minute from one address), so IncarnaMind keeps to
 * one request a second and backs off on a 429.
 */
import { isRecord } from "../../errors";
import { SourceError } from "../http";
import { readTrial } from "../trial";
import type { FullText, ProviderContext, ProviderWork, SourceProvider } from "../types";
import { jsonObject, text } from "./common";

export const CLINICALTRIALS_URL = "https://clinicaltrials.gov/api/v2";

const INTERVAL_MS = 1000;

const NCT = /NCT\d{8}/i;

/** What a search result shows of a study: enough to choose by. */
const SEARCH_FIELDS = "NCTId,BriefTitle,OverallStatus,LastUpdatePostDate,LeadSponsorName,Phase";

export function createClinicalTrials(baseUrl: string = CLINICALTRIALS_URL): SourceProvider {
  const terms = { url: "https://clinicaltrials.gov/data-api/about-api", intervalMs: INTERVAL_MS };

  const get = (path: string, params: Record<string, string>, context: ProviderContext) =>
    context.http.text(`${baseUrl}/${path}?${new URLSearchParams({ ...params, format: "json" })}`, {
      lane: "clinicaltrials",
      intervalMs: terms.intervalMs,
      accept: "application/json",
      signal: context.signal,
      timeoutMs: 60_000,
      maxBytes: 50 * 1024 * 1024,
    });

  function fromStudy(study: Record<string, unknown>): ProviderWork | null {
    const protocol = isRecord(study.protocolSection) ? study.protocolSection : {};
    const id = isRecord(protocol.identificationModule) ? protocol.identificationModule : {};
    const status = isRecord(protocol.statusModule) ? protocol.statusModule : {};
    const sponsors = isRecord(protocol.sponsorCollaboratorsModule)
      ? protocol.sponsorCollaboratorsModule
      : {};
    const nct = text(id.nctId).toUpperCase();
    const title = text(id.briefTitle);
    if (!nct || !title) return null;
    const sponsor = text(isRecord(sponsors.leadSponsor) ? sponsors.leadSponsor.name : "");
    const updated = text(
      isRecord(status.lastUpdatePostDateStruct) ? status.lastUpdatePostDateStruct.date : "",
    );
    return {
      ids: { nct },
      fields: {
        type: "document",
        title,
        authors: sponsor ? [{ literal: sponsor }] : [],
        ...(updated && { issued: updated }),
        containerTitle: "ClinicalTrials.gov",
        publisher: "U.S. National Library of Medicine",
        url: `https://clinicaltrials.gov/study/${nct}`,
        language: "en",
      },
      citedBy: null,
      versionDate: updated || null,
      fullText: [
        {
          format: "sections",
          url: `https://clinicaltrials.gov/study/${nct}`,
          provider: "clinicaltrials",
          version: "published",
          licence: null,
          versionDate: updated || null,
        },
      ],
      landingUrl: `https://clinicaltrials.gov/study/${nct}`,
    };
  }

  return {
    id: "clinicaltrials",
    name: "ClinicalTrials.gov",
    service: { id: "https://clinicaltrials.gov", name: "ClinicalTrials.gov" },
    terms,
    resolves: ["nct"],
    searchedByDefault: false,
    async search(query, context) {
      const answer = jsonObject(
        await get(
          "studies",
          { "query.term": query.text, pageSize: String(query.limit), fields: SEARCH_FIELDS },
          context,
        ),
        "ClinicalTrials.gov",
      );
      return (Array.isArray(answer.studies) ? answer.studies : [])
        .filter(isRecord)
        .map(fromStudy)
        .filter((work): work is ProviderWork => work !== null);
    },
    async resolve(identifier, context) {
      if (identifier.kind !== "nct") return null;
      try {
        const study = jsonObject(
          await get(`studies/${identifier.id}`, { fields: SEARCH_FIELDS }, context),
          "ClinicalTrials.gov",
        );
        return fromStudy(study);
      } catch (error) {
        if (error instanceof SourceError && error.reason === "not-found") return null;
        throw error;
      }
    },
    async fetchFullText(location, context): Promise<FullText> {
      const nct = NCT.exec(location.url)?.[0]?.toUpperCase();
      if (!nct) throw new SourceError("not-open", `${location.url} isn't a study record.`);
      const trial = readTrial(
        jsonObject(await get(`studies/${nct}`, {}, context), "ClinicalTrials.gov"),
      );
      if (!trial) throw new SourceError("service-error", "The study record has no title.");
      return {
        format: "sections",
        title: `${trial.nct}: ${trial.title}`,
        sections: trial.sections,
        url: location.url,
        licence: null,
        versionDate: trial.updated,
      };
    },
  };
}
