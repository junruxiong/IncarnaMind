/**
 * Crossref (https://www.crossref.org): a work's record by its DOI, for a DOI
 * the User types. Its metadata is facts, free to reuse ("almost none of the
 * metadata is subject to copyright",
 * https://www.crossref.org/documentation/retrieve-metadata/rest-api/). Its
 * public pool allows five single-DOI requests a second, one at a time
 * (https://www.crossref.org/blog/announcing-changes-to-rest-api-rate-limits/);
 * its faster "polite" pool needs a contact address, which IncarnaMind never
 * invents for a User, so these stay in the public pool, one a second.
 * Crossref knows no open copies itself: OpenAlex, asked at the same time,
 * says where they are.
 */
import { isRecord } from "../../errors";
import { fromCsl } from "../../referenceRecords/csl";
import { SourceError } from "../http";
import type { ProviderWork, SourceProvider } from "../types";
import { jsonObject, optionalNumber, text } from "./common";

export const CROSSREF_URL = "https://api.crossref.org";

export function createCrossref(baseUrl: string = CROSSREF_URL): SourceProvider {
  const terms = {
    url: "https://www.crossref.org/documentation/retrieve-metadata/rest-api/",
    intervalMs: 1000,
  };
  return {
    id: "crossref",
    name: "Crossref",
    service: { id: "https://api.crossref.org", name: "Crossref" },
    terms,
    async resolve(identifier, context) {
      if (identifier.kind !== "doi") return null;
      let body: string;
      try {
        body = await context.http.text(`${baseUrl}/works/${encodeURIComponent(identifier.id)}`, {
          lane: "crossref",
          intervalMs: terms.intervalMs,
          accept: "application/json",
          signal: context.signal,
        });
      } catch (error) {
        if (error instanceof SourceError && error.reason === "not-found") return null;
        throw error;
      }
      const message = jsonObject(body, "Crossref").message;
      if (!isRecord(message))
        throw new SourceError("service-error", "Crossref's answer has no record.");
      const fields = fromCsl(message);
      if (!fields.title) return null;
      const doi = (fields.doi ?? identifier.id).toLowerCase();
      const work: ProviderWork = {
        ids: { doi },
        fields,
        citedBy: optionalNumber(message["is-referenced-by-count"]),
        versionDate: fields.issued ?? null,
        fullText: [],
        landingUrl: text(message.URL) || `https://doi.org/${doi}`,
      };
      return work;
    },
  };
}
