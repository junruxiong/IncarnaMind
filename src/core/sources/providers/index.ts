/**
 * The Source providers IncarnaMind ships, in the order they are asked and
 * listed. A new provider (#303–#307) is a module beside these, added here,
 * with its strings (`sources.provider.<id>.about`) and its ADR-0018 entry.
 */
import type { SourceProvider } from "../types";
import { ARXIV_URLS, createArxiv } from "./arxiv";
import { createClinicalTrials } from "./clinicaltrials";
import { createCrossref } from "./crossref";
import { createDailymed } from "./dailymed";
import { createEcfr } from "./ecfr";
import { createGovinfo, GOVINFO_URLS } from "./govinfo";
import { createOpenAlex } from "./openalex";
import { createOpenfda } from "./openfda";
import { createPubmed } from "./pubmed";

/** Where each provider is asked, by its id: the real services unless a test's stand-in is given. */
export type SourceProviderUrls = Partial<
  Record<
    | "openalex"
    | "arxiv"
    | "pubmed"
    | "crossref"
    | "openfda"
    | "dailymed"
    | "clinicaltrials"
    | "ecfr"
    | "govinfo",
    string
  >
>;

/** The keychain secret holding the User's own key for a provider that takes one. */
export const sourceKeySecret = (providerId: string) => `sources.${providerId}.key`;

export function createSourceProviders(
  urls: SourceProviderUrls = {},
  readKey: (providerId: string) => Promise<string | null> = async () => null,
): SourceProvider[] {
  return [
    createOpenAlex(urls.openalex),
    // A stand-in serves arXiv's API and its files from one address.
    createArxiv(urls.arxiv ? { api: urls.arxiv, files: urls.arxiv } : ARXIV_URLS),
    createPubmed(urls.pubmed),
    createCrossref(urls.crossref),
    createOpenfda(urls.openfda, () => readKey("openfda")),
    createDailymed(urls.dailymed),
    createClinicalTrials(urls.clinicaltrials),
    createEcfr(urls.ecfr),
    // A stand-in serves GovInfo's API and its link service from one address.
    createGovinfo(urls.govinfo ? { api: urls.govinfo, site: urls.govinfo } : GOVINFO_URLS, () =>
      readKey("govinfo"),
    ),
  ];
}
