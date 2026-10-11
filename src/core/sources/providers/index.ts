/**
 * The Source providers IncarnaMind ships, in the order they are asked and
 * listed. A new provider (#303–#307) is a module beside these, added here,
 * with its strings (`sources.provider.<id>.about`) and its ADR-0018 entry.
 */
import type { SourceProvider } from "../types";
import { ARXIV_URLS, createArxiv } from "./arxiv";
import { createCrossref } from "./crossref";
import { createOpenAlex } from "./openalex";
import { createPubmed } from "./pubmed";

/** Where each provider is asked, by its id: the real services unless a test's stand-in is given. */
export type SourceProviderUrls = Partial<
  Record<"openalex" | "arxiv" | "pubmed" | "crossref", string>
>;

export function createSourceProviders(urls: SourceProviderUrls = {}): SourceProvider[] {
  return [
    createOpenAlex(urls.openalex),
    // A stand-in serves arXiv's API and its files from one address.
    createArxiv(urls.arxiv ? { api: urls.arxiv, files: urls.arxiv } : ARXIV_URLS),
    createPubmed(urls.pubmed),
    createCrossref(urls.crossref),
  ];
}
