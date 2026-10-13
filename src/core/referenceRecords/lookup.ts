/**
 * Look up (#286): a work's record by its DOI, from Crossref's REST API, or
 * by its arXiv id, from arXiv's API. Only the identifier is sent: in the
 * request's address, with no key, cookie or other data. The caller asks for
 * consent first (../consent). Both answer with metadata that may be reused
 * freely: Crossref's is facts ("almost none of the metadata is subject to
 * copyright", https://www.crossref.org/documentation/retrieve-metadata/rest-api/),
 * arXiv's is CC0 (https://info.arxiv.org/help/api/tou.html).
 */
import type { ExternalService, ReferenceFields, ReferenceLookupFailureReason } from "../api";
import { isRecord } from "../errors";
import { parseArxivFeed } from "../sources/providers/arxiv";
import { fromCsl } from "./csl";

export const CROSSREF: ExternalService = { id: "https://api.crossref.org", name: "Crossref" };
export const ARXIV: ExternalService = { id: "https://export.arxiv.org", name: "arXiv" };

/** How long a service may take to answer. */
const TIMEOUT_MS = 15_000;

/** Crossref asks callers to say who they are, so it can reach them about their traffic. */
const USER_AGENT = "IncarnaMind (https://github.com/IncarnaMind/IncarnaMind)";

export class LookupError extends Error {
  override name = "LookupError";
  constructor(
    readonly reason: Exclude<
      ReferenceLookupFailureReason,
      "no-identifier" | "local-mode" | "declined"
    >,
    message: string,
  ) {
    super(message);
  }
}

async function get(url: string, accept: string, signal: AbortSignal): Promise<string> {
  let response: Response;
  try {
    response = await fetch(url, {
      headers: { accept, "user-agent": USER_AGENT },
      signal: AbortSignal.any([signal, AbortSignal.timeout(TIMEOUT_MS)]),
    });
  } catch (error) {
    if (signal.aborted) throw signal.reason;
    const message = error instanceof Error ? error.message : String(error);
    throw new LookupError("unreachable", `Couldn't reach ${new URL(url).origin}: ${message}`);
  }
  if (response.status === 404) throw new LookupError("not-found", `${url} answered 404.`);
  if (!response.ok) {
    throw new LookupError("service-error", `${url} answered ${response.status}.`);
  }
  return response.text();
}

/** A work's record from Crossref, by its DOI. */
export async function lookUpDoi(
  baseUrl: string,
  doi: string,
  signal: AbortSignal,
): Promise<Partial<ReferenceFields>> {
  const body = await get(`${baseUrl}/works/${encodeURIComponent(doi)}`, "application/json", signal);
  let answer: unknown;
  try {
    answer = JSON.parse(body);
  } catch {
    throw new LookupError("service-error", "Crossref's answer isn't JSON.");
  }
  if (!isRecord(answer) || !isRecord(answer.message)) {
    throw new LookupError("service-error", "Crossref's answer has no record in it.");
  }
  return fromCsl(answer.message);
}

/** A preprint's record from arXiv, by its id: arXiv's API answers in Atom, read as Find sources reads it. */
export async function lookUpArxiv(
  baseUrl: string,
  id: string,
  signal: AbortSignal,
): Promise<Partial<ReferenceFields>> {
  const body = await get(
    `${baseUrl}/api/query?id_list=${encodeURIComponent(id)}&max_results=1`,
    "application/atom+xml",
    signal,
  );
  let works: ReturnType<typeof parseArxivFeed>;
  try {
    works = parseArxivFeed(body);
  } catch {
    throw new LookupError("service-error", "arXiv's answer isn't readable.");
  }
  // An id arXiv doesn't know comes back as an entry titled "Error", or as none.
  const work = works[0];
  if (!work) throw new LookupError("not-found", `arXiv has no record for ${id}.`);
  return work.fields;
}
