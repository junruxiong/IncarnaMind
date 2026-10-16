/** The public pool, one request a second: the polite pool needs a contact address, which IncarnaMind never invents. */
import { isRecord } from "../errors";
import { fromCsl } from "../referenceRecords/csl";
import { SourceError, type SourceHttp } from "../sources/http";
import type { WorkRecord } from "./match";

export const CROSSREF_URL = "https://api.crossref.org";
/** Crossref's public pool: one request a second, on the lane Find sources uses too. */
const LANE = { lane: "crossref", intervalMs: 1000 } as const;
const ROWS = 5;
/** The fields a search asks for: enough to compare, nothing more. */
const SELECT = "DOI,title,author,issued,published-print,published-online,container-title,type";

/** The years in a Crossref date ("issued": {"date-parts": [[2015, 5, 28]]}). */
function yearOf(value: unknown): number | null {
  if (!isRecord(value) || !Array.isArray(value["date-parts"])) return null;
  const first: unknown = value["date-parts"][0];
  const year: unknown = Array.isArray(first) ? first[0] : null;
  return typeof year === "number" && Number.isInteger(year) ? year : null;
}

/** A Crossref work as a check reads it; null without a title. */
export function workRecord(item: unknown): WorkRecord | null {
  if (!isRecord(item)) return null;
  const fields = fromCsl(item);
  if (!fields.title) return null;
  const years = [item.issued, item["published-print"], item["published-online"]]
    .map(yearOf)
    .filter((year): year is number => year !== null);
  return { fields, years: [...new Set(years)], preprint: item.type === "posted-content" };
}

/** The "message" of a Crossref answer. */
function messageOf(body: string): unknown {
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    throw new SourceError("service-error", "Crossref's answer can't be read.");
  }
  return isRecord(parsed) ? parsed.message : undefined;
}

export interface CrossrefClient {
  /** The work a DOI names; null when Crossref has none. */
  byDoi(doi: string, signal: AbortSignal): Promise<WorkRecord | null>;
  /**
   * The agency that registered a DOI ("crossref", "datacite", …); null when
   * none did, so the DOI leads nowhere.
   */
  agency(doi: string, signal: AbortSignal): Promise<string | null>;
  /** The works whose records best match a reference as written, best first. */
  search(reference: string, signal: AbortSignal): Promise<WorkRecord[]>;
}

export function createCrossrefClient(http: SourceHttp, baseUrl = CROSSREF_URL): CrossrefClient {
  const get = async (path: string, signal: AbortSignal) =>
    messageOf(
      await http.text(`${baseUrl}${path}`, { ...LANE, accept: "application/json", signal }),
    );
  /** The answer at a DOI's `path`; null when Crossref has nothing there. */
  const ofDoi = async (path: string, signal: AbortSignal): Promise<unknown> => {
    try {
      return await get(path, signal);
    } catch (error) {
      if (error instanceof SourceError && error.reason === "not-found") return null;
      throw error;
    }
  };
  return {
    async byDoi(doi, signal) {
      return workRecord(await ofDoi(`/works/${encodeURIComponent(doi)}`, signal));
    },

    async agency(doi, signal) {
      const message = await ofDoi(`/works/${encodeURIComponent(doi)}/agency`, signal);
      const agency = isRecord(message) && isRecord(message.agency) ? message.agency.id : null;
      return typeof agency === "string" ? agency : null;
    },

    async search(reference, signal) {
      const query = new URLSearchParams({
        "query.bibliographic": reference.slice(0, 1000),
        rows: String(ROWS),
        select: SELECT,
      });
      const message = await get(`/works?${query}`, signal);
      const items = isRecord(message) && Array.isArray(message.items) ? message.items : [];
      return items.map(workRecord).filter((record): record is WorkRecord => record !== null);
    },
  };
}
