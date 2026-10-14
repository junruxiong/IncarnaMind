/**
 * Crossref's REST API for checking references (#711): a work's record by its
 * DOI, or the works whose records best match a reference as written
 * (`query.bibliographic`, Crossref's own reference matching). Only the
 * reference is sent, in the request's address. Its public pool takes one
 * request a second here, as Find sources asks it (../sources/providers/crossref):
 * the faster polite pool needs a contact address, which IncarnaMind never
 * invents for a User.
 */
import { isRecord } from "../errors";
import { fromCsl } from "../referenceRecords/csl";
import { SourceError, type SourceHttp } from "../sources/http";
import type { WorkRecord } from "./match";

export const CROSSREF_URL = "https://api.crossref.org";
/** Crossref's public pool: one request a second, on the lane Find sources uses too. */
const LANE = { lane: "crossref", intervalMs: 1000 } as const;
/** How many of the best matches a search reads. */
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
  return { fields, years: [...new Set(years)] };
}

const parse = (body: string): unknown => {
  try {
    return JSON.parse(body);
  } catch {
    throw new SourceError("service-error", "Crossref's answer can't be read.");
  }
};

export interface CrossrefClient {
  /** The work a DOI names; null when Crossref has none. */
  byDoi(doi: string, signal: AbortSignal): Promise<WorkRecord | null>;
  /** The works whose records best match a reference as written, best first. */
  search(reference: string, signal: AbortSignal): Promise<WorkRecord[]>;
}

export function createCrossrefClient(http: SourceHttp, baseUrl = CROSSREF_URL): CrossrefClient {
  return {
    async byDoi(doi, signal) {
      let body: string;
      try {
        body = await http.text(`${baseUrl}/works/${encodeURIComponent(doi)}`, {
          ...LANE,
          accept: "application/json",
          signal,
        });
      } catch (error) {
        if (error instanceof SourceError && error.reason === "not-found") return null;
        throw error;
      }
      const message = (parse(body) as { message?: unknown } | null)?.message;
      return workRecord(message);
    },

    async search(reference, signal) {
      const query = new URLSearchParams({
        "query.bibliographic": reference.slice(0, 1000),
        rows: String(ROWS),
        select: SELECT,
      });
      const body = await http.text(`${baseUrl}/works?${query}`, {
        ...LANE,
        accept: "application/json",
        signal,
      });
      const message = (parse(body) as { message?: unknown } | null)?.message;
      const items = isRecord(message) && Array.isArray(message.items) ? message.items : [];
      return items.map(workRecord).filter((record): record is WorkRecord => record !== null);
    },
  };
}
