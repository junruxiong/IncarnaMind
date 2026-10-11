/**
 * One list from many providers: the works they share (by DOI, arXiv ID,
 * PubMed or PMC ID, or OpenAlex ID) become one, with the best of each
 * provider's record, every open copy any of them knows, and a rank by
 * reciprocal rank fusion, as hybrid search fuses its lists (k = 60): found by
 * more providers, and higher by each, ranks higher.
 */
import { emptyReferenceFields, isEmptyValue } from "../../shared/references";
import type { ReferenceField, ReferenceFields, SourceVersion } from "../api";
import type { FullTextLocation, ProviderWork, WorkIds } from "./types";

/** Reciprocal rank fusion's k. */
const K = 60;

/** Whose record wins a field both have: the registries' first, a preprint server's last. */
const RECORD_ORDER = ["crossref", "openalex", "pubmed", "arxiv"];

/** Which open copy is tried first: the published version, then one whose version isn't known, then a manuscript. */
const VERSION_ORDER: Record<SourceVersion | "unknown", number> = {
  published: 0,
  unknown: 1,
  accepted: 2,
  submitted: 3,
};

export interface MergedWork {
  key: string;
  ids: WorkIds;
  fields: ReferenceFields;
  /** Provider ids, best first. */
  foundBy: string[];
  citedBy: number | null;
  versionDate: string | null;
  /** Every open copy known, the one to try first first. */
  fullText: FullTextLocation[];
  landingUrl: string | null;
  score: number;
}

interface Found {
  provider: string;
  /** 1 for a provider's first result. */
  rank: number;
  work: ProviderWork;
}

const ID_KINDS = ["doi", "arxiv", "pmid", "pmcid", "openalex"] as const;

const shares = (a: WorkIds, b: WorkIds) =>
  ID_KINDS.some((kind) => a[kind] !== undefined && a[kind] === b[kind]);

/** A work's key: its DOI, else its arXiv, PubMed or OpenAlex ID, else its title. */
export function workKey(ids: WorkIds, title: string): string {
  if (ids.doi) return `doi:${ids.doi}`;
  if (ids.arxiv) return `arxiv:${ids.arxiv}`;
  if (ids.pmid) return `pmid:${ids.pmid}`;
  if (ids.openalex) return `openalex:${ids.openalex}`;
  return `title:${title.toLowerCase().replace(/\s+/g, " ").trim()}`;
}

const recordRank = (provider: string) => {
  const at = RECORD_ORDER.indexOf(provider);
  return at < 0 ? RECORD_ORDER.length : at;
};

function mergeGroup(group: readonly Found[]): MergedWork {
  const byRecord = [...group].sort(
    (a, b) => recordRank(a.provider) - recordRank(b.provider) || a.rank - b.rank,
  );
  const fields = emptyReferenceFields();
  for (const field of Object.keys(fields) as ReferenceField[]) {
    const from = byRecord.find((each) => {
      const value = each.work.fields[field];
      return value !== undefined && !isEmptyValue(field, value);
    });
    if (from)
      (fields as Record<ReferenceField, ReferenceFields[ReferenceField]>)[field] = from.work.fields[
        field
      ] as ReferenceFields[ReferenceField];
  }
  const ids: WorkIds = {};
  for (const { work } of byRecord) {
    for (const kind of ID_KINDS) {
      const id = work.ids[kind];
      if (id !== undefined && ids[kind] === undefined) ids[kind] = id;
    }
  }
  const seen = new Set<string>();
  const copies = byRecord
    .flatMap((each) => each.work.fullText)
    .filter((copy) => {
      const id = `${copy.provider ?? ""}\n${copy.url}`;
      if (seen.has(id)) return false;
      seen.add(id);
      return true;
    })
    .map((copy, order) => ({ copy, order }))
    .sort(
      (a, b) =>
        VERSION_ORDER[a.copy.version ?? "unknown"] - VERSION_ORDER[b.copy.version ?? "unknown"] ||
        a.order - b.order,
    )
    .map(({ copy }) => copy);
  const providers: string[] = [];
  for (const each of [...group].sort((a, b) => a.rank - b.rank)) {
    if (!providers.includes(each.provider)) providers.push(each.provider);
  }
  const counts = group
    .map((each) => each.work.citedBy)
    .filter((count): count is number => count !== null);
  return {
    key: workKey(ids, fields.title),
    ids,
    fields,
    foundBy: providers,
    citedBy: counts.length > 0 ? Math.max(...counts) : null,
    versionDate:
      copies[0]?.versionDate ??
      byRecord.find((each) => each.work.versionDate)?.work.versionDate ??
      null,
    fullText: copies,
    landingUrl: byRecord.find((each) => each.work.landingUrl)?.work.landingUrl ?? null,
    score: group.reduce((sum, each) => sum + 1 / (K + each.rank), 0),
  };
}

/** Each provider's works, in its own order of relevance, merged and ranked. */
export function mergeWorks(
  lists: readonly { provider: string; works: readonly ProviderWork[] }[],
): MergedWork[] {
  const groups: Found[][] = [];
  for (const { provider, works } of lists) {
    works.forEach((work, index) => {
      const found: Found = { provider, rank: index + 1, work };
      const matching = groups.filter((group) =>
        group.some((each) => shares(each.work.ids, work.ids)),
      );
      const merged = [...matching.flat(), found];
      for (const group of matching) groups.splice(groups.indexOf(group), 1);
      groups.push(merged);
    });
  }
  return groups
    .map(mergeGroup)
    .sort(
      (a, b) =>
        b.score - a.score ||
        b.foundBy.length - a.foundBy.length ||
        (b.citedBy ?? 0) - (a.citedBy ?? 0),
    );
}
