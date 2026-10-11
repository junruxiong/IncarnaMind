/**
 * Telling whether two references are the same work (#287): by DOI, then by
 * arXiv id, then by title with the year and the first author. Two different
 * DOIs are two works, whatever their titles say (a preprint and its
 * published version are cited apart).
 */
import { arxivIdOf } from "../../../shared/references";
import type { ReferenceFields, ReferenceName } from "../../api";

export type MatchedBy = "doi" | "arxiv" | "title";

export interface Identity {
  doi: string;
  arxiv: string;
  title: string;
  year: string;
  author: string;
}

/** Letters and digits of any script, lower-cased and with accents folded: "Deep-Learning!" is "deeplearning". */
export const squash = (text: string) =>
  text
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]/gu, "");

const ARXIV_DOI = /^10\.48550\/arxiv\.(.+)$/i;

const familyOf = (name: ReferenceName) => ("family" in name ? name.family : name.literal);

/** A work's identity from the fields an export or a record gives. */
export function identityOf(
  fields: Partial<ReferenceFields>,
  /** Another title to try when the fields have none: a Document's own, or its name. */
  fallbackTitle = "",
): Identity {
  const doi = (fields.doi ?? "").toLowerCase();
  const fromDoi = ARXIV_DOI.exec(doi)?.[1];
  const fromUrl = arxivIdOf(fields.url ?? "");
  const arxiv = (fromDoi ?? fromUrl ?? "").replace(/v\d+$/i, "").toLowerCase();
  const first = fields.authors?.[0];
  return {
    // arXiv's own DOI is the arXiv id, not a second work to tell apart.
    doi: fromDoi ? "" : doi,
    arxiv,
    title: squash(fields.title || fallbackTitle),
    year: (fields.issued ?? "").slice(0, 4),
    author: first ? squash(familyOf(first)) : "",
  };
}

/** The shortest title (once squashed) that says enough on its own when nothing else can tell works apart. */
const LONG_ENOUGH = 12;

function compatible(a: Identity, b: Identity): boolean {
  if (a.doi && b.doi && a.doi !== b.doi) return false;
  if (a.year && b.year && a.year !== b.year) return false;
  if (a.author && b.author && a.author !== b.author) return false;
  const confirmed = (a.year && b.year) || (a.author && b.author);
  return Boolean(confirmed) || a.title.length >= LONG_ENOUGH;
}

/** Things found by what identifies them. */
export class Index<T> {
  private readonly byDoi = new Map<string, { value: T; identity: Identity }>();
  private readonly byArxiv = new Map<string, { value: T; identity: Identity }>();
  private readonly byTitle = new Map<string, { value: T; identity: Identity }[]>();

  add(identity: Identity, value: T): void {
    const entry = { value, identity };
    if (identity.doi && !this.byDoi.has(identity.doi)) this.byDoi.set(identity.doi, entry);
    if (identity.arxiv && !this.byArxiv.has(identity.arxiv))
      this.byArxiv.set(identity.arxiv, entry);
    if (identity.title.length >= 4) {
      const list = this.byTitle.get(identity.title) ?? [];
      list.push(entry);
      this.byTitle.set(identity.title, list);
    }
  }

  find(identity: Identity): { value: T; by: MatchedBy } | null {
    const doi = identity.doi ? this.byDoi.get(identity.doi) : undefined;
    if (doi) return { value: doi.value, by: "doi" };
    const arxiv = identity.arxiv ? this.byArxiv.get(identity.arxiv) : undefined;
    if (arxiv) return { value: arxiv.value, by: "arxiv" };
    for (const candidate of this.byTitle.get(identity.title) ?? []) {
      if (identity.title !== "" && compatible(identity, candidate.identity)) {
        return { value: candidate.value, by: "title" };
      }
    }
    return null;
  }
}
