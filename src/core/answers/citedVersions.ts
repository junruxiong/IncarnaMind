/**
 * Which versions of which Documents the Citations in Minds quote, for the
 * garbage collection of old versions' text (see `collectOldVersions` in
 * ../documents), and which of their Units, for the text kept when a Linked
 * folder is unlinked. Citations live in Minds' Yjs documents, anywhere: in
 * Answers, or copied into Notes, at any depth. Exports read them the same
 * way (../exports/model).
 */
import * as Y from "yjs";
import { CITATION_NODE } from "../api";

/**
 * What a Citation's check reads: the stored text of one version of a
 * Document, Units `pageFrom` to `pageTo` (see `pageTexts` in ../documents).
 */
export interface CitedUnits {
  documentId: string;
  /** The version it quotes; null for a Citation that doesn't say, which means every version. */
  contentHash: string | null;
  /** The cited Units, from 1; both null for a whole TXT or Markdown file, cited before Units: all of them. */
  pageFrom: number | null;
  pageTo: number | null;
}

const unitNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isInteger(value) && value >= 1 ? value : null;

/** Adds the Units that the Citations under `fragment` point to in these Documents to `cited`. */
export function addCitedUnits(
  fragment: Y.XmlFragment | Y.XmlElement,
  documentIds: ReadonlySet<string>,
  cited: CitedUnits[],
): void {
  for (const child of fragment.toArray()) {
    if (!(child instanceof Y.XmlElement)) continue;
    if (child.nodeName !== CITATION_NODE) {
      addCitedUnits(child, documentIds, cited);
      continue;
    }
    const documentId = child.getAttribute("documentId");
    if (typeof documentId !== "string" || !documentIds.has(documentId)) continue;
    const contentHash = child.getAttribute("contentHash");
    const from = unitNumber(child.getAttribute("pageFrom"));
    // Without a first Unit, it points at all of them.
    const to = from === null ? null : (unitNumber(child.getAttribute("pageTo")) ?? from);
    cited.push({
      documentId,
      contentHash: typeof contentHash === "string" && contentHash !== "" ? contentHash : null,
      pageFrom: from === null || to === null ? null : Math.min(from, to),
      pageTo: from === null || to === null ? null : Math.max(from, to),
    });
  }
}

/** The versions Citations quote: each Document's cited content hashes, or "any" for every version. */
export type CitedVersions = Map<string, Set<string> | "any">;

/** Adds the versions the Citations under `fragment` quote to `cited`. */
export function addCitedVersions(fragment: Y.XmlFragment | Y.XmlElement, cited: CitedVersions) {
  for (const child of fragment.toArray()) {
    if (!(child instanceof Y.XmlElement)) continue;
    if (child.nodeName === CITATION_NODE) {
      const documentId = child.getAttribute("documentId");
      if (typeof documentId !== "string" || documentId === "") continue;
      const contentHash = child.getAttribute("contentHash");
      const versions = cited.get(documentId);
      // A Citation that doesn't say which version it quotes keeps every version.
      if (typeof contentHash !== "string" || contentHash === "") cited.set(documentId, "any");
      else if (versions === undefined) cited.set(documentId, new Set([contentHash]));
      else if (versions !== "any") versions.add(contentHash);
      continue;
    }
    addCitedVersions(child, cited);
  }
}

/** Whether a Citation in `cited` quotes this version of this Document. */
export function isCited(cited: CitedVersions, documentId: string, contentHash: string): boolean {
  const versions = cited.get(documentId);
  return versions === "any" || (versions?.has(contentHash) ?? false);
}
