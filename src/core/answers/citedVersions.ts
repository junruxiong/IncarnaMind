/**
 * Which versions of which Documents the Citations in Minds quote, for the
 * garbage collection of old versions' text (see `collectOldVersions` in
 * ../documents). Citations live in Minds' Yjs documents, anywhere: in
 * Answers, or copied into Notes, at any depth. Exports read them the same
 * way (../exports/model).
 */
import * as Y from "yjs";
import { CITATION_NODE } from "../api";

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
