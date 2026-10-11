/** Checks what the window asks `formatCitations` for. */
import { isCitationStyleId } from "../../shared/citationStyles";
import { parseLocation } from "../../shared/locations";
import type { CitationCluster, CitationFormatRequest, CitedWork } from "../api";
import { InvalidInputError, isRecord } from "../errors";

/** More Citations than any Mind holds, to keep a request from being unbounded. */
const MAX_WORKS = 20_000;

const text = (value: unknown) => (typeof value === "string" ? value : null);

function parseWork(value: unknown): CitedWork {
  if (!isRecord(value)) throw new InvalidInputError("A cited work must be an object.");
  return {
    documentId: text(value.documentId),
    documentName: text(value.documentName),
    location: value.location === null ? null : parseLocation(value.location),
  };
}

export function parseRequest(input: unknown): CitationFormatRequest {
  if (!isRecord(input)) throw new InvalidInputError("A format request must be an object.");
  if (!isCitationStyleId(input.style)) {
    throw new InvalidInputError(`"${String(input.style)}" isn't a citation style.`);
  }
  if (!Array.isArray(input.clusters)) throw new InvalidInputError("clusters must be a list.");
  let works = 0;
  const clusters: CitationCluster[] = input.clusters.map((cluster: unknown) => {
    if (!isRecord(cluster) || !Array.isArray(cluster.works) || cluster.works.length === 0) {
      throw new InvalidInputError("Each cluster must list one or more cited works.");
    }
    works += cluster.works.length;
    if (works > MAX_WORKS) throw new InvalidInputError("Too many Citations to format at once.");
    return { works: cluster.works.map(parseWork) };
  });
  return { style: input.style, clusters };
}
