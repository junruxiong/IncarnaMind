import type { ArtifactReference } from "./api";
import { isRecord } from "./errors";

/** A reference to a Block of an artifact, by the Block's stable ID. */
export function blockReference(artifactId: string, blockId: string): ArtifactReference {
  return { artifactId, anchor: { kind: "block", blockId } };
}

/**
 * Reads a reference from stored or received data, or null when it isn't one
 * (including an anchor kind this version doesn't know).
 */
export function parseReference(value: unknown): ArtifactReference | null {
  if (!isRecord(value) || !isRecord(value.anchor)) return null;
  const { artifactId, anchor } = value;
  if (typeof artifactId !== "string" || artifactId === "") return null;
  if (anchor.kind !== "block" || typeof anchor.blockId !== "string" || anchor.blockId === "") {
    return null;
  }
  return blockReference(artifactId, anchor.blockId);
}
