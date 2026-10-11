import type { ArtifactReference } from "./api";
import { isRecord } from "./errors";

/** A reference to a Block of a Mind, by the Block's stable ID. */
export function blockReference(mindId: string, blockId: string): ArtifactReference {
  return { artifactKind: "mind", artifactId: mindId, anchor: { kind: "block", blockId } };
}

/** A reference to a message (a Question or an Answer) of one of the assistant's conversations. */
export function messageReference(conversationId: string, messageId: string): ArtifactReference {
  return {
    artifactKind: "conversation",
    artifactId: conversationId,
    anchor: { kind: "message", messageId },
  };
}

const isId = (value: unknown): value is string => typeof value === "string" && value !== "";

/**
 * Reads a reference from stored or received data, or null when it isn't one
 * (including an artifact or anchor kind this version doesn't know, or an anchor
 * that doesn't fit its artifact). One without `artifactKind` was stored before
 * conversations, and is a Mind's (ADR-0017).
 */
export function parseReference(value: unknown): ArtifactReference | null {
  if (!isRecord(value) || !isRecord(value.anchor) || !isId(value.artifactId)) return null;
  const { artifactKind = "mind", artifactId, anchor } = value;
  if (artifactKind === "mind" && anchor.kind === "block" && isId(anchor.blockId)) {
    return blockReference(artifactId, anchor.blockId);
  }
  if (artifactKind === "conversation" && anchor.kind === "message" && isId(anchor.messageId)) {
    return messageReference(artifactId, anchor.messageId);
  }
  return null;
}
