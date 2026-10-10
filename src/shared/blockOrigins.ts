/**
 * Where a Mind's Blocks came from (#201): a Note added from an Answer in a
 * chat keeps a link back to it, one reference (`ArtifactReference`, #199) by
 * Block id in the Mind's Yjs document (`MIND_ORIGINS_FIELD`). Kept beside the
 * Blocks, not in their attributes, so an older version's editor, which strips
 * attributes it doesn't know, leaves it alone. A Block deleted since leaves
 * its entry unused. Shared by the core, the renderer and the tests.
 */
import type * as Y from "yjs";
import { type BlockOrigin, MIND_ORIGINS_FIELD } from "../core/api";
import { parseReference } from "../core/references";

const originsOf = (doc: Y.Doc): Y.Map<unknown> => doc.getMap(MIND_ORIGINS_FIELD);

export function parseOrigin(value: unknown): BlockOrigin | null {
  if (typeof value !== "object" || value === null) return null;
  const { from, label } = value as Record<string, unknown>;
  const reference = parseReference(from);
  if (!reference) return null;
  return { from: reference, label: typeof label === "string" ? label : "" };
}

export function originOf(doc: Y.Doc, blockId: string): BlockOrigin | null {
  return parseOrigin(originsOf(doc).get(blockId));
}

export function allOrigins(doc: Y.Doc): Map<string, BlockOrigin> {
  const origins = new Map<string, BlockOrigin>();
  originsOf(doc).forEach((value, blockId) => {
    const origin = parseOrigin(value);
    if (origin) origins.set(blockId, origin);
  });
  return origins;
}

export function setOrigins(doc: Y.Doc, blockIds: readonly string[], origin: BlockOrigin): void {
  const origins = originsOf(doc);
  const stored = { from: origin.from, label: origin.label };
  doc.transact(() => {
    for (const blockId of blockIds) origins.set(blockId, stored);
  });
}

export function removeOrigins(doc: Y.Doc, blockIds: readonly string[]): void {
  const origins = originsOf(doc);
  doc.transact(() => {
    for (const blockId of blockIds) if (origins.has(blockId)) origins.delete(blockId);
  });
}

export function observeOrigins(doc: Y.Doc, listener: () => void): () => void {
  const origins = originsOf(doc);
  const onChange = () => listener();
  origins.observe(onChange);
  return () => origins.unobserve(onChange);
}
