import {
  CONTENT_SCHEMA_VERSION,
  DEFAULT_MIND_KIND,
  MIND_KINDS,
  type MindAccess,
  type MindKind,
} from "./api";

export const isKnownKind = (kind: string): kind is MindKind =>
  (MIND_KINDS as readonly string[]).includes(kind);

/** Reads a kind from input: absent means the default; anything else must be one this version knows. */
export function parseKind(kind: unknown): MindKind | null {
  if (kind === undefined) return DEFAULT_MIND_KIND;
  return typeof kind === "string" && isKnownKind(kind) ? kind : null;
}

/** The version recorded in a Mind's settings; none, or anything malformed, reads as version 1. */
export function readContentVersion(recorded: unknown): number {
  return typeof recorded === "number" && Number.isInteger(recorded) && recorded >= 1 ? recorded : 1;
}

/** How a Mind of this kind, written with this content schema version, opens in this version. */
export function accessFor(kind: string, contentVersion: number): MindAccess {
  if (!isKnownKind(kind)) return "update-required";
  return contentVersion > CONTENT_SCHEMA_VERSION ? "read-only" : "edit";
}
