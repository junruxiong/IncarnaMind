/**
 * A sentence's check (#96), as its mark stores it in the Mind
 * (`ClaimCheckAttributes`), read the same way by the core and the editor,
 * and whether it still holds: a check made from text, Documents or a scope
 * that have changed since needs a re-check.
 */
import {
  type CheckedVersion,
  type CitationAttributes,
  type ClaimCheckAction,
  type ClaimCheckAttributes,
  type ClaimCheckFailure,
  type ClaimCheckScope,
  type ClaimVerdict,
  claimCheckActions,
  type EvidenceExclusion,
  type LeftOutDocument,
} from "../core/api";
import { comparableSentence } from "./sentences";

const STATES: readonly ClaimCheckAttributes["state"][] = [
  "checking",
  "supported",
  "unsupported",
  "contradicted",
  "not-checked",
];

const FAILURES: readonly ClaimCheckFailure[] = [
  "refused",
  "malformed",
  "empty",
  "no-evidence",
  "rate-limit",
  "cost-cap",
  "plan-limit",
  "failed",
  "stopped",
];

const EXCLUSIONS: readonly EvidenceExclusion[] = ["own-file", "checked-document"];

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const text = (value: unknown): string | null => (typeof value === "string" ? value : null);

function versionsOf(value: unknown): CheckedVersion[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((each) => {
    if (!isRecord(each)) return [];
    const documentId = text(each.documentId);
    const contentHash = text(each.contentHash);
    return documentId && contentHash ? [{ documentId, contentHash }] : [];
  });
}

function excludedOf(value: unknown): LeftOutDocument[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((each) => {
    if (!isRecord(each)) return [];
    const documentId = text(each.documentId);
    const reason = EXCLUSIONS.find((known) => known === each.reason);
    return documentId && reason
      ? [{ documentId, documentName: text(each.documentName) ?? "", reason }]
      : [];
  });
}

function scopeOf(value: unknown): ClaimCheckScope {
  if (!isRecord(value)) return { folderId: null, widened: false };
  return { folderId: text(value.folderId), widened: value.widened === true };
}

/** A check's attributes as a mark holds them, or null when they aren't one's. Unknown values read safely. */
export function readClaimCheck(value: unknown): ClaimCheckAttributes | null {
  if (!isRecord(value)) return null;
  const id = text(value.id);
  const state = STATES.find((each) => each === value.state);
  if (!id || !state) return null;
  const action: ClaimCheckAction =
    claimCheckActions.find((each) => each === value.action) ?? "find-support";
  return {
    id,
    request: text(value.request) ?? "",
    action,
    state,
    reason: FAILURES.find((each) => each === value.reason) ?? null,
    detail: text(value.detail),
    text: text(value.text) ?? "",
    evidence: isRecord(value.evidence) ? (value.evidence as unknown as CitationAttributes) : null,
    says: text(value.says),
    scope: scopeOf(value.scope),
    versions: versionsOf(value.versions),
    excluded: excludedOf(value.excluded),
    checkedAt: text(value.checkedAt) ?? "",
  };
}

/**
 * Why a check no longer holds: its sentence was edited (an accepted change
 * or a restored Checkpoint edits it too), a Document it read has a new
 * version or was deleted, or the Mind moved to another Folder.
 */
export type RecheckReason = "edited" | "new-version" | "document-removed" | "scope";

/** A check as it shows now. */
export type ClaimCheckShown =
  | { state: "checking" }
  | { state: ClaimVerdict }
  | { state: "not-checked"; reason: ClaimCheckFailure }
  | { state: "needs-recheck"; why: RecheckReason };

/** What a check is compared with. */
export interface ClaimCheckNow {
  /** The sentence as it reads now: all of the sentences its marked text is in. */
  text: string;
  /** Each live Document's current version, by id; null while they aren't known. */
  versions: ReadonlyMap<string, string> | null;
  /** The Mind's Folder now; null for none. */
  folderId: string | null;
}

/** How a check shows, given the sentence, the Documents and the Mind's Folder as they are now. */
export function claimCheckShown(check: ClaimCheckAttributes, now: ClaimCheckNow): ClaimCheckShown {
  if (check.state === "checking") return { state: "checking" };
  if (comparableSentence(now.text) !== comparableSentence(check.text)) {
    return { state: "needs-recheck", why: "edited" };
  }
  if (now.versions) {
    for (const version of check.versions) {
      const current = now.versions.get(version.documentId);
      if (current === undefined) return { state: "needs-recheck", why: "document-removed" };
      if (current !== version.contentHash) return { state: "needs-recheck", why: "new-version" };
    }
  }
  if (!check.scope.widened && check.scope.folderId !== now.folderId) {
    return { state: "needs-recheck", why: "scope" };
  }
  if (check.state === "not-checked")
    return { state: "not-checked", reason: check.reason ?? "failed" };
  return { state: check.state };
}
