/**
 * A sentence's check (#96), as its mark stores it in the Mind
 * (`ClaimCheckAttributes`), read the same way by the core and the editor,
 * and whether it still holds: a check made from text, Documents or a scope
 * that have changed since needs a re-check.
 */
import {
  type CalculatedFigure,
  type CheckedVersion,
  type Citation,
  type CitationAttributes,
  type ClaimCalculation,
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
  "unread",
  "rate-limit",
  "cost-cap",
  "plan-limit",
  "failed",
  "stopped",
];

const EXCLUSIONS: readonly EvidenceExclusion[] = ["own-file", "own-workings", "checked-document"];

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

/** A figure worked out that decided a check (#707), as its mark keeps it. */
function calculationOf(value: unknown): ClaimCalculation | null {
  if (!isRecord(value) || !isRecord(value.figure)) return null;
  const stated = text(value.stated);
  const { figure } = value;
  if (stated === null || !Array.isArray(figure.inputs) || typeof figure.shown !== "string")
    return null;
  return {
    figure: figure as unknown as CalculatedFigure,
    stated,
    agrees: value.agrees === true,
  };
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
    providerId: text(value.providerId),
    modelId: text(value.modelId),
    text: text(value.text) ?? "",
    evidence: isRecord(value.evidence) ? (value.evidence as unknown as CitationAttributes) : null,
    says: text(value.says),
    calculation: calculationOf(value.calculation),
    citedCaseUnchecked: value.citedCaseUnchecked === true,
    citedCaseFound: value.citedCaseFound === true,
    scope: scopeOf(value.scope),
    versions: versionsOf(value.versions),
    excluded: excludedOf(value.excluded),
    checkedAt: text(value.checkedAt) ?? "",
  };
}

/**
 * Why a check no longer holds: its sentence was edited (an accepted change
 * or a restored Checkpoint edits it too), a Document it read has a new
 * version, one its result rests on was deleted, or the Mind moved to another
 * Folder.
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

/** What a check's result rests on: its evidence, and the quotes a worked-out figure read its numbers from. */
export function restsOn(check: ClaimCheckAttributes): (CitationAttributes | Citation)[] {
  const inputs = check.calculation?.figure.inputs ?? [];
  return [
    check.evidence,
    ...inputs.flatMap((input) => [input.citation, input.unitStatement?.citation ?? null]),
  ].filter((each): each is CitationAttributes | Citation => each !== null);
}

/** How a check shows, given the sentence, the Documents and the Mind's Folder as they are now. */
export function claimCheckShown(check: ClaimCheckAttributes, now: ClaimCheckNow): ClaimCheckShown {
  if (check.state === "checking") return { state: "checking" };
  if (comparableSentence(now.text) !== comparableSentence(check.text)) {
    return { state: "needs-recheck", why: "edited" };
  }
  if (now.versions) {
    const relied = new Set(restsOn(check).map((each) => each.documentId));
    for (const version of check.versions) {
      const current = now.versions.get(version.documentId);
      // A Document only read, gone, takes nothing from the result: an example's, swapped for the interface language, among them (#962).
      if (current === undefined && relied.has(version.documentId)) {
        return { state: "needs-recheck", why: "document-removed" };
      }
      if (current !== undefined && current !== version.contentHash) {
        return { state: "needs-recheck", why: "new-version" };
      }
    }
  }
  if (!check.scope.widened && check.scope.folderId !== now.folderId) {
    return { state: "needs-recheck", why: "scope" };
  }
  if (check.state === "not-checked")
    return { state: "not-checked", reason: check.reason ?? "failed" };
  return { state: check.state };
}
