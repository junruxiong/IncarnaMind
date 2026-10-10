import type { ProviderErrorKind } from "../../core/api";
import type { MessageKey } from "../../shared/i18n";
import type { SettingsPage } from "./settingsPages";

/**
 * What a failed Answer says, and the one thing it offers (DESIGN.md,
 * Problems): its line, and either a page of Settings to fix it on, trying
 * again, or nothing when neither would help.
 */
export type AnswerFix =
  | { kind: "settings"; page: SettingsPage; label: MessageKey }
  | { kind: "retry"; label: MessageKey }
  | { kind: "none" };

export interface AnswerProblem {
  /** The line, which names the model as `{model}` when it can. */
  line: MessageKey;
  fix: AnswerFix;
}

const retry: AnswerFix = { kind: "retry", label: "answer.error.retry" };

export const ANSWER_PROBLEMS: Record<ProviderErrorKind, AnswerProblem> = {
  auth: {
    line: "answer.error.auth",
    fix: { kind: "settings", page: "models", label: "answer.error.checkKey" },
  },
  model: {
    line: "answer.error.model",
    fix: { kind: "settings", page: "models", label: "answer.error.chooseModel" },
  },
  // A declined data flow is allowed again on the Privacy page.
  "consent-declined": {
    line: "answer.error.consent-declined",
    fix: { kind: "settings", page: "privacy", label: "answer.error.allow" },
  },
  "rate-limit": { line: "answer.error.rate-limit", fix: retry },
  network: { line: "answer.error.network", fix: retry },
  provider: { line: "answer.error.provider", fix: retry },
  // A local model's context window: trying again can't help, shortening the Question does.
  "too-long": { line: "answer.error.too-long", fix: { kind: "none" } },
  unknown: { line: "answer.error.unknown", fix: retry },
  // The experimental ChatGPT plan provider.
  "not-signed-in": {
    line: "answer.error.not-signed-in",
    fix: { kind: "settings", page: "models", label: "answer.error.signIn" },
  },
  "plan-limit": { line: "answer.error.plan-limit", fix: retry },
  blocked: {
    line: "answer.error.blocked",
    fix: { kind: "settings", page: "models", label: "answer.error.useKey" },
  },
};

/**
 * Why a file couldn't be written, in a few words, for "Couldn't save: …".
 * Null when the system's own words are all there is.
 */
export function saveFailureReason(message: string): MessageKey | null {
  if (/\b(EACCES|EPERM|EROFS)\b/.test(message)) return "export.failed.readOnly";
  if (/\b(ENOSPC|EDQUOT)\b/.test(message)) return "export.failed.full";
  if (/\bENOENT\b/.test(message)) return "export.failed.gone";
  return null;
}
