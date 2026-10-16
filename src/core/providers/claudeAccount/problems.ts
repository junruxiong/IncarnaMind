/**
 * What the Claude account route says where it stopped: a line naming Claude,
 * and the ways back, the first the main one. Kept apart from the request
 * adapter so the renderer can import it.
 */
import type { MessageKey } from "../../../shared/i18n";
import type { ClaudeAccountProblem } from "../../api";

export type ClaudeAccountWayBack =
  | "sign-in"
  | "sign-in-again"
  | "use-another-model"
  | "paste-api-key"
  | "use-claude-code";

export const CLAUDE_ACCOUNT_WAYS_BACK: Record<ClaudeAccountWayBack, MessageKey> = {
  "sign-in": "claudeAccount.wayBack.signIn",
  "sign-in-again": "claudeAccount.wayBack.signInAgain",
  "use-another-model": "claudeAccount.wayBack.useAnotherModel",
  "paste-api-key": "claudeAccount.wayBack.pasteApiKey",
  "use-claude-code": "claudeAccount.wayBack.useClaudeCode",
};

export interface ClaudeAccountProblemWords {
  line: MessageKey;
  /** The line when the provider says when it ends, with `{time}`. */
  lineUntil?: MessageKey;
  waysBack: readonly ClaudeAccountWayBack[];
}

export const CLAUDE_ACCOUNT_PROBLEMS: Record<ClaudeAccountProblem, ClaudeAccountProblemWords> = {
  "sign-in-cancelled": { line: "claudeAccount.problem.signInCancelled", waysBack: ["sign-in"] },
  "signed-out": { line: "claudeAccount.problem.signedOut", waysBack: ["sign-in"] },
  "sign-in-expired": { line: "claudeAccount.problem.signInExpired", waysBack: ["sign-in-again"] },
  refused: {
    line: "claudeAccount.problem.refused",
    waysBack: ["paste-api-key", "use-claude-code"],
  },
  "plan-limit": {
    line: "claudeAccount.problem.planLimit",
    lineUntil: "claudeAccount.problem.planLimitUntil",
    waysBack: ["use-another-model"],
  },
  "unexpected-response": {
    line: "claudeAccount.problem.unexpectedResponse",
    waysBack: ["paste-api-key", "use-claude-code"],
  },
};
