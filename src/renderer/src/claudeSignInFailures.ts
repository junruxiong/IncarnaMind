import type { ClaudeSignInErrorKind } from "../../core/api";
import type { MessageKey } from "../../shared/i18n";

// The User's own Cancel is never shown, so a cancelled sign-in that is shown ran out of time.
export const CLAUDE_SIGN_IN_FAILURES: Record<ClaudeSignInErrorKind, MessageKey> = {
  "sign-in-cancelled": "claudePlan.failure.timedOut",
  refused: "claudeAccount.problem.refused",
  "unexpected-response": "claudeAccount.problem.unexpectedResponse",
  "secret-storage": "claudePlan.failure.secretStorage",
  unavailable: "claudePlan.failed",
  failed: "claudePlan.failed",
};
