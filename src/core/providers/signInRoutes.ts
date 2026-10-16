/**
 * What a route signed in with the User's own account says where it stopped:
 * a line naming the provider, and the ways back, the first the main one.
 * Kept apart from the request adapters so the renderer can import it.
 */
import type { MessageKey } from "../../shared/i18n";
import type { ChatProviderKind, SignInRouteProblem } from "../api";

export type SignInRoute = Extract<ChatProviderKind, "claude-account" | "copilot">;

/** A way back; "sign-in" signs in with the route's provider, which its label names. */
export type RouteWayBack =
  | "sign-in"
  | "sign-in-again"
  | "use-another-model"
  | "paste-api-key"
  | "use-claude-code";

export interface RouteProblemWords {
  line: MessageKey;
  /** The line when the provider says when it ends, with `{time}`. */
  lineUntil?: MessageKey;
  waysBack: readonly RouteWayBack[];
}

export interface SignInRouteWords {
  signIn: MessageKey;
  problems: Partial<Record<SignInRouteProblem, RouteProblemWords>>;
}

const WAYS_BACK: Record<Exclude<RouteWayBack, "sign-in">, MessageKey> = {
  "sign-in-again": "signInRoute.wayBack.signInAgain",
  "use-another-model": "signInRoute.wayBack.useAnotherModel",
  "paste-api-key": "signInRoute.wayBack.pasteApiKey",
  "use-claude-code": "signInRoute.wayBack.useClaudeCode",
};

export const SIGN_IN_ROUTES: Record<SignInRoute, SignInRouteWords> = {
  "claude-account": {
    signIn: "claudeAccount.wayBack.signIn",
    problems: {
      "sign-in-cancelled": { line: "claudeAccount.problem.signInCancelled", waysBack: ["sign-in"] },
      "signed-out": { line: "claudeAccount.problem.signedOut", waysBack: ["sign-in"] },
      "sign-in-expired": {
        line: "claudeAccount.problem.signInExpired",
        waysBack: ["sign-in-again"],
      },
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
    },
  },
  copilot: {
    signIn: "copilot.wayBack.signIn",
    problems: {
      "sign-in-cancelled": { line: "copilot.problem.signInCancelled", waysBack: ["sign-in"] },
      "signed-out": { line: "copilot.problem.signedOut", waysBack: ["sign-in"] },
      "sign-in-expired": { line: "copilot.problem.signInExpired", waysBack: ["sign-in-again"] },
      refused: {
        line: "copilot.problem.refused",
        waysBack: ["sign-in-again", "paste-api-key"],
      },
      "no-plan": {
        line: "copilot.problem.noPlan",
        waysBack: ["sign-in-again", "use-another-model"],
      },
      "policy-blocked": {
        line: "copilot.problem.policyBlocked",
        waysBack: ["use-another-model", "sign-in-again"],
      },
      "model-unavailable": {
        line: "copilot.problem.modelUnavailable",
        waysBack: ["use-another-model"],
      },
      "plan-limit": {
        line: "copilot.problem.planLimit",
        lineUntil: "copilot.problem.planLimitUntil",
        waysBack: ["use-another-model"],
      },
      "unexpected-response": {
        line: "copilot.problem.unexpectedResponse",
        waysBack: ["paste-api-key", "use-another-model"],
      },
    },
  },
};

export const wayBackLabel = (route: SignInRoute, wayBack: RouteWayBack): MessageKey =>
  wayBack === "sign-in" ? SIGN_IN_ROUTES[route].signIn : WAYS_BACK[wayBack];
