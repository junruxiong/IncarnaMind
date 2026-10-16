/** Failures specific to the ChatGPT plan provider, sorted by `classifyProviderError`. */

/** There is no usable ChatGPT sign-in on this device: never signed in, signed out, or a refresh failed. */
export class ChatGptSignInRequiredError extends Error {
  override name = "ChatGptSignInRequiredError";
  constructor(
    readonly reason: "signed-out" | "expired",
    message = reason === "expired"
      ? "Your ChatGPT sign-in expired. Continue with ChatGPT to sign in again."
      : "Continue with ChatGPT to sign in first.",
  ) {
    super(message);
  }
}

/**
 * OpenAI refused a request made with a valid sign-in: "plan-limit" when the
 * plan's or this app's usage limit is reached, "blocked" when the plan can't
 * be used here (an ineligible plan, a region, a refused permission).
 */
export class ChatGptPlanError extends Error {
  override name = "ChatGptPlanError";
  constructor(
    readonly kind: "plan-limit" | "blocked",
    message: string,
    readonly status: number,
    /** When a plan limit resets (epoch milliseconds), when OpenAI says. */
    readonly resetsAt: number | null = null,
  ) {
    super(message);
  }
}

/** The plan's or this app's usage limit is reached: a request refused, or tagging waiting for it to reset. */
export const isPlanLimit = (error: unknown): error is ChatGptPlanError =>
  error instanceof ChatGptPlanError && error.kind === "plan-limit";

/** The User signed in but didn't allow IncarnaMind to use their ChatGPT plan. */
export class ChatGptPlanNotAllowedError extends Error {
  override name = "ChatGptPlanNotAllowedError";
  constructor() {
    super("ChatGPT plan use wasn't allowed, so IncarnaMind can't use your plan.");
  }
}

/** A sign-in came back for another ChatGPT account than the one this device is registered with. */
export class ChatGptOtherAccountError extends Error {
  override name = "ChatGptOtherAccountError";
  constructor(readonly expected: string | null) {
    super(
      expected
        ? `That's another ChatGPT account. Sign in as ${expected}, or use another account.`
        : "That's another ChatGPT account than before. Sign in with the same one, or use another account.",
    );
  }
}
