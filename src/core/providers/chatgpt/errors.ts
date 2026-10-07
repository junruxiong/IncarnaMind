/** Failures specific to the ChatGPT plan provider, sorted by `classifyProviderError`. */

/** There is no usable ChatGPT sign-in on this device: never signed in, signed out, or a refresh failed. */
export class ChatGptSignInRequiredError extends Error {
  override name = "ChatGptSignInRequiredError";
  constructor(
    readonly reason: "signed-out" | "expired",
    message = reason === "expired"
      ? "Your ChatGPT sign-in expired. Sign in again."
      : "Sign in to ChatGPT first.",
  ) {
    super(message);
  }
}

/**
 * The ChatGPT plan's endpoint refused a request made with a valid sign-in:
 * "plan-limit" when the plan's usage limit is reached (or the plan doesn't
 * include this use), "blocked" for 401 or 403.
 */
export class ChatGptPlanError extends Error {
  override name = "ChatGptPlanError";
  constructor(
    readonly kind: "plan-limit" | "blocked",
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}
