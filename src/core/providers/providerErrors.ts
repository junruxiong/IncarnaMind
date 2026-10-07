import { APICallError, RetryError } from "ai";
import type { ProviderError } from "../api";
import { ConsentDeclinedError } from "../errors";
import { OAuthTokenError } from "../oauth";
import { ChatGptPlanError, ChatGptSignInRequiredError } from "./chatgpt/errors";

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/**
 * Sorts a failed provider request into a kind the UI can explain: a bad key,
 * an unknown model, rate limiting, no connection, or the provider's own error.
 * For the ChatGPT plan, also a missing or expired sign-in, a reached plan
 * limit, or OpenAI refusing the sign-in.
 * Answers (#29) reuse it for errors shown inside an Answer.
 */
export function classifyProviderError(error: unknown): ProviderError {
  const cause = RetryError.isInstance(error) ? error.lastError : error;
  const message = messageOf(cause);

  if (cause instanceof ConsentDeclinedError) return { kind: "consent-declined", message };
  if (cause instanceof ChatGptSignInRequiredError) return { kind: "not-signed-in", message };
  if (cause instanceof ChatGptPlanError) return { kind: cause.kind, message };
  // The ChatGPT sign-in couldn't be refreshed for a passing reason (e.g. OpenAI's server failed).
  if (cause instanceof OAuthTokenError) return { kind: "provider", message };

  if (APICallError.isInstance(cause)) {
    const status = cause.statusCode;
    // No status: the request never got a response.
    if (status === undefined) return { kind: "network", message };
    if (status === 401 || status === 403) return { kind: "auth", message };
    // Google answers a bad key with 400 "API key not valid".
    if (status === 400 && /api[ _-]?key/i.test(message)) return { kind: "auth", message };
    if (status === 404) return { kind: "model", message };
    if (status === 429) return { kind: "rate-limit", message };
    return { kind: "provider", message };
  }

  if (cause instanceof Error) {
    if (cause.name === "AbortError" || cause.name === "TimeoutError") {
      return { kind: "network", message };
    }
    if (cause instanceof TypeError && /fetch failed|network/i.test(message)) {
      return { kind: "network", message };
    }
  }
  return { kind: "unknown", message };
}
