import { APICallError, RetryError } from "ai";
import type { ProviderError } from "../api";
import { ConsentDeclinedError } from "../errors";

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/**
 * Sorts a failed provider request into a kind the UI can explain: a bad key,
 * an unknown model, rate limiting, no connection, or the provider's own error.
 * Answers (#29) reuse it for errors shown inside an Answer.
 */
export function classifyProviderError(error: unknown): ProviderError {
  const cause = RetryError.isInstance(error) ? error.lastError : error;
  const message = messageOf(cause);

  if (cause instanceof ConsentDeclinedError) return { kind: "consent-declined", message };

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
