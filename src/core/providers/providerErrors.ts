import { APICallError, RetryError, UnsupportedFunctionalityError } from "ai";
import type { ProviderError, ProviderErrorKind } from "../api";
import { ConsentDeclinedError } from "../errors";
import { OAuthTokenError } from "../oauth";
import { ChatGptPlanError, ChatGptSignInRequiredError } from "./chatgpt/errors";
import { JevRequestError } from "./jev";

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** A local model can't take a request, found before anything is sent: e.g. it isn't a chat model. */
export class LocalModelError extends Error {
  constructor(
    readonly kind: Extract<ProviderErrorKind, "model" | "too-long">,
    message: string,
  ) {
    super(message);
    this.name = "LocalModelError";
  }
}

/**
 * Ollama's refusal of a request longer than the model's context window (sent
 * with `truncate: false`): "request (6029 tokens) exceeds the available
 * context size (4096 tokens)". The counts, when it gives them.
 */
export function contextOverflow(
  error: unknown,
): { promptTokens: number | null; windowTokens: number | null } | null {
  const cause = RetryError.isInstance(error) ? error.lastError : error;
  if (!APICallError.isInstance(cause) || cause.statusCode !== 400) return null;
  const text = `${cause.message} ${cause.responseBody ?? ""}`;
  if (!/exceeds? the available context size|exceed_context_size/i.test(text)) return null;
  const count = (pattern: RegExp) => {
    const found = pattern.exec(text);
    return found ? Number(found[1]) : null;
  };
  return {
    promptTokens: count(/request \((\d+) tokens?\)/i),
    windowTokens: count(/context size \((\d+) tokens?\)/i),
  };
}

/**
 * Whether a provider refused a request because the model can't use `feature`:
 * e.g. Ollama's "model does not support tools", vLLM's "--enable-auto-tool-choice",
 * a server that rejects `response_format`, or OpenAI's "Unsupported parameter:
 * 'temperature'" for a reasoning model. Auth, rate limits and outages never count.
 */
export function refusesFeature(
  error: unknown,
  feature: "tools" | "structured-output" | "temperature",
): boolean {
  const cause = RetryError.isInstance(error) ? error.lastError : error;
  const subject =
    feature === "tools"
      ? /tool|function/i
      : feature === "temperature"
        ? /temperature/i
        : /response_format|response format|json_schema|json schema|json mode|json_object|structured output|format/i;
  if (UnsupportedFunctionalityError.isInstance(cause)) return subject.test(cause.functionality);
  if (!APICallError.isInstance(cause)) return false;
  const status = cause.statusCode;
  if (status === undefined || [401, 403, 404, 408, 429].includes(status) || status >= 502) {
    return false;
  }
  const text = `${cause.message} ${cause.responseBody ?? ""}`;
  const refusal =
    /not support|unsupported|doesn't support|does not support|not enabled|not available|isn't available|requires --|requires the --|not allowed|is invalid|invalid value|unknown (?:field|parameter|argument)|unrecognized/i;
  // A temperature is also refused as deprecated, or as other than the default.
  const temperatureRefusal = /deprecated|only the default/i;
  return (
    subject.test(text) &&
    (refusal.test(text) || (feature === "temperature" && temperatureRefusal.test(text)))
  );
}

/**
 * Sorts a failed provider request into a kind the UI can explain: a bad key,
 * an unknown model, rate limiting, no connection, or the provider's own error.
 * For the ChatGPT plan, also a missing or expired sign-in, a reached plan
 * limit, or OpenAI refusing the sign-in. Jev's errors come sorted already.
 * For a local model, a request too long for its context window.
 * Answers (#29) reuse it for errors shown inside an Answer.
 */
export function classifyProviderError(error: unknown): ProviderError {
  const cause = RetryError.isInstance(error) ? error.lastError : error;
  const message = messageOf(cause);

  if (cause instanceof ConsentDeclinedError) return { kind: "consent-declined", message };
  if (cause instanceof LocalModelError) return { kind: cause.kind, message };
  if (contextOverflow(cause)) return { kind: "too-long", message };
  if (cause instanceof JevRequestError) return { kind: cause.kind, message };
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
