import { APICallError, RetryError, UnsupportedFunctionalityError } from "ai";
import type { ProviderError, ProviderErrorKind, SignInRouteProblem } from "../api";
import { ConsentDeclinedError, TranslatableError } from "../errors";
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

export class OllamaNotRunningError extends TranslatableError {
  override name = "OllamaNotRunningError";
  constructor() {
    super({ key: "error.provider.ollamaNotRunning" });
  }
}

/** A request could pass the User's monthly cap, so it wasn't sent (see ../cost/meter). */
export class CostCapError extends Error {
  override name = "CostCapError";
  constructor() {
    super("The monthly cap would be passed, so nothing was sent.");
  }
}

/**
 * A plan's own allowance stopped a request: an Ollama Cloud account's usage is
 * used up. (The ChatGPT plan's limit is `ChatGptPlanError`.)
 */
export class PlanLimitError extends Error {
  override name = "PlanLimitError";
  constructor(
    message: string,
    /** When the limit resets (epoch milliseconds), when the provider says. */
    readonly resetsAt: number | null = null,
  ) {
    super(message);
  }
}

/**
 * A route signed in with the User's own account (a Claude account, Copilot)
 * stopped, for one of the reasons its own lines and ways back explain (./signInRoutes).
 */
export class SignInRouteError extends Error {
  override name = "SignInRouteError";
  constructor(
    readonly problem: Exclude<SignInRouteProblem, "sign-in-cancelled">,
    message: string,
    /** When the plan's limit resets (epoch milliseconds), when the provider says. */
    readonly resetsAt: number | null = null,
  ) {
    super(message);
  }
}

const ROUTE_KINDS: Record<SignInRouteError["problem"], ProviderErrorKind> = {
  "signed-out": "not-signed-in",
  "sign-in-expired": "not-signed-in",
  refused: "blocked",
  "plan-limit": "plan-limit",
  "unexpected-response": "provider",
  "no-plan": "blocked",
  "policy-blocked": "blocked",
  "model-unavailable": "model",
  "rate-limited": "rate-limit",
};

/** When a reached limit resets, as ISO 8601, from an error that says (epoch milliseconds). */
function resetsAtOf(error: Error): { resetsAt?: string } {
  const at = "resetsAt" in error && typeof error.resetsAt === "number" ? error.resetsAt : null;
  return at === null ? {} : { resetsAt: new Date(at).toISOString() };
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
 * limit, or OpenAI refusing the sign-in; for a route signed in with the
 * User's own account (Claude, Copilot), its own problems (`signInRoute`); for an Ollama Cloud model, its
 * used-up allowance; for any priced model, the monthly cap. Jev's errors come sorted already.
 * For a local model, a request too long for its context window.
 * Answers (#29) reuse it for errors shown inside an Answer.
 */
export function classifyProviderError(error: unknown): ProviderError {
  const retried = RetryError.isInstance(error) ? error.lastError : error;
  // A plan's refusal that ends a stream arrives wrapped in the AI SDK's own error.
  const cause =
    retried instanceof Error &&
    (retried.cause instanceof ChatGptPlanError ||
      retried.cause instanceof PlanLimitError ||
      retried.cause instanceof SignInRouteError)
      ? retried.cause
      : retried;
  const message = messageOf(cause);

  if (cause instanceof ConsentDeclinedError) return { kind: "consent-declined", message };
  if (cause instanceof LocalModelError) return { kind: cause.kind, message };
  if (cause instanceof OllamaNotRunningError) {
    return {
      kind: "ollama-not-running",
      message,
      ...(cause.wording && { wording: cause.wording }),
    };
  }
  if (contextOverflow(cause)) return { kind: "too-long", message };
  if (cause instanceof JevRequestError) return { kind: cause.kind, message };
  if (cause instanceof ChatGptSignInRequiredError) return { kind: "not-signed-in", message };
  if (cause instanceof ChatGptPlanError) return { kind: cause.kind, message, ...resetsAtOf(cause) };
  if (cause instanceof PlanLimitError) return { kind: "plan-limit", message, ...resetsAtOf(cause) };
  if (cause instanceof SignInRouteError) {
    return {
      kind: ROUTE_KINDS[cause.problem],
      message,
      signInRoute: cause.problem,
      ...resetsAtOf(cause),
    };
  }
  if (cause instanceof CostCapError) return { kind: "cost-cap", message };
  // A plan's sign-in couldn't be refreshed for a passing reason (e.g. the provider's server failed).
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
  return {
    kind: "unknown",
    message,
    ...(cause instanceof TranslatableError && cause.wording && { wording: cause.wording }),
  };
}
