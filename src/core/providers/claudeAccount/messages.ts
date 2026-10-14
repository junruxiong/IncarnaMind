/**
 * Requests that use the User's Claude plan: Anthropic's Messages API with the
 * access token of a Claude account sign-in (see ./account). This adapter is
 * the only place that knows how they differ from an API key's:
 *
 * - the token goes as a bearer token, and no API key is sent;
 * - they identify as IncarnaMind, and carry what the route's profile adds
 *   (./config: system blocks first, headers);
 * - a token the API refuses is refreshed once and the request retried;
 * - refusals become `ClaudeAccountError`: the plan's usage limit, a refused
 *   sign-in, and an answer the route doesn't know.
 *
 * Everything else is the AI SDK's Anthropic model, given a custom `fetch`.
 */
import { createAnthropic } from "@ai-sdk/anthropic";
import { version } from "../../../../package.json";
import { isRecord } from "../../errors";
import type { OAuthCredentials } from "../../oauthAccount";
import type { ChatLanguageModel } from "../models";
import { ClaudeAccountError } from "../providerErrors";
import type { RequestTransform } from "./config";

const USER_AGENT = `IncarnaMind/${version}`;

/** Set on a 429 when the plan's own limit, not a passing rate limit, stopped the request. */
const PLAN_LIMIT_STATUS = "anthropic-ratelimit-unified-status";
/** When the plan's limit resets, in seconds since the epoch. */
const PLAN_LIMIT_RESET = "anthropic-ratelimit-unified-reset";

/** The request's system blocks with `prefix` before them, unless they already start with it. */
function withSystemPrefix(
  body: Record<string, unknown>,
  prefix: readonly string[],
): Record<string, unknown> {
  const { system } = body;
  const blocks: unknown[] =
    typeof system === "string" && system !== ""
      ? [{ type: "text", text: system }]
      : Array.isArray(system)
        ? system
        : [];
  const starts = prefix.every((text, index) => {
    const block = blocks[index];
    return isRecord(block) && block.type === "text" && block.text === text;
  });
  if (prefix.length === 0 || starts) return body;
  return { ...body, system: [...prefix.map((text) => ({ type: "text", text })), ...blocks] };
}

const bodyText = (body: RequestInit["body"]): string | null =>
  typeof body === "string"
    ? body
    : body instanceof Uint8Array
      ? new TextDecoder().decode(body)
      : null;

/** The body to send: the request's, with the system prefix when it is a Messages request. */
function shapedBody(url: string, body: RequestInit["body"], prefix: readonly string[]) {
  if (prefix.length === 0 || !new URL(url).pathname.endsWith("/messages")) return body;
  const raw = bodyText(body);
  if (raw === null) return body;
  const parsed: unknown = JSON.parse(raw);
  if (!isRecord(parsed)) return body;
  return JSON.stringify(withSystemPrefix(parsed, prefix));
}

/** The flags of two `anthropic-beta` headers, each once. */
function joinedBetas(current: string | null, added: string): string {
  const flags = [current ?? "", added].flatMap((header) =>
    header
      .split(",")
      .map((flag) => flag.trim())
      .filter(Boolean),
  );
  return [...new Set(flags)].join(",");
}

async function errorMessage(response: Response): Promise<string> {
  const raw = await response.text();
  try {
    const body: unknown = JSON.parse(raw);
    if (isRecord(body) && isRecord(body.error) && typeof body.error.message === "string") {
      return body.error.message;
    }
  } catch {
    // Not Anthropic's error shape: its first words say what came back.
  }
  return raw.slice(0, 200) || response.statusText;
}

/**
 * A response the route can't use, as `ClaudeAccountError`; null for one the
 * AI SDK handles as it does an API key's (an answer, an unknown model, a
 * passing rate limit, a server error).
 */
async function refusalOf(response: Response): Promise<ClaudeAccountError | null> {
  const { status } = response;
  if (status === 401 || status === 403) {
    const message = await errorMessage(response);
    return new ClaudeAccountError(
      "refused",
      `Anthropic refused the Claude sign-in (${status}): ${message}`,
    );
  }
  if (status === 429 && response.headers.get(PLAN_LIMIT_STATUS) === "rejected") {
    const reset = Number(response.headers.get(PLAN_LIMIT_RESET));
    await response.body?.cancel();
    return new ClaudeAccountError(
      "plan-limit",
      "Your Claude plan's usage limit is reached.",
      Number.isFinite(reset) && reset > 0 ? reset * 1000 : null,
    );
  }
  // An API key's request would be answered: here it means the endpoint changed what it accepts.
  if (status === 400) {
    const message = await errorMessage(response);
    return new ClaudeAccountError(
      "unexpected-response",
      `Anthropic didn't accept the request (400): ${message}`,
    );
  }
  if (response.ok) {
    const type = response.headers.get("content-type") ?? "";
    if (!/application\/json|text\/event-stream/.test(type)) {
      await response.body?.cancel();
      return new ClaudeAccountError(
        "unexpected-response",
        `Anthropic answered with ${type || "no content type"}, not an answer.`,
      );
    }
  }
  return null;
}

function createAccountFetch(
  credentials: OAuthCredentials,
  request: RequestTransform,
  fetcher: typeof globalThis.fetch = globalThis.fetch,
): typeof globalThis.fetch {
  const accountFetch = async (
    input: Parameters<typeof globalThis.fetch>[0],
    init?: RequestInit,
  ) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const body = shapedBody(url, init?.body, request.systemBlocks ?? []);

    const send = async (rejected?: string) => {
      const { accessToken } = await credentials.access(rejected ? { rejected } : {});
      const headers = new Headers(init?.headers);
      headers.delete("x-api-key");
      headers.set("authorization", `Bearer ${accessToken}`);
      headers.set("user-agent", USER_AGENT);
      for (const [name, value] of Object.entries(request.headers ?? {})) {
        headers.set(
          name,
          name.toLowerCase() === "anthropic-beta" ? joinedBetas(headers.get(name), value) : value,
        );
      }
      headers.delete("content-length");
      return { accessToken, response: await fetcher(url, { ...init, headers, body }) };
    };

    const first = await send();
    let { response } = first;
    if (response.status === 401) {
      // The token may have been revoked early: refresh once and retry. A refusal after that is final.
      await response.body?.cancel();
      ({ response } = await send(first.accessToken));
    }
    const refused = await refusalOf(response);
    if (refused) throw refused;
    return response;
  };
  return accountFetch as typeof globalThis.fetch;
}

/** A model on the User's Claude plan, as an AI SDK language model. */
export function createClaudeAccountChatModel(options: {
  baseUrl: string;
  modelId: string;
  credentials: OAuthCredentials;
  /** What the route's profile adds to every request. */
  request: RequestTransform;
  fetch?: typeof globalThis.fetch;
}): ChatLanguageModel {
  return createAnthropic({
    baseURL: options.baseUrl,
    // Replaced by the sign-in's access token in every request; never read from the environment.
    apiKey: "claude-account",
    fetch: createAccountFetch(options.credentials, options.request, options.fetch),
  })(options.modelId);
}
