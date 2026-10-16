/**
 * Requests with a Claude account's token instead of an API key: the token
 * only to the API's origin, the profile's additions first, one refresh on a
 * refused token, and refusals as `ClaudeAccountError`. Everything else is
 * the AI SDK's Anthropic model, given a custom `fetch`.
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

/** A 400 that refuses the account's token for this request, rather than the request itself. */
const CREDENTIAL_REFUSAL = /credential|oauth/i;

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

function shapedBody(url: string, body: RequestInit["body"], prefix: readonly string[]) {
  if (prefix.length === 0 || !new URL(url).pathname.endsWith("/messages")) return body;
  const raw = bodyText(body);
  if (raw === null) return body;
  const parsed: unknown = JSON.parse(raw);
  if (!isRecord(parsed)) return body;
  return JSON.stringify(withSystemPrefix(parsed, prefix));
}

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

/** Null for what the AI SDK handles as it does an API key's: an answer, a passing rate limit, a server error. */
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
  // Other 400s, such as a prompt that is too long, are the request's own and stay the AI SDK's.
  if (status === 400) {
    const message = await errorMessage(response.clone());
    if (CREDENTIAL_REFUSAL.test(message)) {
      await response.body?.cancel();
      return new ClaudeAccountError(
        "refused",
        `Anthropic refused the Claude sign-in (400): ${message}`,
      );
    }
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

/** The `fetch` the AI SDK's Anthropic model is given; it signs requests to `apiBaseUrl`'s origin only. */
export function createAccountFetch(
  credentials: OAuthCredentials,
  request: RequestTransform,
  apiBaseUrl: string,
  fetcher: typeof globalThis.fetch = globalThis.fetch,
): typeof globalThis.fetch {
  const origin = new URL(apiBaseUrl).origin;
  const accountFetch = async (
    input: Parameters<typeof globalThis.fetch>[0],
    init?: RequestInit,
  ) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (new URL(url).origin !== origin) {
      throw new Error(`The Claude sign-in's token is only sent to ${origin}.`);
    }
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
      // A redirect would carry the token to another address.
      const response = await fetcher(url, { ...init, headers, body, redirect: "error" });
      return { accessToken, response };
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
  return accountFetch;
}

export function createClaudeAccountChatModel(options: {
  baseUrl: string;
  modelId: string;
  credentials: OAuthCredentials;
  request: RequestTransform;
  fetch?: typeof globalThis.fetch;
}): ChatLanguageModel {
  return createAnthropic({
    baseURL: options.baseUrl,
    // Replaced by the sign-in's access token in every request; never read from the environment.
    apiKey: "claude-account",
    fetch: createAccountFetch(options.credentials, options.request, options.baseUrl, options.fetch),
  })(options.modelId);
}
