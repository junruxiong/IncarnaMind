/**
 * The model endpoint a ChatGPT plan is used through: the one OpenAI's Codex
 * CLI calls, `https://chatgpt.com/backend-api/codex/responses`. It speaks the
 * OpenAI Responses API with requirements of its own, and this adapter is the
 * only place that knows them:
 *
 * - every request carries the access token, the `chatgpt-account-id` header
 *   (from the sign-in's ID token), `OpenAI-Beta: responses=experimental` and
 *   an `originator` naming the calling app;
 * - the body must say `store: false` and `stream: true`, and must carry
 *   `instructions` (the system prompt goes there rather than into `input`);
 * - it rejects parameters the public API takes, such as `max_output_tokens`,
 *   so only the fields the Codex CLI sends are passed on;
 * - it only streams, so a non-streaming call is answered by collecting the
 *   stream into the JSON the AI SDK expects.
 *
 * Everything else is the AI SDK's OpenAI Responses model, given a custom
 * `fetch`. Answers (#29) get it from the model factory like any other model.
 *
 * Sources, checked 2026-10-07: the Codex CLI (github.com/openai/codex at
 * 24edd7b: `codex-rs/core/src/client.rs`, `codex-rs/model-provider/src/bearer_auth_provider.rs`,
 * `codex-rs/codex-api/src/api_bridge.rs`, `codex-rs/models-manager/models.json`)
 * and the MIT-licensed pi-ai package's "openai-codex" provider
 * (@earendil-works/pi-ai 1.0.4, `dist/api/openai-codex-responses.js`). The
 * header set and body rules follow them; no code is copied from either.
 */
import { createOpenAI } from "@ai-sdk/openai";
import { APICallError, wrapLanguageModel } from "ai";
import type { ChatGptPlanModel } from "../../api";
import { isRecord } from "../../errors";
import type { ChatLanguageModel } from "../models";
import { ChatGptPlanError } from "./errors";

/** The endpoint's base; the AI SDK appends `/responses`. */
export const CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex";

/** Where Questions go, for data-flow consent. The same whatever URL tests point the endpoint at. */
export const CHATGPT_SERVICE = { id: "https://chatgpt.com", name: "ChatGPT" } as const;

/**
 * Names the calling app to OpenAI, on the sign-in page and on every request.
 * IncarnaMind says who it is rather than posing as the Codex CLI.
 */
export const ORIGINATOR = "incarnamind";

/**
 * The models the endpoint accepts: those the Codex CLI lists to its users
 * (`codex-rs/models-manager/models.json`, visibility "list"), in its order of
 * priority. The first is the default.
 */
export const CHATGPT_PLAN_MODELS: readonly ChatGptPlanModel[] = [
  { id: "gpt-6.1-sol", name: "GPT-6.1 Sol" },
  { id: "gpt-6-astra", name: "GPT-6 Astra" },
  { id: "gpt-6-sol", name: "GPT-6 Sol" },
  { id: "gpt-6-luna", name: "GPT-6 Luna" },
  { id: "gpt-5.6-sol", name: "GPT-5.6 Sol" },
  { id: "gpt-5.6-terra", name: "GPT-5.6 Terra" },
  { id: "gpt-5.6-luna", name: "GPT-5.6 Luna" },
  { id: "gpt-5.5", name: "GPT-5.5" },
];

export const isChatGptPlanModel = (modelId: string) =>
  CHATGPT_PLAN_MODELS.some((model) => model.id === modelId);

/** What the endpoint needs from the sign-in. The official "Sign in with ChatGPT" can provide the same. */
export interface ChatGptCredentials {
  /**
   * A current access token and the ChatGPT account it is for, refreshed first
   * when it is about to expire. `rejected` is a token the endpoint refused: if
   * it is still the current one, it is refreshed even so. Throws
   * `ChatGptSignInRequiredError` when there is no sign-in or it can't be refreshed.
   */
  access(options?: { rejected?: string }): Promise<{ accessToken: string; accountId: string }>;
}

/** Used when a request has no system prompt: the endpoint requires instructions. */
const DEFAULT_INSTRUCTIONS = "You are a helpful assistant.";

/** The body fields the Codex CLI sends (besides those set below); anything else is dropped. */
const FORWARDED_FIELDS = [
  "model",
  "input",
  "tools",
  "tool_choice",
  "parallel_tool_calls",
  "reasoning",
  "text",
  "include",
  "service_tier",
  "prompt_cache_key",
] as const;

/** Output items the AI SDK's non-streaming parser knows; others are left out of collected answers. */
const COLLECTED_ITEM_TYPES = new Set(["message", "reasoning", "function_call", "custom_tool_call"]);

/** A leading system or developer message, which becomes the instructions. */
function instructionText(item: unknown): string | null {
  if (!isRecord(item) || (item.type !== undefined && item.type !== "message")) return null;
  if (item.role !== "system" && item.role !== "developer") return null;
  if (typeof item.content === "string") return item.content;
  if (!Array.isArray(item.content)) return null;
  const parts = item.content.map((part) =>
    isRecord(part) && typeof part.text === "string" ? part.text : null,
  );
  return parts.every((part) => part !== null) ? parts.join("") : null;
}

/** Turns the AI SDK's Responses API body into one the Codex endpoint accepts. */
export function shapeCodexRequest(body: Record<string, unknown>): Record<string, unknown> {
  const shaped: Record<string, unknown> = {};
  for (const field of FORWARDED_FIELDS) {
    if (body[field] !== undefined) shaped[field] = body[field];
  }
  const input = Array.isArray(body.input) ? [...body.input] : [];
  const instructions: string[] = [];
  for (let text = instructionText(input[0]); text !== null; text = instructionText(input[0])) {
    instructions.push(text);
    input.shift();
  }
  const include = Array.isArray(body.include) ? body.include : [];
  return {
    ...shaped,
    instructions: instructions.join("\n\n").trim() || DEFAULT_INSTRUCTIONS,
    input,
    store: false,
    stream: true,
    include: [...new Set([...include, "reasoning.encrypted_content"])],
  };
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function bodyText(body: RequestInit["body"]): string {
  if (typeof body === "string") return body;
  if (body instanceof Uint8Array) return new TextDecoder().decode(body);
  throw new Error("The ChatGPT plan adapter expects a JSON request body.");
}

/**
 * The type or code and message of an error body or event: `{ error: {…} }`,
 * or the event itself. The endpoint names usage limits in `type`; events may use `code`.
 */
function errorDetails(value: unknown): { type: string | null; message: string | null } {
  const error = isRecord(value) && isRecord(value.error) ? value.error : value;
  if (!isRecord(error)) return { type: null, message: null };
  const pick = (key: string) => (typeof error[key] === "string" ? (error[key] as string) : null);
  const type = pick("type");
  return {
    type: type === null || type === "error" ? pick("code") : type,
    message: pick("message"),
  };
}

const PLAN_LIMIT_TYPES = new Set(["usage_limit_reached", "usage_not_included"]);

/** "Your ChatGPT plan's usage limit is reached (plus plan). It resets at 2026-10-07T12:00:00.000Z." */
function planLimitMessage(value: unknown): string {
  const error = isRecord(value) && isRecord(value.error) ? value.error : {};
  const { type, message } = errorDetails(value);
  const plan = typeof error.plan_type === "string" ? ` (${error.plan_type} plan)` : "";
  const resets =
    typeof error.resets_at === "number"
      ? ` It resets at ${new Date(error.resets_at * 1000).toISOString()}.`
      : "";
  const what =
    type === "usage_not_included"
      ? "Your ChatGPT plan doesn't include this use"
      : "Your ChatGPT plan's usage limit is reached";
  return `${(message ?? what).replace(/\.$/, "")}${plan}.${resets}`;
}

/**
 * Sorts out refusals that only make sense for a ChatGPT plan and throws them
 * as `ChatGptPlanError`. Other failures go back to the AI SDK unchanged.
 */
async function checkRefusal(response: Response): Promise<Response> {
  const text = await response.text();
  const parsed = parseJson(text);
  const { status } = response;
  if (status === 401 || status === 403) {
    const { message } = errorDetails(parsed);
    throw new ChatGptPlanError(
      "blocked",
      `OpenAI refused the request (${status}): ${message ?? (text.slice(0, 300) || response.statusText)}`,
      status,
    );
  }
  if (status === 429 && PLAN_LIMIT_TYPES.has(errorDetails(parsed).type ?? "")) {
    throw new ChatGptPlanError("plan-limit", planLimitMessage(parsed), status);
  }
  return new Response(text, {
    status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

/** The JSON payload of each server-sent event in a stream. */
async function* serverSentEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const decoder = new TextDecoder();
  let buffer = "";
  const eventsIn = function* (chunk: string) {
    for (const block of chunk.split(/\r?\n\r?\n/)) {
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data && data !== "[DONE]") {
        const event = parseJson(data);
        if (event !== undefined) yield event;
      }
    }
  };
  for await (const chunk of body) {
    buffer += decoder.decode(chunk, { stream: true });
    const end = Math.max(buffer.lastIndexOf("\n\n"), buffer.lastIndexOf("\r\n\r\n"));
    if (end === -1) continue;
    yield* eventsIn(buffer.slice(0, end));
    buffer = buffer.slice(end).replace(/^(\r?\n)+/, "");
  }
  buffer += decoder.decode();
  yield* eventsIn(buffer);
}

/** Gives message text parts the `annotations` list the AI SDK's parser requires. */
function normalizeItem(item: Record<string, unknown>): Record<string, unknown> {
  if (item.type !== "message" || !Array.isArray(item.content)) return item;
  return {
    ...item,
    content: item.content.map((part) =>
      isRecord(part) && part.type === "output_text" && !Array.isArray(part.annotations)
        ? { ...part, annotations: [] }
        : part,
    ),
  };
}

/**
 * Reads a streamed answer to its end and returns it as the JSON a
 * non-streaming Responses API call returns. Items arrive one by one in
 * `response.output_item.done` events; the final event carries the usage.
 */
async function collectStream(response: Response, url: string): Promise<Response> {
  if (!response.body) throw new Error("The ChatGPT plan's endpoint sent no answer.");
  const items = new Map<number, Record<string, unknown>>();
  let final: Record<string, unknown> | null = null;
  for await (const event of serverSentEvents(response.body)) {
    if (!isRecord(event)) continue;
    if (event.type === "response.output_item.done" && isRecord(event.item)) {
      const index = typeof event.output_index === "number" ? event.output_index : items.size;
      items.set(index, event.item);
    } else if (
      (event.type === "response.completed" || event.type === "response.incomplete") &&
      isRecord(event.response)
    ) {
      final = event.response;
    } else if (event.type === "response.failed" || event.type === "error") {
      const source = event.type === "response.failed" ? event.response : event;
      const { type, message } = errorDetails(source);
      if (PLAN_LIMIT_TYPES.has(type ?? "")) {
        throw new ChatGptPlanError("plan-limit", planLimitMessage(source), 429);
      }
      throw new APICallError({
        message: message ?? "The ChatGPT plan's endpoint failed to answer.",
        url,
        requestBodyValues: {},
        statusCode: 502,
        responseBody: JSON.stringify(event),
        isRetryable: false,
      });
    }
  }
  if (!final) {
    throw new APICallError({
      message: "The ChatGPT plan's endpoint stopped before the answer was complete.",
      url,
      requestBodyValues: {},
      statusCode: 502,
      isRetryable: true,
    });
  }
  const streamed = [...items.entries()].sort(([a], [b]) => a - b).map(([, item]) => item);
  const finalOutput = Array.isArray(final.output) ? final.output.filter(isRecord) : [];
  const output = (finalOutput.length > 0 ? finalOutput : streamed)
    .filter((item) => COLLECTED_ITEM_TYPES.has(String(item.type)))
    .map(normalizeItem);
  return new Response(JSON.stringify({ ...final, output }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

/**
 * The `fetch` the AI SDK's Responses model uses for the ChatGPT plan: signs
 * the request, reshapes the body, retries once with a refreshed token after a
 * 401, and turns refusals into `ChatGptPlanError`.
 */
export function createCodexFetch(
  credentials: ChatGptCredentials,
  fetcher: typeof globalThis.fetch = globalThis.fetch,
): typeof globalThis.fetch {
  const codexFetch = async (input: Parameters<typeof globalThis.fetch>[0], init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const original = parseJson(bodyText(init?.body));
    if (!isRecord(original)) throw new Error("The ChatGPT plan adapter expects a JSON object.");
    const streaming = original.stream === true;
    const body = JSON.stringify(shapeCodexRequest(original));

    const send = async (rejected?: string) => {
      const { accessToken, accountId } = await credentials.access(rejected ? { rejected } : {});
      const headers = new Headers(init?.headers);
      headers.set("authorization", `Bearer ${accessToken}`);
      headers.set("chatgpt-account-id", accountId);
      headers.set("openai-beta", "responses=experimental");
      headers.set("originator", ORIGINATOR);
      headers.set("accept", "text/event-stream");
      headers.set("content-type", "application/json");
      headers.delete("content-length");
      return {
        accessToken,
        response: await fetcher(url, { ...init, method: "POST", headers, body }),
      };
    };

    const first = await send();
    let { response } = first;
    if (response.status === 401) {
      // The token may have been revoked early: refresh once and retry. A refusal after that is final.
      await response.body?.cancel();
      ({ response } = await send(first.accessToken));
    }
    if (!response.ok) return checkRefusal(response);
    return streaming ? response : collectStream(response, url);
  };
  return codexFetch as typeof globalThis.fetch;
}

/** The ChatGPT plan's model, as an AI SDK language model. */
export function createCodexChatModel(options: {
  baseUrl: string;
  modelId: string;
  credentials: ChatGptCredentials;
  fetch?: typeof globalThis.fetch;
}): ChatLanguageModel {
  const provider = createOpenAI({
    name: "chatgpt",
    baseURL: options.baseUrl,
    // Replaced by the sign-in's access token in every request; never read from the environment.
    apiKey: "chatgpt-sign-in",
    fetch: createCodexFetch(options.credentials, options.fetch),
  });
  return wrapLanguageModel({
    model: provider.responses(options.modelId),
    middleware: {
      specificationVersion: "v4",
      // Nothing is stored at OpenAI, so earlier steps are sent in full rather than by reference.
      transformParams: async ({ params }) => ({
        ...params,
        providerOptions: {
          ...params.providerOptions,
          openai: { ...params.providerOptions?.openai, store: false },
        },
      }),
    },
  });
}
