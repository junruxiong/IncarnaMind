/**
 * Requests that use the User's ChatGPT plan: OpenAI's public Responses API,
 * `POST https://api.openai.com/v1/responses`, with the access token of a
 * Sign in with ChatGPT sign-in (see ./signIn). The route has rules of its
 * own (https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations),
 * and this adapter is the only place that knows them:
 *
 * - every request says `store: false` and `stream: true`, and carries its
 *   whole history in `input`;
 * - fields the route rejects (`max_output_tokens`, `temperature`,
 *   `metadata`, `previous_response_id`…) are left out: only the fields below
 *   are passed on;
 * - system messages are rejected, so a leading one becomes `instructions`
 *   and any other a developer message;
 * - function and custom Tools go in a namespace, and calls of them sent back
 *   name it, as the route asks;
 * - it only streams, so a non-streaming call is answered by collecting the
 *   stream into the JSON the AI SDK expects;
 * - its refusals (usage limits, an ineligible plan, a region) become
 *   `ChatGptPlanError`, also when one arrives after the stream has begun.
 *
 * Everything else is the AI SDK's OpenAI Responses model, given a custom
 * `fetch`. Answers (#29) get it from the model factory like any other model.
 */
import { createOpenAI } from "@ai-sdk/openai";
import { APICallError } from "ai";
import type { ChatGptPlanModel } from "../../api";
import { isRecord } from "../../errors";
import type { ChatLanguageModel } from "../models";
import { withoutStorage } from "../responsesStore";
import { retryAfterMs } from "../retryAfter";
import { ChatGptPlanError } from "./errors";
import { CHATGPT_PLAN_RESOURCE, type ChatGptCredentials } from "./signIn";

/** The API's base; the AI SDK appends `/responses`. */
export const CHATGPT_PLAN_BASE_URL = CHATGPT_PLAN_RESOURCE;

/**
 * Models to offer before the account's own list can be read (before signing
 * in, or when OpenAI can't be reached), in OpenAI's order. The first is the
 * default. Once signed in, the account's list from `/v1/models` replaces it.
 */
export const KNOWN_PLAN_MODELS: readonly ChatGptPlanModel[] = [
  { id: "gpt-6.1-sol", name: "GPT-6.1 Sol" },
  { id: "gpt-6-astra", name: "GPT-6 Astra" },
  { id: "gpt-6-luna", name: "GPT-6 Luna" },
  { id: "gpt-5.5", name: "GPT-5.5" },
];

/** The namespace function Tools are grouped in, as OpenAI's own clients name it. */
const TOOL_NAMESPACE = "functions";

/** The body fields passed on (besides those set below); anything else is dropped. */
const FORWARDED_FIELDS = [
  "model",
  "input",
  "tools",
  "tool_choice",
  "parallel_tool_calls",
  "reasoning",
  "text",
  "include",
  "prompt_cache_key",
] as const;

/** Output items the AI SDK's non-streaming parser knows; others are left out of collected answers. */
const COLLECTED_ITEM_TYPES = new Set(["message", "reasoning", "function_call", "custom_tool_call"]);

/** A system or developer message's text, or null when it isn't one. */
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

const isNamespacedTool = (tool: unknown) =>
  isRecord(tool) && (tool.type === "function" || tool.type === "custom");

/** Function and custom Tools grouped in one namespace; hosted Tools stay as they are. */
function groupTools(tools: unknown): { tools: unknown[]; names: Set<string> } | null {
  if (!Array.isArray(tools) || tools.length === 0) return null;
  const grouped = tools.filter(isNamespacedTool);
  const names = new Set(
    grouped.flatMap((tool) => (isRecord(tool) && typeof tool.name === "string" ? [tool.name] : [])),
  );
  if (grouped.length === 0) return { tools, names };
  const others = tools.filter((tool) => !isNamespacedTool(tool));
  return {
    tools: [
      { type: "namespace", name: TOOL_NAMESPACE, description: "", tools: grouped },
      ...others,
    ],
    names,
  };
}

/** Turns the AI SDK's Responses API body into one the ChatGPT plan's route accepts. */
export function shapePlanRequest(body: Record<string, unknown>): Record<string, unknown> {
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
  const grouped = groupTools(body.tools);
  const reshaped = input.map((item) => {
    if (!isRecord(item)) return item;
    // A system message further on: the route takes developer messages only.
    if (item.role === "system" && (item.type === undefined || item.type === "message")) {
      return { ...item, role: "developer" };
    }
    // A call of a grouped Tool, sent back as history, names its namespace as the model did.
    if (
      (item.type === "function_call" || item.type === "custom_tool_call") &&
      item.namespace === undefined &&
      typeof item.name === "string" &&
      grouped?.names.has(item.name)
    ) {
      return { ...item, namespace: TOOL_NAMESPACE };
    }
    return item;
  });
  const include = Array.isArray(body.include) ? body.include : [];
  const text = instructions.join("\n\n").trim();
  return {
    ...shaped,
    ...(grouped && { tools: grouped.tools }),
    ...(text && { instructions: text }),
    input: reshaped,
    store: false,
    stream: true,
    // With nothing stored at OpenAI, reasoning goes back encrypted in the next request.
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
 * What a refusal says: the Responses API's `{ error: { code, message, param } }`,
 * an event's own fields, or the `{ detail }` the route answers with before a
 * request starts. The code is null when there is none.
 */
function refusalOf(value: unknown): { code: string | null; message: string | null } {
  if (isRecord(value) && typeof value.detail === "string") {
    return { code: null, message: value.detail };
  }
  const error =
    isRecord(value) && isRecord(value.error)
      ? value.error
      : isRecord(value) && isRecord(value.response) && isRecord(value.response.error)
        ? value.response.error
        : value;
  if (!isRecord(error)) return { code: null, message: null };
  const pick = (key: string) => (typeof error[key] === "string" ? (error[key] as string) : null);
  const param = pick("param");
  const message = pick("message");
  return {
    code: pick("code") ?? pick("type"),
    message: param && message ? `${message} (${param})` : message,
  };
}

/** Codes that mean the plan's or this app's usage limit is reached. */
const PLAN_LIMIT_CODES = new Set(["subscription_sharing_usage_limit_exceeded"]);

/** Codes that mean the plan can't be used here, whatever is asked. */
const BLOCKED_CODES = new Set([
  "subscription_sharing_user_not_eligible",
  "subscription_sharing_route_not_supported",
  "chatpass_v2_scope_not_authorized",
  "chatpass_v2_invalid_authorization_context",
]);

/** Codes that mean OpenAI couldn't check the plan just now: worth trying again later. */
const PASSING_CODES = new Set([
  "subscription_sharing_usage_unavailable",
  "subscription_sharing_user_unavailable",
]);

const PLAN_LIMIT_MESSAGE =
  "Your ChatGPT plan's usage limit, or the one you set for IncarnaMind, is reached.";

/** The refusal as a `ChatGptPlanError`, or null when it isn't one only a ChatGPT plan gets. */
function planError(
  status: number,
  value: unknown,
  resetsAt: number | null = null,
): ChatGptPlanError | null {
  const { code, message } = refusalOf(value);
  if (code && PLAN_LIMIT_CODES.has(code)) {
    return new ChatGptPlanError("plan-limit", PLAN_LIMIT_MESSAGE, status, resetsAt);
  }
  if ((code && BLOCKED_CODES.has(code)) || status === 401 || status === 403) {
    const said = message ?? code ?? `HTTP ${status}`;
    return new ChatGptPlanError(
      "blocked",
      code === "subscription_sharing_user_not_eligible"
        ? `This ChatGPT plan can't be used in other apps. (${said})`
        : `OpenAI didn't let IncarnaMind use your ChatGPT plan (${status}): ${said}`,
      status,
    );
  }
  return null;
}

/**
 * Sorts out refusals that only make sense for a ChatGPT plan and throws them
 * as `ChatGptPlanError`. Others go back to the AI SDK as they came.
 */
async function checkRefusal(response: Response, url: string): Promise<Response> {
  const text = await response.text();
  const parsed = parseJson(text);
  const { status } = response;
  // When the limit resets, if OpenAI says, so tagging waits no longer than it must (#180).
  const wait = retryAfterMs(response.headers);
  const refused = planError(status, parsed, wait === null ? null : Date.now() + wait);
  if (refused) throw refused;
  const { code, message } = refusalOf(parsed);
  if (status === 503 || (code && PASSING_CODES.has(code))) {
    throw new APICallError({
      message: message ?? "OpenAI couldn't check your ChatGPT plan just now.",
      url,
      requestBodyValues: {},
      statusCode: 503,
      responseBody: text,
      isRetryable: true,
    });
  }
  return new Response(text, {
    status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

/** The JSON payload of one server-sent event, or undefined when it has none. */
function eventOf(block: string): unknown {
  const data = block
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  return data && data !== "[DONE]" ? parseJson(data) : undefined;
}

/** The JSON payload of each server-sent event in a stream. */
async function* serverSentEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const decoder = new TextDecoder();
  let buffer = "";
  const eventsIn = function* (chunk: string) {
    for (const block of chunk.split(/\r?\n\r?\n/)) {
      const event = eventOf(block);
      if (event !== undefined) yield event;
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

/** A failed stream's refusal, when only a ChatGPT plan gets it (e.g. the usage limit mid-answer). */
function failedEventError(event: Record<string, unknown>): ChatGptPlanError | null {
  if (event.type !== "response.failed" && event.type !== "error") return null;
  const { code } = refusalOf(event);
  return code && PLAN_LIMIT_CODES.has(code)
    ? new ChatGptPlanError("plan-limit", PLAN_LIMIT_MESSAGE, 429)
    : null;
}

/**
 * Passes a streamed answer on event by event, but ends it with
 * `ChatGptPlanError` instead of the event when the plan's usage limit stops
 * it after it has begun, so the Answer says so rather than "it failed".
 */
function watchStream(response: Response): Response {
  const body = response.body;
  if (!body) return response;
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  let stopped = false;
  /** Forwards one event, or ends the stream with the plan's refusal. */
  const pass = (block: string, controller: TransformStreamDefaultController<Uint8Array>) => {
    const event = eventOf(block);
    const refused = isRecord(event) ? failedEventError(event) : null;
    if (refused) {
      stopped = true;
      controller.error(refused);
    } else {
      controller.enqueue(encoder.encode(`${block}\n\n`));
    }
  };
  const events = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      buffer += decoder.decode(chunk, { stream: true });
      for (let end = buffer.search(/\r?\n\r?\n/); end !== -1 && !stopped; ) {
        const separator = /^\r?\n\r?\n/.exec(buffer.slice(end))?.[0] ?? "\n\n";
        const block = buffer.slice(0, end);
        buffer = buffer.slice(end + separator.length);
        if (block.trim()) pass(block, controller);
        end = buffer.search(/\r?\n\r?\n/);
      }
    },
    flush(controller) {
      buffer += decoder.decode();
      if (buffer.trim() && !stopped) pass(buffer, controller);
    },
  });
  return new Response(body.pipeThrough(events), {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
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
  if (!response.body) throw new Error("OpenAI sent no answer.");
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
      const refused = failedEventError(event);
      if (refused) throw refused;
      const { code, message } = refusalOf(event);
      throw new APICallError({
        message: message ?? "OpenAI failed to answer.",
        url,
        requestBodyValues: {},
        statusCode: code && PASSING_CODES.has(code) ? 503 : 502,
        responseBody: JSON.stringify(event),
        isRetryable: code !== null && PASSING_CODES.has(code),
      });
    }
  }
  if (!final) {
    throw new APICallError({
      message: "OpenAI stopped before the answer was complete.",
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
function createPlanFetch(
  credentials: ChatGptCredentials,
  fetcher: typeof globalThis.fetch = globalThis.fetch,
): typeof globalThis.fetch {
  const planFetch = async (input: Parameters<typeof globalThis.fetch>[0], init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const original = parseJson(bodyText(init?.body));
    if (!isRecord(original)) throw new Error("The ChatGPT plan adapter expects a JSON object.");
    const streaming = original.stream === true;
    const body = JSON.stringify(shapePlanRequest(original));

    const send = async (rejected?: string) => {
      const { accessToken } = await credentials.access(rejected ? { rejected } : {});
      const headers = new Headers(init?.headers);
      headers.set("authorization", `Bearer ${accessToken}`);
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
    if (!response.ok) return checkRefusal(response, url);
    return streaming ? watchStream(response) : collectStream(response, url);
  };
  return planFetch as typeof globalThis.fetch;
}

/** The ChatGPT plan's model, as an AI SDK language model. */
export function createPlanChatModel(options: {
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
    fetch: createPlanFetch(options.credentials, options.fetch),
  });
  // Nothing is stored at OpenAI, so earlier steps are sent in full rather than by reference.
  return withoutStorage(provider.responses(options.modelId));
}

/**
 * The models the signed-in account can use, for its model picker, in
 * OpenAI's order: `GET /v1/models` with the plan's token lists them as
 * `{ models: [{ slug, display_name, visibility }] }`; those meant for a
 * picker have `visibility: "list"`. Throws when the list can't be read.
 */
export async function listPlanModels(options: {
  baseUrl: string;
  credentials: ChatGptCredentials;
  fetch?: typeof globalThis.fetch;
  signal?: AbortSignal;
}): Promise<ChatGptPlanModel[]> {
  const fetcher = options.fetch ?? globalThis.fetch;
  const get = async (rejected?: string) => {
    const { accessToken } = await options.credentials.access(rejected ? { rejected } : {});
    const response = await fetcher(`${options.baseUrl}/models`, {
      headers: { authorization: `Bearer ${accessToken}`, accept: "application/json" },
      ...(options.signal && { signal: options.signal }),
    });
    return { accessToken, response };
  };
  const first = await get();
  let { response } = first;
  if (response.status === 401) {
    await response.body?.cancel();
    ({ response } = await get(first.accessToken));
  }
  if (!response.ok) throw new Error(`OpenAI's model list answered ${response.status}.`);
  const body: unknown = await response.json();
  const models = isRecord(body) && Array.isArray(body.models) ? body.models : [];
  return models.flatMap((model): ChatGptPlanModel[] => {
    if (!isRecord(model) || typeof model.slug !== "string") return [];
    if (model.visibility !== undefined && model.visibility !== "list") return [];
    const name = typeof model.display_name === "string" ? model.display_name : model.slug;
    return [{ id: model.slug, name }];
  });
}
