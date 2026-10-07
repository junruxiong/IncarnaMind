/**
 * Chat with a local model through Ollama's own API, `/api/chat`, rather than
 * its OpenAI-compatible `/v1`, which can't set the context window: Ollama
 * then uses its default (4,096 tokens on a computer with less than 24 GiB of
 * GPU memory) and silently cuts longer requests (docs/research/ollama-integration.md).
 *
 * The AI SDK provider is `ollama-ai-provider-v2`, wrapped as the ChatGPT
 * plan's model is (see ./chatgpt/codexEndpoint). Every request carries the
 * model's fixed settings (see ./ollamaModels):
 *
 * - Middleware sets what the provider supports, in its `options`:
 *   `num_ctx` (the same on every request, so Ollama never reloads the model),
 *   the request's temperature and top-p, `num_predict` (the request's output
 *   limit, at most the model's output cap), and `think`. The provider sends
 *   temperature, top-p and the output limit as top-level fields too, which
 *   `/api/chat` ignores.
 * - A `fetch` adds what it doesn't: `keep_alive`, and `truncate: false`, so
 *   a request longer than the window is refused ("exceeds the available
 *   context size") instead of cut; the engine keeps requests within it (see
 *   ../answers/window). It drops the fields `/api/chat` doesn't have,
 *   including `tool_choice`: a request that may call no Tool is sent without
 *   its Tools instead. And it gives Ollama's errors (`{"error": "…"}`) the
 *   shape the provider reads, so they keep their message.
 */
import { wrapLanguageModel } from "ai";
import { createOllama } from "ollama-ai-provider-v2";
import { isRecord } from "../errors";
import type { ChatLanguageModel } from "./models";
import type { OllamaModelSettings } from "./ollamaModels";

/** Fields the provider sends that `/api/chat` doesn't read (their values are in `options`). */
const NOT_READ = ["temperature", "top_p", "max_output_tokens", "tool_choice"] as const;

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function bodyText(body: RequestInit["body"]): string | null {
  if (typeof body === "string") return body;
  if (body instanceof Uint8Array) return new TextDecoder().decode(body);
  return null;
}

/** The `options` of a request: the model's settings, with the request's own sampling and output limit. */
export function chatOptions(
  settings: OllamaModelSettings,
  request: { temperature?: number; topP?: number; maxOutputTokens?: number },
): Record<string, number> {
  return {
    num_ctx: settings.numCtx,
    num_predict:
      request.maxOutputTokens === undefined
        ? settings.outputTokens
        : Math.min(request.maxOutputTokens, settings.outputTokens),
    ...(request.temperature !== undefined && { temperature: request.temperature }),
    ...(request.topP !== undefined && { top_p: request.topP }),
  };
}

/** A chat request's body as `/api/chat` reads it: with `keep_alive` and `truncate: false`. */
export function shapeChatRequest(
  body: Record<string, unknown>,
  settings: OllamaModelSettings,
): Record<string, unknown> {
  const shaped: Record<string, unknown> = { ...body };
  for (const field of NOT_READ) delete shaped[field];
  // A request that may call no Tool (the last step of the loop, which must write) goes without them.
  if (body.tool_choice === "none" || (Array.isArray(body.tools) && body.tools.length === 0)) {
    delete shaped.tools;
  }
  return { ...shaped, keep_alive: settings.keepAlive, truncate: false };
}

/** Ollama's error text, from its `{"error": "…"}`, which may itself hold the runner's JSON error. */
export function ollamaErrorMessage(body: unknown): string | null {
  const error = isRecord(body) ? body.error : undefined;
  if (isRecord(error) && typeof error.message === "string") return error.message;
  if (typeof error !== "string") return null;
  return ollamaErrorMessage(parseJson(error)) ?? error;
}

/** Gives a failed response the `{ error: { message } }` shape the provider reads. */
async function withErrorShape(response: Response): Promise<Response> {
  const text = await response.text();
  const message =
    ollamaErrorMessage(parseJson(text)) ?? (text.slice(0, 500) || response.statusText);
  return new Response(JSON.stringify({ error: { message } }), {
    status: response.status,
    statusText: response.statusText,
    headers: { "content-type": "application/json" },
  });
}

/** The `fetch` the provider uses: shapes chat requests, and failed responses. */
export function createOllamaFetch(
  settings: OllamaModelSettings,
  fetcher: typeof globalThis.fetch = globalThis.fetch,
): typeof globalThis.fetch {
  const ollamaFetch = async (input: Parameters<typeof globalThis.fetch>[0], init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    let next = init;
    const text = bodyText(init?.body);
    const body = text === null ? undefined : parseJson(text);
    if (new URL(url).pathname.endsWith("/api/chat") && isRecord(body)) {
      const headers = new Headers(init?.headers);
      headers.delete("content-length");
      next = { ...init, headers, body: JSON.stringify(shapeChatRequest(body, settings)) };
    }
    const response = await fetcher(url, next);
    return response.ok ? response : withErrorShape(response);
  };
  return ollamaFetch as typeof globalThis.fetch;
}

/** A model in Ollama, as an AI SDK language model that speaks `/api/chat`. */
export function createOllamaChatModel(options: {
  /** The Ollama server, e.g. "http://127.0.0.1:11434" (without "/api"). */
  baseUrl: string;
  modelId: string;
  settings: OllamaModelSettings;
  fetch?: typeof globalThis.fetch;
}): ChatLanguageModel {
  const { settings } = options;
  const provider = createOllama({
    baseURL: `${options.baseUrl}/api`,
    fetch: createOllamaFetch(settings, options.fetch),
  });
  return wrapLanguageModel({
    model: provider(options.modelId),
    middleware: {
      specificationVersion: "v4",
      transformParams: async ({ params }) => ({
        ...params,
        providerOptions: {
          ...params.providerOptions,
          ollama: {
            ...params.providerOptions?.ollama,
            think: settings.think,
            options: chatOptions(settings, params),
          },
        },
      }),
    },
  });
}
