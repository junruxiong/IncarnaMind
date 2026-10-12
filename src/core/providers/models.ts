/**
 * The seam between provider settings and the AI SDK: everything that builds a
 * chat model goes through a `ChatModelFactory`. The core uses the AI SDK
 * providers by default; tests inject AI SDK mock models instead (ADR-0005).
 * Where a hosted provider's API is, and what every request to it carries,
 * come from the catalog (see ./catalog).
 */
import { createAlibaba } from "@ai-sdk/alibaba";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createDeepSeek } from "@ai-sdk/deepseek";
import { createGoogle } from "@ai-sdk/google";
import { createMistral } from "@ai-sdk/mistral";
import { createMoonshotAI } from "@ai-sdk/moonshotai";
import { createOpenAI } from "@ai-sdk/openai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createXai } from "@ai-sdk/xai";
import { createZai } from "@ai-sdk/zai";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import type { LanguageModel } from "ai";
import type { ChatProviderKind } from "../api";
import { catalogFacts, modelsProviderOfKind } from "./catalog";
import { catalogProviderOfKind } from "./catalog/providers";
import { createPlanChatModel } from "./chatgpt/planEndpoint";
import type { ChatGptCredentials } from "./chatgpt/signIn";
import { endpointUrl } from "./kinds";
import { createOllamaChatModel } from "./ollamaChat";
import { DEFAULT_OLLAMA_SETTINGS, type OllamaModelSettings } from "./ollamaModels";
import { withCacheBreakpoints } from "./promptCache";
import { withoutStorage, withProviderOptions } from "./responsesStore";

/** A model object, never a model id string (which the AI SDK would send to its gateway). */
export type ChatLanguageModel = Exclude<LanguageModel, string>;

/** Everything needed to build a chat model. */
export interface ChatModelSpec {
  kind: ChatProviderKind;
  /**
   * The API's base URL: a hosted provider's endpoint from the catalog (null:
   * its first), the server's for "openai-compatible" and "ollama", the plan's
   * model endpoint for "chatgpt".
   */
  baseUrl: string | null;
  /** A hosted provider's endpoint id, to tell which region's models it is (its first when left out). */
  endpoint?: string | null;
  /**
   * "off": a small call that doesn't write the Answer (tagging, classifying),
   * which asks a model that thinks to skip it, where the provider lets it.
   * "summary": a call that writes the Answer, which asks a model that thinks
   * for its provider's summary of its reasoning, where there is one
   * (`reasoningSummaryOptions`).
   */
  thinking?: "off" | "summary";
  apiKey: string | null;
  modelId: string;
  /** "chatgpt" only: the User's ChatGPT sign-in, which signs (and refreshes) every request. */
  credentials?: ChatGptCredentials;
  /** "ollama" only: the settings every request to the model carries (see ./ollamaModels). */
  ollama?: OllamaModelSettings;
}

/**
 * A model's context window: what one request may hold, in tokens, output
 * included, and how much of it is kept for the output. For a local model it
 * is the window IncarnaMind sets, and its server refuses a longer request
 * rather than cut it; for a cloud model it is the model's own, from the
 * catalog (see ./capabilities). The Answer engine keeps each request within it.
 */
export interface ContextWindow {
  tokens: number;
  outputTokens: number;
}

export type ChatModelFactory = (spec: ChatModelSpec) => ChatLanguageModel;

function requireKey(spec: ChatModelSpec): string {
  // Never fall back to the AI SDK's environment variables: keys come from the keychain only.
  if (!spec.apiKey) throw new Error(`An API key is required for ${spec.kind}.`);
  return spec.apiKey;
}

function requireBaseUrl(spec: ChatModelSpec): string {
  if (!spec.baseUrl) throw new Error(`A server URL is required for ${spec.kind}.`);
  return spec.baseUrl;
}

/** A hosted provider's API: the spec's endpoint, else the catalog's first for its kind. */
const hostedUrl = (spec: ChatModelSpec): string =>
  spec.baseUrl ?? endpointUrl(spec.kind) ?? requireBaseUrl(spec);

/**
 * The AI SDK provider for each kind, with what every request to its provider
 * carries: for OpenAI and xAI, `store: false`, as the Responses API keeps
 * requests unless every one says not to; for a small call, thinking off; and
 * for the Answer's, the request for a summary of the model's reasoning.
 */
export const createAiSdkChatModel: ChatModelFactory = (spec) => {
  const request = catalogProviderOfKind(spec.kind)?.request;
  let model = providerModel(spec);
  if (request?.store === false) {
    model = withoutStorage(model, spec.kind === "xai" ? "xai" : "openai");
  }
  if (spec.kind === "anthropic") model = withCacheBreakpoints(model);
  if (spec.thinking === "off" && request?.thinkingOff && canSkipThinking(spec, request)) {
    model = withProviderOptions(model, request.thinkingOff.options);
  }
  const summary = spec.thinking === "summary" ? reasoningSummaryOptions(spec) : null;
  if (summary) model = withProviderOptions(model, summary);
  return model;
};

/** The ChatGPT plan is OpenAI's Responses API: its reasoning is summarized as OpenAI's is. */
const summaryQuirk = (kind: ChatProviderKind) =>
  catalogProviderOfKind(kind === "chatgpt" ? "openai" : kind)?.request?.reasoningSummary;

/**
 * Whether a provider's reasoning comes back as its own summary (Anthropic,
 * OpenAI's Responses API), which an Answer may show; any other's is raw chain
 * of thought, which is never kept or shown (#391).
 */
export const summarizesReasoning = (kind: ChatProviderKind): boolean =>
  summaryQuirk(kind) !== undefined;

/**
 * The provider options that ask a model for a summary of its reasoning: none
 * for a provider that gives none, a model that doesn't think by default (it
 * isn't turned on for one), or one the catalog knows doesn't reason.
 */
export function reasoningSummaryOptions(
  spec: Pick<ChatModelSpec, "kind" | "modelId" | "endpoint">,
): Record<string, Record<string, unknown>> | null {
  const summary = summaryQuirk(spec.kind);
  if (!summary) return null;
  if (summary.only && !summary.only.some((prefix) => spec.modelId.startsWith(prefix))) return null;
  const set = modelsProviderOfKind(spec.kind, spec.endpoint);
  return catalogFacts(set, spec.modelId)?.reasoning === false ? null : summary.options;
}

/** Whether thinking can be turned off on this model: not on one the provider says always thinks, nor one the catalog knows doesn't think. */
function canSkipThinking(
  spec: ChatModelSpec,
  request: NonNullable<ReturnType<typeof catalogProviderOfKind>>["request"],
): boolean {
  const { unless = [] } = request?.thinkingOff ?? {};
  if (unless.some((prefix) => spec.modelId.startsWith(prefix))) return false;
  const set = modelsProviderOfKind(spec.kind, spec.endpoint);
  return catalogFacts(set, spec.modelId)?.reasoning !== false;
}

function providerModel(spec: ChatModelSpec): ChatLanguageModel {
  switch (spec.kind) {
    case "openai":
      return createOpenAI({ apiKey: requireKey(spec), baseURL: hostedUrl(spec) }).responses(
        spec.modelId,
      );
    case "anthropic":
      return createAnthropic({ apiKey: requireKey(spec), baseURL: hostedUrl(spec) })(spec.modelId);
    case "google":
      return createGoogle({ apiKey: requireKey(spec), baseURL: hostedUrl(spec) })(spec.modelId);
    case "deepseek":
      return createDeepSeek({ apiKey: requireKey(spec), baseURL: hostedUrl(spec) })(spec.modelId);
    case "qwen":
      return createAlibaba({ apiKey: requireKey(spec), baseURL: hostedUrl(spec) })(spec.modelId);
    case "kimi":
      return createMoonshotAI({ apiKey: requireKey(spec), baseURL: hostedUrl(spec) })(spec.modelId);
    case "glm":
      return createZai({ apiKey: requireKey(spec), baseURL: hostedUrl(spec) })(spec.modelId);
    case "mistral":
      return createMistral({ apiKey: requireKey(spec), baseURL: hostedUrl(spec) })(spec.modelId);
    case "xai":
      // The Responses API, which is the AI SDK's default for xAI (made through `withoutStorage`).
      return createXai({ apiKey: requireKey(spec), baseURL: hostedUrl(spec) })(spec.modelId);
    case "openrouter":
      // "strict" asks for streamed token usage; the body carries the provider's routing
      // preferences (no providers that keep or train on prompts).
      return createOpenRouter({
        apiKey: requireKey(spec),
        baseURL: hostedUrl(spec),
        compatibility: "strict",
        extraBody: catalogProviderOfKind("openrouter")?.request?.body ?? {},
      })(spec.modelId);
    case "siliconflow": {
      const quirks = catalogProviderOfKind("siliconflow")?.request?.openAiCompatible;
      return createOpenAICompatible({
        name: "siliconflow",
        baseURL: hostedUrl(spec),
        apiKey: requireKey(spec),
        includeUsage: quirks?.includeUsage ?? false,
        supportsStructuredOutputs: quirks?.supportsStructuredOutputs ?? false,
      })(spec.modelId);
    }
    case "openai-compatible":
      return createOpenAICompatible({
        name: "openai-compatible",
        baseURL: requireBaseUrl(spec),
        apiKey: spec.apiKey ?? undefined,
      })(spec.modelId);
    case "ollama":
      // Ollama's own API, which sets the context window and refuses rather than cuts a
      // longer request (see ./ollamaChat). It holds a reply to a JSON schema too, which
      // Answers from models without Tools use for their Citations (#30).
      return createOllamaChatModel({
        baseUrl: requireBaseUrl(spec),
        modelId: spec.modelId,
        settings: spec.ollama ?? DEFAULT_OLLAMA_SETTINGS,
      });
    case "chatgpt":
      if (!spec.credentials) throw new Error("A ChatGPT sign-in is required.");
      return createPlanChatModel({
        baseUrl: requireBaseUrl(spec),
        modelId: spec.modelId,
        credentials: spec.credentials,
      });
  }
}
