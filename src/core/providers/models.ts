/**
 * The seam between provider settings and the AI SDK: everything that builds a
 * chat model goes through a `ChatModelFactory`. The core uses the AI SDK
 * providers by default; tests inject AI SDK mock models instead (ADR-0005).
 */
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogle } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { LanguageModel } from "ai";
import type { ChatProviderKind } from "../api";

/** A model object, never a model id string (which the AI SDK would send to its gateway). */
export type ChatLanguageModel = Exclude<LanguageModel, string>;

/** Everything needed to build a chat model. */
export interface ChatModelSpec {
  kind: ChatProviderKind;
  /** Set for "openai-compatible" and "ollama". */
  baseUrl: string | null;
  apiKey: string | null;
  modelId: string;
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

/** The AI SDK provider for each kind. */
export const createAiSdkChatModel: ChatModelFactory = (spec) => {
  switch (spec.kind) {
    case "openai":
      return createOpenAI({ apiKey: requireKey(spec) })(spec.modelId);
    case "anthropic":
      return createAnthropic({ apiKey: requireKey(spec) })(spec.modelId);
    case "google":
      return createGoogle({ apiKey: requireKey(spec) })(spec.modelId);
    case "openai-compatible":
      return createOpenAICompatible({
        name: "openai-compatible",
        baseURL: requireBaseUrl(spec),
        apiKey: spec.apiKey ?? undefined,
      })(spec.modelId);
    case "ollama":
      // Ollama serves the OpenAI chat API under /v1.
      return createOpenAICompatible({ name: "ollama", baseURL: `${requireBaseUrl(spec)}/v1` })(
        spec.modelId,
      );
  }
};
