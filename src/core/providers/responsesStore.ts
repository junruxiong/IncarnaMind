import { wrapLanguageModel } from "ai";
import type { ChatLanguageModel } from "./models";

/**
 * A model whose every request carries these provider options (by the AI SDK
 * provider's key, e.g. `{ openai: { store: false } }`), merged under what the
 * call itself sets for the same key.
 */
export function withProviderOptions(
  model: ChatLanguageModel,
  options: Record<string, Record<string, unknown>>,
): ChatLanguageModel {
  return wrapLanguageModel({
    model,
    middleware: {
      specificationVersion: "v4",
      transformParams: async ({ params }) => {
        const providerOptions = { ...params.providerOptions } as Record<
          string,
          Record<string, unknown>
        >;
        for (const [key, value] of Object.entries(options)) {
          providerOptions[key] = { ...providerOptions[key], ...value };
        }
        return { ...params, providerOptions } as typeof params;
      },
    },
  });
}

/**
 * A Responses API model that asks its provider not to store requests: every
 * request carries `store: false`. The AI SDK's OpenAI provider uses the
 * Responses API, which keeps each request and its response on the provider's
 * servers unless told not to; so does xAI's (`providerKey` "xai"). Reasoning
 * summaries are not asked for (the AI SDK leaves them off unless
 * `reasoningSummary` is set), and with `store: false` the SDK itself asks for
 * encrypted reasoning, so earlier steps of a Tool loop are sent in full rather
 * than by reference.
 *
 * Every Responses API model is made through this, so each call site of the
 * chat model (Answers, tagging, Organize, the connection test) gets it with
 * the model.
 */
export function withoutStorage(
  model: ChatLanguageModel,
  providerKey = "openai",
): ChatLanguageModel {
  return withProviderOptions(model, { [providerKey]: { store: false } });
}
