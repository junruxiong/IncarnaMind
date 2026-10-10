/**
 * Hand-written corrections to the generated model facts (./models.json), for
 * what models.dev doesn't say or says too coarsely. A model's own entry wins
 * over its provider's. Each says why; tests check that every model named here
 * is in the generated facts.
 */
import type { CatalogOverrides } from "./types";

export const OVERRIDES: CatalogOverrides = {
  providers: {
    // Google asks for its Gemini 3 models to run at their default temperature: lower makes
    // them loop or reason worse. The 2.5 models retire on 2026-10-20.
    google: { temperature: false },
  },
  models: {
    anthropic: {
      // These return 400 for a forced Tool choice ("any" or a named Tool).
      "claude-fable-5-1": { forcedToolChoice: false },
      "claude-opus-5-5": { forcedToolChoice: false },
      "claude-sonnet-5-5": { forcedToolChoice: false },
      "claude-haiku-5-5": {
        price: {
          currency: "USD",
          input: 0.1,
          output: 0.5,
          cacheRead: 0.01,
          note: {
            en: "Prompts over 100K tokens: $0.50 / $2.50",
            zh: "提示超过 10 万 token：$0.50 / $2.50",
          },
        },
      },
    },
    openai: {
      // It calls functions (OpenAI's docs, and LiteLLM); models.dev says it doesn't.
      "gpt-3.5-turbo": { tools: true },
    },
    google: {
      "gemini-3.8-flash": {
        price: {
          currency: "USD",
          input: 0.75,
          output: 3.75,
          cacheRead: 0.075,
          note: { en: "From 2027-01-01: $1.50 / $7.50", zh: "2027-01-01 起：$1.50 / $7.50" },
        },
      },
    },
  },
};
