import { APICallError } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import type { ChatModelFactory, ChatModelSpec } from "../../src/core";

/** A chat model that answers "OK" to every request. */
export function replyingModel(text = "OK"): MockLanguageModelV4 {
  return new MockLanguageModelV4({
    doGenerate: async () => ({
      content: [{ type: "text", text }],
      finishReason: { unified: "stop", raw: undefined },
      usage: {
        inputTokens: { total: 5, noCache: 5, cacheRead: undefined, cacheWrite: undefined },
        outputTokens: { total: 1, text: 1, reasoning: undefined },
      },
      warnings: [],
    }),
  });
}

/** A chat model whose provider answers every request with an HTTP error. */
export function failingModel(statusCode: number, message: string): MockLanguageModelV4 {
  return new MockLanguageModelV4({
    doGenerate: async () => {
      throw new APICallError({
        message,
        url: "https://api.example.com/v1/chat/completions",
        requestBodyValues: {},
        statusCode,
        isRetryable: false,
      });
    },
  });
}

/** A chat model that can't reach its server. */
export function unreachableModel(): MockLanguageModelV4 {
  return new MockLanguageModelV4({
    doGenerate: async () => {
      throw new APICallError({
        message: "Cannot connect to API: connect ECONNREFUSED 127.0.0.1:1",
        url: "http://127.0.0.1:1/v1/chat/completions",
        requestBodyValues: {},
        cause: new TypeError("fetch failed"),
        isRetryable: true,
      });
    },
  });
}

/** A model factory that always returns `model` and records what it was asked to build. */
export function scriptedModels(model: MockLanguageModelV4) {
  const specs: ChatModelSpec[] = [];
  const createChatModel: ChatModelFactory = (spec) => {
    specs.push(spec);
    return model;
  };
  return { createChatModel, specs, model };
}
