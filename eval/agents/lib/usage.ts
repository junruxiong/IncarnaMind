/**
 * What IncarnaMind's own model calls cost in a run: the built-in assistant's
 * steps (A0, A0-L) and the Answer behind `ask` (Q). The core builds every
 * chat model through its `createChatModel` adapter, so the harness wraps the
 * default factory and counts each call's usage there; nothing in the core
 * changes. Runs are one at a time, so the meter is reset at each run's start.
 */
import type { LanguageModelV4Usage } from "@ai-sdk/provider";
import { type LanguageModelMiddleware, wrapLanguageModel } from "ai";
import { type ChatModelFactory, createAiSdkChatModel } from "../../../src/core";

export interface ModelUsage {
  /** Model calls (requests), each one step of a Tool-calling loop or one small call. */
  calls: number;
  /** Input tokens, cached ones included. */
  inputTokens: number;
  /** Of those, read from the provider's cache. */
  cachedInputTokens: number;
  /** Of those, written to the provider's cache. */
  cacheWriteTokens: number;
  outputTokens: number;
  /** Of the output, reasoning (thinking). */
  reasoningTokens: number;
}

export const NO_USAGE: ModelUsage = {
  calls: 0,
  inputTokens: 0,
  cachedInputTokens: 0,
  cacheWriteTokens: 0,
  outputTokens: 0,
  reasoningTokens: 0,
};

export function addUsage(a: ModelUsage, b: ModelUsage): ModelUsage {
  return {
    calls: a.calls + b.calls,
    inputTokens: a.inputTokens + b.inputTokens,
    cachedInputTokens: a.cachedInputTokens + b.cachedInputTokens,
    cacheWriteTokens: a.cacheWriteTokens + b.cacheWriteTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    reasoningTokens: a.reasoningTokens + b.reasoningTokens,
  };
}

export interface UsageMeter {
  /** Builds chat models for the core, counting what each call uses. */
  factory: ChatModelFactory;
  /** What was used since the last reset, and starts counting afresh. */
  take(): ModelUsage;
}

function fromUsage(usage: LanguageModelV4Usage | undefined): ModelUsage {
  return {
    calls: 1,
    inputTokens: usage?.inputTokens.total ?? 0,
    cachedInputTokens: usage?.inputTokens.cacheRead ?? 0,
    cacheWriteTokens: usage?.inputTokens.cacheWrite ?? 0,
    outputTokens: usage?.outputTokens.total ?? 0,
    reasoningTokens: usage?.outputTokens.reasoning ?? 0,
  };
}

export function createUsageMeter(base: ChatModelFactory = createAiSdkChatModel): UsageMeter {
  let used: ModelUsage = NO_USAGE;
  const record = (usage: LanguageModelV4Usage | undefined) => {
    used = addUsage(used, fromUsage(usage));
  };
  const middleware: LanguageModelMiddleware = {
    wrapGenerate: async ({ doGenerate }) => {
      const result = await doGenerate();
      record(result.usage);
      return result;
    },
    wrapStream: async ({ doStream }) => {
      const result = await doStream();
      let finished = false;
      const stream = result.stream.pipeThrough(
        new TransformStream({
          transform(part, controller) {
            if (part.type === "finish" && !finished) {
              finished = true;
              record(part.usage);
            }
            controller.enqueue(part);
          },
          flush() {
            // A stream cut short still made a call.
            if (!finished) record(undefined);
          },
        }),
      );
      return { ...result, stream };
    },
  };
  return {
    factory: (spec) => wrapLanguageModel({ model: base(spec), middleware }),
    take() {
      const taken = used;
      used = NO_USAGE;
      return taken;
    },
  };
}
