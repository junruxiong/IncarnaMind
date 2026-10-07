/**
 * The Answer engine: the one place that turns Question context into a
 * streamed Answer by calling a chat model. Everything else about Answers
 * (building the context, writing the Answer into the Mind, stop and
 * regenerate, the event stream) depends only on this port, so the agent layer
 * behind it can change (the AI SDK today) without touching the rest.
 */
import { streamText } from "ai";
import type { ProviderError } from "../api";
import type { ChatLanguageModel } from "../providers/models";
import { classifyProviderError } from "../providers/providerErrors";

/** One message of Question context. */
export interface AnswerMessage {
  role: "user" | "assistant";
  /** Markdown. */
  content: string;
}

export interface AnswerRequest {
  /** The instructions: how to answer. */
  system: string;
  /** The Question context, oldest first; the last message is the User's and ends with the Question. */
  messages: AnswerMessage[];
  /** The model to answer with, from `Core.prepareChatModel`. */
  model: ChatLanguageModel;
  /** Stops generating. The stream then ends, with neither "finished" nor "failed". */
  signal: AbortSignal;
}

export type AnswerEngineEvent =
  /** More of the Answer's text (Markdown), in order. */
  | { type: "text-delta"; text: string }
  /** The Answer is complete. Nothing follows. */
  | { type: "finished" }
  /** The model or its provider failed. Nothing follows. */
  | { type: "failed"; error: ProviderError };

export interface AnswerEngine {
  /** Streams the Answer to the Question context as events. It never throws: failures are events. */
  generate(request: AnswerRequest): AsyncIterable<AnswerEngineEvent>;
}

/** The engine on the Vercel AI SDK: one `streamText` call, no Tools yet (#30 adds them). */
export function createAiSdkAnswerEngine(): AnswerEngine {
  return {
    async *generate({ system, messages, model, signal }) {
      const failed = (error: unknown): AnswerEngineEvent => ({
        type: "failed",
        error: classifyProviderError(error),
      });
      try {
        const result = streamText({
          model,
          instructions: system,
          messages,
          abortSignal: signal,
          // Errors arrive as stream parts; don't also log them.
          onError: () => undefined,
        });
        for await (const part of result.fullStream) {
          if (signal.aborted || part.type === "abort") return;
          if (part.type === "text-delta") {
            if (part.text) yield { type: "text-delta", text: part.text };
          } else if (part.type === "error") {
            yield failed(part.error);
            return;
          }
        }
      } catch (error) {
        if (signal.aborted) return;
        yield failed(error);
        return;
      }
      if (!signal.aborted) yield { type: "finished" };
    },
  };
}
