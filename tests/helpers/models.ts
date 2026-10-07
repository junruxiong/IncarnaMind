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

type StreamResult = Awaited<ReturnType<MockLanguageModelV4["doStream"]>>;
type StreamPart = StreamResult["stream"] extends ReadableStream<infer Part> ? Part : never;

const STREAM_USAGE = {
  inputTokens: { total: 5, noCache: 5, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 1, text: 1, reasoning: undefined },
};

const finishParts = (): StreamPart[] => [
  { type: "text-end", id: "answer" },
  { type: "finish", finishReason: { unified: "stop", raw: undefined }, usage: STREAM_USAGE },
];

/**
 * A chat model that streams each reply in turn, one per request (the last one
 * again after that), split into chunks of `chunkSize` characters sent
 * `delayMs` apart.
 */
export function streamingModel(
  replies: string | readonly string[],
  { chunkSize = 4, delayMs = 0 }: { chunkSize?: number; delayMs?: number } = {},
): MockLanguageModelV4 {
  const all = typeof replies === "string" ? [replies] : replies;
  let calls = 0;
  return new MockLanguageModelV4({
    doStream: async ({ abortSignal }) => {
      const reply = all[Math.min(calls++, all.length - 1)] ?? "";
      const parts: StreamPart[] = [
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "answer" },
      ];
      for (let at = 0; at < reply.length; at += chunkSize) {
        parts.push({ type: "text-delta", id: "answer", delta: reply.slice(at, at + chunkSize) });
      }
      parts.push(...finishParts());
      const stream = new ReadableStream<StreamPart>({
        async pull(controller) {
          const part = parts.shift();
          if (!part) {
            controller.close();
            return;
          }
          if (delayMs > 0 && part.type === "text-delta") {
            await new Promise((resolve) => setTimeout(resolve, delayMs));
          }
          if (abortSignal?.aborted) {
            controller.error(new DOMException("The request was aborted.", "AbortError"));
            return;
          }
          controller.enqueue(part);
        },
      });
      return { stream };
    },
  });
}

export interface ControlledModel {
  model: MockLanguageModelV4;
  /** Resolves once the model has been asked to answer (the nth time, from 1). */
  requested(count?: number): Promise<void>;
  /** Streams more text of the current Answer. */
  push(text: string): void;
  finish(): void;
  /** Whether the current request was aborted. */
  readonly aborted: boolean;
}

/** A chat model whose Answer the test streams by hand: `push` text, then `finish`. */
export function controlledModel(): ControlledModel {
  let controller: ReadableStreamDefaultController<StreamPart> | null = null;
  let aborted = false;
  let requests = 0;
  const waiting: { count: number; resolve: () => void }[] = [];
  const model = new MockLanguageModelV4({
    doStream: async ({ abortSignal }) => {
      aborted = false;
      const stream = new ReadableStream<StreamPart>({
        start(started) {
          controller = started;
          started.enqueue({ type: "stream-start", warnings: [] });
          started.enqueue({ type: "text-start", id: "answer" });
        },
      });
      abortSignal?.addEventListener("abort", () => {
        aborted = true;
        try {
          controller?.error(new DOMException("The request was aborted.", "AbortError"));
        } catch {
          // Already closed.
        }
      });
      requests++;
      for (const wait of waiting.filter((each) => each.count <= requests)) wait.resolve();
      return { stream };
    },
  });
  const current = () => {
    if (!controller) throw new Error("The model hasn't been asked yet.");
    return controller;
  };
  return {
    model,
    requested: (count = 1) =>
      requests >= count
        ? Promise.resolve()
        : new Promise((resolve) => waiting.push({ count, resolve })),
    push(text) {
      current().enqueue({ type: "text-delta", id: "answer", delta: text });
    },
    finish() {
      const stream = current();
      for (const part of finishParts()) stream.enqueue(part);
      stream.close();
    },
    get aborted() {
      return aborted;
    },
  };
}

/** A chat model whose provider refuses to stream, with an HTTP status, or with no connection (no status). */
export function failingStreamModel(statusCode: number | undefined, message: string) {
  return new MockLanguageModelV4({
    doStream: async () => {
      throw new APICallError({
        message,
        url: "https://api.example.com/v1/chat/completions",
        requestBodyValues: {},
        statusCode,
        cause: statusCode === undefined ? new TypeError("fetch failed") : undefined,
        isRetryable: false,
      });
    },
  });
}

/** What a model was asked, simplified: the instructions, then each message's role and text. */
export function promptOf(model: MockLanguageModelV4, call = 0): { role: string; text: string }[] {
  const options = model.doStreamCalls[call];
  if (!options) throw new Error(`The model wasn't asked ${call + 1} time(s).`);
  return options.prompt.map((message) => ({
    role: message.role,
    text:
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => ("text" in part ? part.text : "")).join(""),
  }));
}

type CallOptions = Parameters<MockLanguageModelV4["doStream"]>[0];

/** One request to a scripted model, as its script sees it. */
export interface ModelCall {
  /** Which request this is, from 0. */
  index: number;
  /** The names of the Tools offered. */
  tools: string[];
  /** Whether the request asks for a JSON object (structured output). */
  json: boolean;
  /** The system prompt. */
  system: string;
  /** The results of the Tool calls so far, oldest first: their Tool's name and text. */
  results: { tool: string; text: string }[];
  options: CallOptions;
}

/** What a scripted model replies to one request. */
export type ScriptedReply =
  | {
      text?: string;
      calls?: { tool: string; input: unknown }[];
      /** Stops streaming once `after` characters of the text are out, until `until` resolves. */
      pause?: { after: number; until: Promise<void> };
    }
  /** The provider refuses the request with an HTTP error. */
  | { error: { status: number; message: string } };

function resultsOf(options: CallOptions): ModelCall["results"] {
  const results: ModelCall["results"] = [];
  for (const message of options.prompt) {
    if (message.role !== "tool") continue;
    for (const part of message.content) {
      if (part.type !== "tool-result") continue;
      const output = part.output;
      const text =
        output.type === "text" || output.type === "error-text"
          ? output.value
          : output.type === "json" || output.type === "error-json"
            ? JSON.stringify(output.value)
            : "";
      results.push({ tool: part.toolName, text });
    }
  }
  return results;
}

/**
 * A chat model whose every reply comes from `script`, given the request: text
 * (streamed in chunks of `chunkSize`), Tool calls, or a provider error. For
 * scripting the Tool-calling loop: search, cite, then answer.
 */
export function scriptedModel(
  script: (call: ModelCall) => ScriptedReply,
  { chunkSize = 8 }: { chunkSize?: number } = {},
): MockLanguageModelV4 {
  let index = 0;
  return new MockLanguageModelV4({
    doStream: async (options) => {
      const system = options.prompt
        .filter((message) => message.role === "system")
        .map((message) => (typeof message.content === "string" ? message.content : ""))
        .join("\n");
      const call: ModelCall = {
        index: index++,
        tools: (options.tools ?? []).map((each) => each.name),
        json: options.responseFormat?.type === "json",
        system,
        results: resultsOf(options),
        options,
      };
      const reply = script(call);
      if ("error" in reply) {
        throw new APICallError({
          message: reply.error.message,
          url: "http://127.0.0.1:11434/v1/chat/completions",
          requestBodyValues: {},
          statusCode: reply.error.status,
          responseBody: JSON.stringify({ error: { message: reply.error.message } }),
          isRetryable: false,
        });
      }
      const parts: StreamPart[] = [{ type: "stream-start", warnings: [] }];
      const text = reply.text ?? "";
      if (text) {
        parts.push({ type: "text-start", id: "answer" });
        for (let at = 0; at < text.length; at += chunkSize) {
          parts.push({ type: "text-delta", id: "answer", delta: text.slice(at, at + chunkSize) });
        }
        parts.push({ type: "text-end", id: "answer" });
      }
      (reply.calls ?? []).forEach((each, number) => {
        parts.push({
          type: "tool-call",
          toolCallId: `call-${call.index}-${number}`,
          toolName: each.tool,
          input: JSON.stringify(each.input),
        });
      });
      parts.push({
        type: "finish",
        finishReason: {
          unified: reply.calls?.length ? "tool-calls" : "stop",
          raw: undefined,
        },
        usage: STREAM_USAGE,
      });
      const { pause } = reply;
      let streamed = 0;
      const stream = new ReadableStream<StreamPart>({
        async pull(controller) {
          const part = parts.shift();
          if (!part) {
            controller.close();
            return;
          }
          if (pause && streamed >= pause.after && part.type !== "stream-start") {
            await pause.until;
          }
          if (part.type === "text-delta") streamed += part.delta.length;
          if (options.abortSignal?.aborted) {
            controller.error(new DOMException("The request was aborted.", "AbortError"));
            return;
          }
          controller.enqueue(part);
        },
      });
      return { stream };
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
