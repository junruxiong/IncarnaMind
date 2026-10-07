/**
 * A scripted chat model for the smoke tests: every provider answers with the
 * same short Markdown Answer, streamed a few words at a time, with no network
 * and no keys, and tags a Document with every Tag whose name is a word in it.
 * Only test builds (`electron-vite build --mode test`) contain this file, and
 * only `INCARNAMIND_FAKE_CHAT=1` turns it on.
 */
import { MockLanguageModelV4 } from "ai/test";
import type { ChatModelFactory } from "../core";
import { EXCERPT_END, EXCERPT_START } from "../core/tags/classify";

type StreamResult = Awaited<ReturnType<MockLanguageModelV4["doStream"]>>;
type StreamPart = StreamResult["stream"] extends ReadableStream<infer Part> ? Part : never;
type Prompt = Parameters<MockLanguageModelV4["doStream"]>[0]["prompt"];
type GenerateOptions = Parameters<MockLanguageModelV4["doGenerate"]>[0];

/** Time between two streamed words: slow enough for a test to watch the Answer grow. */
const WORD_DELAY_MS = 60;

/** The scripted Answer to a Question. */
export function fakeAnswer(question: string): string {
  return [
    `This is a **scripted** Answer to: ${question}`,
    "",
    "- It streams in a few words at a time",
    "- Math renders too: $E = mc^2$",
    "",
    "```js",
    'console.log("IncarnaMind");',
    "```",
    "",
    "That is all.",
  ].join("\n");
}

/** The text of the last user message: the Question, at the end of its Question context. */
function lastQuestion(prompt: Prompt): string {
  const last = prompt.filter((message) => message.role === "user").at(-1);
  if (!last || typeof last.content === "string") return "";
  const text = last.content.map((part) => (part.type === "text" ? part.text : "")).join("");
  return text.split("\n\n").at(-1) ?? "";
}

const USAGE = {
  inputTokens: { total: 0, noCache: 0, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 0, text: 0, reasoning: undefined },
};

const escapeRegExp = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * The scripted tagger: of the Tag names offered, those that appear as a word
 * (ignoring case) in the Document's excerpt, i.e. its name or text.
 */
export function fakeTags(prompt: Prompt, offered: readonly string[]): string[] {
  const text = prompt
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => (part.type === "text" ? part.text : "")).join(""),
    )
    .join("\n");
  // The instructions name the markers too: the excerpt is between the last ones.
  const excerpt = text.slice(text.lastIndexOf(EXCERPT_START), text.lastIndexOf(EXCERPT_END));
  return offered.filter((name) =>
    new RegExp(`(?<![\\p{L}\\p{N}])${escapeRegExp(name)}(?![\\p{L}\\p{N}])`, "iu").test(excerpt),
  );
}

/** The Tag names a tagging request's structured output may choose from. */
function offeredTags(options: GenerateOptions): string[] {
  const format = options.responseFormat;
  const schema = (format?.type === "json" ? format.schema : undefined) as
    | { properties?: { tags?: { items?: { enum?: unknown[] } } } }
    | undefined;
  const names: unknown[] = schema?.properties?.tags?.items?.enum ?? [];
  return names.filter((name): name is string => typeof name === "string");
}

export const createFakeChatModel: ChatModelFactory = (spec) =>
  new MockLanguageModelV4({
    provider: "incarnamind-fake",
    modelId: spec.modelId,
    // Structured output (automatic tagging) gets its Tags as JSON; anything else (a connection test) gets "OK".
    doGenerate: async (options) => ({
      content: [
        {
          type: "text",
          text:
            options.responseFormat?.type === "json"
              ? JSON.stringify({ tags: fakeTags(options.prompt, offeredTags(options)) })
              : "OK",
        },
      ],
      finishReason: { unified: "stop", raw: undefined },
      usage: USAGE,
      warnings: [],
    }),
    doStream: async ({ prompt, abortSignal }) => {
      const words = fakeAnswer(lastQuestion(prompt)).split(/(?<=\s)/);
      const parts: StreamPart[] = [
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "answer" },
        ...words.map((word): StreamPart => ({ type: "text-delta", id: "answer", delta: word })),
        { type: "text-end", id: "answer" },
        { type: "finish", finishReason: { unified: "stop", raw: undefined }, usage: USAGE },
      ];
      const stream = new ReadableStream<StreamPart>({
        async pull(controller) {
          const part = parts.shift();
          if (!part) {
            controller.close();
            return;
          }
          if (part.type === "text-delta") {
            await new Promise((resolve) => setTimeout(resolve, WORD_DELAY_MS));
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
