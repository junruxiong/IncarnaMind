/**
 * A scripted chat model for the smoke tests: no network and no keys. Only
 * test builds (`electron-vite build --mode test`) contain this file, and only
 * `INCARNAMIND_FAKE_CHAT=1` turns it on.
 *
 * With no Documents (no Tools offered), every Question gets the same short
 * Markdown Answer, streamed a few words at a time. With Documents, it does
 * what a Tool-calling model does: it searches for the Question, cites the line
 * of the first Passage that shares a word with the Question (on the page that
 * line is on), then answers with a marker. A Question with "misquote" in it
 * gets a quote that isn't on the page.
 */
import { MockLanguageModelV4 } from "ai/test";
import type { ChatModelFactory } from "../core";

type StreamResult = Awaited<ReturnType<MockLanguageModelV4["doStream"]>>;
type StreamPart = StreamResult["stream"] extends ReadableStream<infer Part> ? Part : never;
type Prompt = Parameters<MockLanguageModelV4["doStream"]>[0]["prompt"];

/** Time between two streamed words: slow enough for a test to watch the Answer grow. */
const WORD_DELAY_MS = 60;

/** The scripted Answer to a Question, when there are no Documents. */
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

/** The scripted Answer when it cited something: one sentence with a marker. */
const CITED_ANSWER = "Your Documents answer this [^1].\n\nThat is all.";

/** The text of the last user message: the Question, at the end of its Question context. */
function lastQuestion(prompt: Prompt): string {
  const last = prompt.filter((message) => message.role === "user").at(-1);
  if (!last || typeof last.content === "string") return "";
  const text = last.content.map((part) => (part.type === "text" ? part.text : "")).join("");
  return text.split("\n\n").at(-1) ?? "";
}

/** The text results of the Tool calls so far, by Tool name, oldest first. */
function toolResults(prompt: Prompt, tool: string): string[] {
  const results: string[] = [];
  for (const message of prompt) {
    if (message.role !== "tool") continue;
    for (const part of message.content) {
      if (part.type === "tool-result" && part.toolName === tool && part.output.type === "text") {
        results.push(part.output.value);
      }
    }
  }
  return results;
}

/**
 * The record for the first Passage of a search result: its line that shares
 * a word with the Question, the page that line is on, and the line as the quote.
 */
function recordFor(result: string, question: string): Record<string, unknown> | null {
  const match =
    /<passage id="([^"]+)" document="[^"]*"(?: pages="(\d+)(?:-\d+)?")?>\n([\s\S]*?)\n<\/passage>/.exec(
      result,
    );
  if (!match) return null;
  const [, id, firstPage, text = ""] = match;
  const words = question
    .toLowerCase()
    .split(/\W+/)
    .filter((word) => word.length >= 4);
  let page = firstPage ? Number(firstPage) : null;
  let chosen: { line: string; page: number | null; shared: number } | null = null;
  for (const raw of text.split("\n")) {
    const mark = /^\[p\. (\d+)\] /.exec(raw);
    if (mark) page = Number(mark[1]);
    const line = raw.replace(/^\[p\. \d+\] /, "").trim();
    if (!line) continue;
    const shared = words.filter((word) => line.toLowerCase().includes(word)).length;
    if (!chosen || shared > chosen.shared) chosen = { line, page, shared };
  }
  if (!chosen) return null;
  const quote = /misquote/i.test(question)
    ? `${chosen.line.replace(/[.!?]$/, "")} and tomorrow.`
    : chosen.line;
  return {
    marker: 1,
    passage: id,
    ...(chosen.page === null ? {} : { pageFrom: chosen.page, pageTo: chosen.page }),
    quote,
  };
}

const USAGE = {
  inputTokens: { total: 0, noCache: 0, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 0, text: 0, reasoning: undefined },
};

/** What the model does next: call a Tool, or stream its Answer. */
function nextReply(
  prompt: Prompt,
  tools: readonly string[],
): { text: string } | { tool: string; input: unknown } {
  const question = lastQuestion(prompt);
  if (!tools.includes("search_documents")) return { text: fakeAnswer(question) };
  const searches = toolResults(prompt, "search_documents");
  if (searches.length === 0) return { tool: "search_documents", input: { query: question } };
  if (toolResults(prompt, "cite").length === 0) {
    const record = recordFor(searches.at(-1) ?? "", question);
    if (record) return { tool: "cite", input: { citations: [record] } };
    return { text: fakeAnswer(question) };
  }
  return { text: CITED_ANSWER };
}

export const createFakeChatModel: ChatModelFactory = (spec) =>
  new MockLanguageModelV4({
    provider: "incarnamind-fake",
    modelId: spec.modelId,
    doStream: async ({ prompt, tools, abortSignal }) => {
      const reply = nextReply(
        prompt,
        (tools ?? []).map((tool) => tool.name),
      );
      const parts: StreamPart[] = [{ type: "stream-start", warnings: [] }];
      if ("tool" in reply) {
        parts.push(
          {
            type: "tool-call",
            toolCallId: `fake-${reply.tool}`,
            toolName: reply.tool,
            input: JSON.stringify(reply.input),
          },
          { type: "finish", finishReason: { unified: "tool-calls", raw: undefined }, usage: USAGE },
        );
      } else {
        const words = reply.text.split(/(?<=\s)/);
        parts.push(
          { type: "text-start", id: "answer" },
          ...words.map((word): StreamPart => ({ type: "text-delta", id: "answer", delta: word })),
          { type: "text-end", id: "answer" },
          { type: "finish", finishReason: { unified: "stop", raw: undefined }, usage: USAGE },
        );
      }
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
