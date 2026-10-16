import type { MockLanguageModelV4 } from "ai/test";

type StreamResult = Awaited<ReturnType<MockLanguageModelV4["doStream"]>>;
export type StreamPart = StreamResult["stream"] extends ReadableStream<infer Part> ? Part : never;
export type Prompt = Parameters<MockLanguageModelV4["doStream"]>[0]["prompt"];
export type GenerateOptions = Parameters<MockLanguageModelV4["doGenerate"]>[0];

export type Reply = ({ text: string } | { tool: string; input: unknown }) & { reasoning?: string };

export const SLOWLY = /\bslowly\b/i;

/** A request to shorten a Mind's text. */
export const SHORTEN = /\bshort(?:en|er)\b|缩短|改短/i;

/** The text of a user message. */
export const userText = (message: Prompt[number] | undefined): string =>
  !message || typeof message.content === "string"
    ? ""
    : message.content.map((part) => (part.type === "text" ? part.text : "")).join("");

/** The text of the last user message: the Question, after what it is asked beside. */
export const lastUserText = (prompt: Prompt): string =>
  userText(prompt.filter((message) => message.role === "user").at(-1));

/** The Question, at the end of its Question context. */
export const lastQuestion = (prompt: Prompt): string =>
  lastUserText(prompt).split("\n\n").at(-1) ?? "";

/** The text results of the Tool calls so far, by Tool name, oldest first. */
export function toolResults(prompt: Prompt, tool: string): string[] {
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

/** The system prompt's text. */
export function systemOf(prompt: Prompt): string {
  return prompt
    .filter((message) => message.role === "system")
    .map((message) => (typeof message.content === "string" ? message.content : ""))
    .join("\n");
}

/** Every result of a Tool's calls so far, as text, whether they worked or not, oldest first. */
export function allResults(prompt: Prompt, tool: string): string[] {
  const found: string[] = [];
  for (const message of prompt) {
    if (message.role !== "tool") continue;
    for (const part of message.content) {
      if (part.type !== "tool-result" || part.toolName !== tool) continue;
      const output = part.output;
      found.push(
        output.type === "text" || output.type === "error-text"
          ? output.value
          : JSON.stringify(output),
      );
    }
  }
  return found;
}

/** What a Tool call gave back, as text, whether it worked or not; undefined if it hasn't been called. */
export function lastResult(prompt: Prompt, tool: string): string | undefined {
  let found: string | undefined;
  for (const message of prompt) {
    if (message.role !== "tool") continue;
    for (const part of message.content) {
      if (part.type !== "tool-result" || part.toolName !== tool) continue;
      const output = part.output;
      found =
        output.type === "text" || output.type === "error-text"
          ? output.value
          : JSON.stringify(output);
    }
  }
  return found;
}

export const USAGE = {
  inputTokens: { total: 0, noCache: 0, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 0, text: 0, reasoning: undefined },
};
