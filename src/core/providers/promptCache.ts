/**
 * Prompt caching (#343). Providers bill a repeated start of a prompt at a
 * fraction of the price, and start answering sooner, when the start is the
 * same, byte for byte, as an earlier request's. Three things follow:
 * - every prompt is ordered from the most stable to the least stable part
 *   (instructions, Tools, earlier turns, then the new Question), with no
 *   per-request value in the stable part;
 * - Anthropic caches only where a request marks it, so `withCacheBreakpoints`
 *   marks the stable parts of each request to it;
 * - OpenAI routes a request to a cache by its `prompt_cache_key`, so a
 *   conversation sends one (`cacheProviderOptions`).
 * DeepSeek and Gemini cache a repeated start on their own and need nothing.
 *
 * Anthropic (https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
 * a request holds at most 4 breakpoints; a cache covers the tools, then the
 * system prompt, then the messages, up to a breakpoint; a breakpoint finds an
 * earlier entry at most 20 positions back; a prompt shorter than the model's
 * minimum (512 tokens for Haiku 5.5 and Sonnet 5.5) is not cached, with no error.
 */
import { createHash } from "node:crypto";
import { type LanguageModelMiddleware, wrapLanguageModel } from "ai";
import type { TokenUsage } from "../runs/engine";
import type { ChatLanguageModel } from "./models";

type Params = Parameters<NonNullable<LanguageModelMiddleware["transformParams"]>>[0]["params"];
type Prompt = Params["prompt"];

const EPHEMERAL = { anthropic: { cacheControl: { type: "ephemeral" } } };

/**
 * The index of each message of `prompt` to mark as the end of a cached prefix:
 * - the last instructions message (so the Tools, which come first, and the
 *   instructions are one cached prefix);
 * - the message just before the Question: the earlier turns, which the next
 *   Question repeats;
 * - in a Tool loop, the last message: the Passages and calls so far, which the
 *   next step repeats.
 * Never the Question itself, which is new with each request. With the last
 * Tool's, that is four at most, Anthropic's limit.
 */
export function breakpointsOf(prompt: Prompt): number[] {
  const marks: number[] = [];
  let instructions = -1;
  while (prompt[instructions + 1]?.role === "system") instructions++;
  if (instructions >= 0) marks.push(instructions);
  let question = prompt.length - 1;
  while (question > instructions && prompt[question]?.role !== "user") question--;
  if (question - 1 > instructions) marks.push(question - 1);
  if (question < prompt.length - 1) marks.push(prompt.length - 1);
  return marks;
}

/**
 * A model whose requests carry cache breakpoints on their stable parts (see
 * `breakpointsOf`), and on the last Tool: the Tools alone stay cached when
 * the instructions change.
 */
export function withCacheBreakpoints(model: ChatLanguageModel): ChatLanguageModel {
  return wrapLanguageModel({
    model,
    middleware: {
      specificationVersion: "v4",
      transformParams: async ({ params }) => {
        const marks = new Set(breakpointsOf(params.prompt));
        const prompt = params.prompt.map((message, at) =>
          marks.has(at)
            ? { ...message, providerOptions: { ...message.providerOptions, ...EPHEMERAL } }
            : message,
        ) as Prompt;
        const tools = params.tools?.map((each, at, all) =>
          at === all.length - 1 && each.type === "function"
            ? { ...each, providerOptions: { ...each.providerOptions, ...EPHEMERAL } }
            : each,
        );
        return { ...params, prompt, ...(tools && { tools }) };
      },
    },
  });
}

/**
 * The provider options that name a conversation's cache: the same key for
 * every request of a conversation (a Mind or an assistant's conversation), so
 * its requests reach the same cache. A hash, so no id of ours leaves the
 * computer. Providers other than OpenAI's ignore the options.
 */
export function cacheProviderOptions(key: string | undefined) {
  if (!key) return undefined;
  const hashed = createHash("sha256").update(key).digest("hex").slice(0, 32);
  return { openai: { promptCacheKey: hashed } };
}

/** What a model call counted, as the AI SDK reports it for every provider, with the cached input tokens. */
export function tokenUsageOf(usage: {
  inputTokens: number | undefined;
  outputTokens: number | undefined;
  inputTokenDetails?: {
    cacheReadTokens: number | undefined;
    cacheWriteTokens: number | undefined;
  };
}): TokenUsage {
  return {
    inputTokens: usage.inputTokens,
    outputTokens: usage.outputTokens,
    cachedInputTokens: usage.inputTokenDetails?.cacheReadTokens,
    cacheWriteTokens: usage.inputTokenDetails?.cacheWriteTokens,
  };
}
