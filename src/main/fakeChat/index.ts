/**
 * A scripted chat model for the smoke tests: no network and no keys. Only
 * test builds (`electron-vite build --mode test`) contain this file, and only
 * `INCARNAMIND_FAKE_CHAT=1` turns it on.
 *
 * A Question with "blank reply" in it gets a reply with no text at all. A model whose id ends
 * "-not-found" fails as one its provider doesn't know when it answers; one ending "-claims-not-found"
 * does when it finds or checks claims. A key with "refused" in it is refused,
 * as a provider refuses a wrong key (HTTP 401). A model whose id ends "-no-tools" refuses
 * Tools and structured output, as a small server model does, so its Answers have no Citations.
 * A Document whose text says "unreachable" can't reach the model the first time it is tagged.
 */

import { APICallError } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import type { ChatModelFactory } from "../../core";
import { searchParts } from "./answers";
import { isCell } from "./cells";
import { fakeClaims } from "./claims";
import { fakeConfirm } from "./confirm";
import { editParts, fakeEdit } from "./edit";
import { fakePlan } from "./plan";
import {
  type GenerateOptions,
  lastQuestion,
  lastUserText,
  SLOWLY,
  type StreamPart,
  USAGE,
} from "./prompt";
import { fakeResearchTags } from "./researchTags";
import { fakeSkillDraft } from "./skillDraft";
import { refusal, replyParts } from "./stream";
import { fakeStructuredOutput, tagsUnreachable } from "./structured";

/** Time between two streamed words: slow enough for a test to watch the Answer grow. */
const WORD_DELAY_MS = 60;
/** For a Question asked "slowly": the wait before each step, and between two words of the Answer. */
const SLOW_STEP_MS = 1200;
const SLOW_WORD_MS = 250;
/** Between two words of Edit with AI's new text: a passage takes a moment, not seconds. */
const EDIT_WORD_MS = 10;

/** The cells whose provider has refused them once already ("busy" in the Question). */
const refusedOnce = new Set<string>();

/**
 * The scripted replies to a request that doesn't ask for JSON, in order: the first that
 * doesn't return null answers it.
 */
const GENERATE_HANDLERS: readonly ((options: GenerateOptions) => string | null)[] = [
  fakePlan,
  fakeResearchTags,
  fakeClaims,
  fakeConfirm,
  fakeSkillDraft,
  (options) => fakeEdit(options.prompt)?.text ?? null,
];

function generatedText(options: GenerateOptions): string {
  for (const handler of GENERATE_HANDLERS) {
    const text = handler(options);
    if (text !== null) return text;
  }
  return "OK";
}

export const createFakeChatModel: ChatModelFactory = (spec) =>
  new MockLanguageModelV4({
    provider: "incarnamind-fake",
    modelId: spec.modelId,
    // Structured output (automatic tagging) gets its Tags as JSON; Edit with AI its new text;
    // anything else (a connection test) gets "OK".
    doGenerate: async (options) => {
      const refused = refusal(spec, options);
      if (refused) throw refused;
      // A model id ending "-claims-not-found" is one its provider doesn't know when it checks or finds claims (#764).
      if (spec.modelId.endsWith("-claims-not-found")) {
        throw new APICallError({
          message: `The model ${spec.modelId} does not exist`,
          url: "https://fake.invalid/chat",
          requestBodyValues: {},
          statusCode: 404,
        });
      }
      if (tagsUnreachable(options)) {
        throw new APICallError({
          message: "Cannot connect to API: fetch failed",
          url: "https://fake.invalid/chat",
          requestBodyValues: {},
          isRetryable: false,
        });
      }
      return {
        content: [
          {
            type: "text",
            text:
              options.responseFormat?.type === "json"
                ? fakeStructuredOutput(options)
                : generatedText(options),
          },
        ],
        finishReason: { unified: "stop", raw: undefined },
        usage: USAGE,
        warnings: [],
      };
    },
    doStream: async ({ prompt, tools, abortSignal, responseFormat }) => {
      const refused = refusal(spec, {
        ...(tools && { tools }),
        ...(responseFormat && { responseFormat }),
      });
      if (refused) throw refused;
      // A model id ending "-not-found" is one its provider doesn't know, for a test of that Problem.
      if (spec.modelId.endsWith("-not-found")) {
        throw new APICallError({
          message: `The model ${spec.modelId} does not exist`,
          url: "https://fake.invalid/chat",
          requestBodyValues: {},
          statusCode: 404,
        });
      }
      // A busy provider refuses each "busy" cell's first request once, as an HTTP 429.
      const cell =
        isCell(prompt) && /\bbusy\b/i.test(lastQuestion(prompt)) ? lastUserText(prompt) : null;
      if (cell && !refusedOnce.has(cell)) {
        refusedOnce.add(cell);
        throw new APICallError({
          message: "Too Many Requests",
          url: "http://127.0.0.1/v1/chat/completions",
          requestBodyValues: {},
          statusCode: 429,
          isRetryable: false,
        });
      }
      const search = tools?.find((each) => each.type === "provider");
      const edit = search ? null : fakeEdit(prompt);
      const parts = search
        ? searchParts(search.name, lastQuestion(prompt))
        : edit
          ? editParts(edit.text)
          : replyParts(prompt, tools ?? []);
      const slow = edit ? edit.slow : !search && SLOWLY.test(lastQuestion(prompt));
      const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
      const stream = new ReadableStream<StreamPart>({
        async pull(controller) {
          const part = parts.shift();
          if (!part) {
            controller.close();
            return;
          }
          if (slow && part.type === "stream-start") await wait(SLOW_STEP_MS);
          if (part.type === "text-delta" || part.type === "tool-input-delta") {
            await wait(slow ? SLOW_WORD_MS : edit ? EDIT_WORD_MS : WORD_DELAY_MS);
          }
          if (slow && part.type === "reasoning-delta") await wait(WORD_DELAY_MS);
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
