/**
 * A real OpenAI model through the app's provider, with `store: false` (#159).
 * Skipped unless INCARNAMIND_REAL_OPENAI=1. The key is read from the file named
 * by INCARNAMIND_ENV_FILE (a `.env.local`), else from OPENAI_API_KEY, inside this
 * process only; it is never printed. A key OpenAI rejects skips the test.
 */
import { readFileSync } from "node:fs";
import { describe, expect, test } from "vitest";
import {
  type AnswerEngineEvent,
  type AnswerTools,
  createAiSdkAnswerEngine,
} from "../../src/core/answers/engine";
import { createAiSdkChatModel } from "../../src/core/providers/models";

function apiKey(): string | null {
  const file = process.env.INCARNAMIND_ENV_FILE;
  if (!file) return process.env.OPENAI_API_KEY ?? null;
  const line = readFileSync(file, "utf8")
    .split("\n")
    .find((each) => each.startsWith("OPENAI_API_KEY="));
  return (
    line
      ?.slice("OPENAI_API_KEY=".length)
      .trim()
      .replace(/^["']|["']$/g, "") || null
  );
}

const passage = `<passage id="P1" document="Tides" pages="1">\nSpring tides happen when the Sun, Moon and Earth line up.\n</passage>`;

describe.runIf(process.env.INCARNAMIND_REAL_OPENAI === "1")(
  "A real OpenAI model with store: false",
  { timeout: 120_000 },
  () => {
    test.each(["gpt-4.1-mini", "gpt-5-mini"])(
      "%s streams an Answer through the Tool loop and cites",
      async (modelId) => {
        const key = apiKey();
        if (!key) return;
        const cited: unknown[] = [];
        const documents: AnswerTools = {
          documentCount: 1,
          async searchDocuments() {
            return { text: passage, passageCount: 1 };
          },
          cite(records) {
            cited.push(...records);
            return "Recorded.";
          },
          hasRecord: (marker) => cited.length >= marker,
        };
        const events: AnswerEngineEvent[] = [];
        for await (const event of createAiSdkAnswerEngine().generate({
          instructions: () =>
            "Search the Documents with search_documents, record one Citation for the Passage you use with cite (marker 1, quote copied word for word), then answer in one sentence with the marker [^1].",
          messages: [{ role: "user", content: "When do spring tides happen?" }],
          question: "When do spring tides happen?",
          model: createAiSdkChatModel({ kind: "openai", baseUrl: null, apiKey: key, modelId }),
          documents,
          tools: [],
          signal: new AbortController().signal,
        })) {
          events.push(event);
        }

        const failed = events.find((event) => event.type === "failed");
        if (failed?.type === "failed" && failed.error.kind === "auth") {
          const why = failed.error.message.replace(/sk-[\w*-]+/g, "sk-…");
          console.warn(`OpenAI rejected the key: the live check is skipped. ${why}`);
          return;
        }
        expect(failed).toBeUndefined();
        expect(events.at(-1)).toEqual({ type: "finished" });
        const text = events.flatMap((event) => (event.type === "text-delta" ? [event.text] : []));
        expect(text.join("")).toMatch(/line up|align/i);
        expect(cited).not.toHaveLength(0);
      },
    );
  },
);
