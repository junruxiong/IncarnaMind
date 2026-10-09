/**
 * How a local model cites, chosen from its capabilities and size before its
 * first request (see `citingMode` in src/core/providers/ollamaModels.ts):
 * against a stub of Ollama's API.
 */
import { describe, expect, test } from "vitest";
import {
  answerChats,
  answering,
  askAndEnd,
  MISTRAL,
  QWEN35,
  QWEN35_9B,
  setUpLocalModel,
  TIDES,
} from "../helpers/localModels";
import { question, writeMind } from "../helpers/minds";
import { startOllamaServer } from "../helpers/ollama";

describe("Choosing how a local model cites, before the first request", () => {
  test("a model without Tools gets structured output from its first request, and is never offered Tools", async () => {
    const ollama = await startOllamaServer({
      models: [MISTRAL],
      reply: answering(() => ({
        content: JSON.stringify({
          answer: "Spring tides happen at full moon [^1].",
          citations: [
            { marker: 1, passage: "P1", quote: "Spring tides happen at new moon and at full moon" },
          ],
        }),
      })),
    });
    const { core, mind, client } = await setUpLocalModel(ollama, "mistral", {
      documents: [{ name: "Tides.md", contents: TIDES }],
    });
    const asked = question("When are spring tides?");
    writeMind(client, [asked]);

    const { ended } = await askAndEnd(core, client, mind.id, asked.attrs.id);

    expect(ended.event).toBe("finished");
    expect(answerChats(ollama)).toHaveLength(1);
    const [first] = answerChats(ollama);
    expect(first?.body.tools).toBeUndefined();
    expect(first?.body.format).toMatchObject({ type: "object" });
    expect(ended.payload).toMatchObject({ citationSupport: "structured-output" });
    expect(ended.event === "finished" && ended.payload.citations).toHaveLength(1);
  });

  test("a model that can call Tools, and isn't small, starts in the Tool loop", async () => {
    const ollama = await startOllamaServer({
      models: [QWEN35_9B],
      reply: answering((_body, index) =>
        index === 0
          ? { toolCalls: [{ name: "search_documents", arguments: { query: "spring tides" } }] }
          : { content: "At new and full moon." },
      ),
    });
    const { core, mind, client } = await setUpLocalModel(ollama, QWEN35_9B.name, {
      documents: [{ name: "Tides.md", contents: TIDES }],
    });
    const asked = question("When are spring tides?");
    writeMind(client, [asked]);

    const { ended } = await askAndEnd(core, client, mind.id, asked.attrs.id);

    expect(ended.payload).toMatchObject({ citationSupport: "tools" });
    const [first] = answerChats(ollama);
    const tools = (first?.body.tools as { function: { name: string } }[]) ?? [];
    expect(tools.map((each) => each.function.name)).toContain("search_documents");
  });

  test("a small model that can call Tools cites with structured output: one search, then its records, its markers placed", async () => {
    const ollama = await startOllamaServer({
      models: [QWEN35],
      reply: answering(() => ({
        // As qwen3.5:4b does: the records, but no marker in the Answer.
        content: JSON.stringify({
          answer: "Spring tides happen at new moon and at full moon.",
          citations: [
            { marker: 1, passage: "P1", quote: "Spring tides happen at new moon and at full moon" },
          ],
        }),
      })),
    });
    const { core, mind, client } = await setUpLocalModel(ollama, QWEN35.name, {
      documents: [{ name: "Tides.md", contents: TIDES }],
    });
    const asked = question("When are spring tides?");
    writeMind(client, [asked]);

    const { ended } = await askAndEnd(core, client, mind.id, asked.attrs.id);

    expect(ended.event).toBe("finished");
    // One request: no Tools offered, the Answer held to the JSON schema.
    expect(answerChats(ollama)).toHaveLength(1);
    const [first] = answerChats(ollama);
    expect(first?.body.tools).toBeUndefined();
    expect(first?.body.format).toMatchObject({ type: "object" });
    expect(ended.payload).toMatchObject({ citationSupport: "structured-output", placedMarkers: 1 });
    expect(ended.event === "finished" && ended.payload.citations).toMatchObject([
      { check: "found" },
    ]);
  });
});
