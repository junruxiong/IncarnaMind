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
import { buildPdf } from "../helpers/pdf";

/** Two short pages: one Passage covers both. */
const TIDES_PDF = buildPdf([
  { lines: ["Tides and the Moon", "Most coasts see two high tides every day."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);

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

  test("a small model's record that names its Passage by number, and the wrong page of it, is cited where its quote is", async () => {
    const ollama = await startOllamaServer({
      models: [QWEN35],
      reply: answering(() => ({
        content: JSON.stringify({
          answer: "Spring tides come at new and full moon [^1]. Neap tides are smaller [^2].",
          citations: [
            // "1" for P1, and p. 1 for a quote on p. 2.
            {
              marker: 1,
              passage: "1",
              location: "p. 1",
              quote: "Spring tides happen at new moon and at full moon.",
            },
            // A quote that is on none of P1's pages stays "not found".
            { marker: 2, passage: "P1", location: "p. 2", quote: "Neap tides are the smallest." },
          ],
        }),
      })),
    });
    const { core, mind, client } = await setUpLocalModel(ollama, QWEN35.name, {
      documents: [{ name: "Tides.pdf", contents: TIDES_PDF }],
    });
    const asked = question("When are spring tides?");
    writeMind(client, [asked]);

    const { ended } = await askAndEnd(core, client, mind.id, asked.attrs.id);

    expect(ended.event).toBe("finished");
    // One request: nothing goes back to the model to fix its records.
    expect(answerChats(ollama)).toHaveLength(1);
    expect(ended.event === "finished" && ended.payload.citations).toMatchObject([
      { pageFrom: 2, pageTo: 2, location: { kind: "page", from: 2, to: 2 }, check: "found" },
      { pageFrom: 2, pageTo: 2, check: "not-found", checkReason: "quote-not-on-pages" },
    ]);
  });
});
