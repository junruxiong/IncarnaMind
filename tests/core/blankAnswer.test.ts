/**
 * An Answer that ends with no text is a problem the User sees, not a blank
 * Answer (#501).
 */
import { afterEach, describe, expect, test, vi } from "vitest";
import { answerEnded, askNew, setUpWithDocuments } from "../helpers/citations";
import { answering, QWEN35, setUpLocalModel } from "../helpers/localModels";
import { answerIn, answerText } from "../helpers/minds";
import { scriptedModel } from "../helpers/models";
import { startOllamaServer } from "../helpers/ollama";

const TIDES = { name: "Tides.md", contents: "# Tides\n\nSpring tides come at new and full moon." };

afterEach(() => {
  vi.restoreAllMocks();
});

describe("An Answer whose reply ends with no text", () => {
  test("after a search it fails as no-answer, and Try again asks the model again", async () => {
    const warnings = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    let asked = 0;
    const model = scriptedModel(
      (call) => {
        if (call.results.length === 0 && asked++ === 0) {
          return { calls: [{ tool: "search_documents", input: { query: "spring tides" } }] };
        }
        return asked > 1 ? { text: "At new and full moon." } : {};
      },
      { modelId: "gpt-oss:20b-cloud" },
    );
    const { core, client, mind } = await setUpWithDocuments(model, [TIDES]);

    const answerId = await askNew(core, client, mind.id, "When are spring tides?");
    const ended = await answerEnded(core, answerId);

    expect(ended).toMatchObject({ event: "failed", payload: { error: { kind: "no-answer" } } });
    expect(answerIn(client, answerId).attrs).toMatchObject({
      status: "failed",
      errorKind: "no-answer",
    });
    expect(answerText(client, answerId)).toBe("");
    expect(warnings).toHaveBeenCalledWith(expect.stringContaining("model=gpt-oss:20b-cloud"));

    asked = 1;
    expect(await core.regenerateAnswer({ mindId: mind.id, answerId })).toEqual({
      asked: true,
      answerId,
    });
    expect(await answerEnded(core, answerId)).toMatchObject({ event: "finished" });
    expect(answerText(client, answerId)).toBe("At new and full moon.");
  });

  test("with no Documents to search it fails the same way", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const model = scriptedModel(() => ({}));
    const { core, client, mind } = await setUpWithDocuments(model, []);

    const answerId = await askNew(core, client, mind.id, "Anything?");

    expect(await answerEnded(core, answerId)).toMatchObject({
      event: "failed",
      payload: { error: { kind: "no-answer" } },
    });
  });
});

describe("An Ollama reply with lines the provider can't read", () => {
  test("and then nothing fails as no-answer, and the log names the provider and the model but not the line", async () => {
    const warnings = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const ollama = await startOllamaServer({
      models: [QWEN35],
      reply: answering(() => ({
        content: "",
        unreadable: ['{"error":"the runner stopped"}', "SECRET-DOCUMENT-TEXT"],
      })),
    });
    const { core, client, mind } = await setUpLocalModel(ollama, QWEN35.name);

    const answerId = await askNew(core, client, mind.id, "Anything?");

    expect(await answerEnded(core, answerId)).toMatchObject({
      event: "failed",
      payload: { error: { kind: "no-answer" } },
    });
    const logged = warnings.mock.calls.map((call) => call.join(" "));
    expect(logged).toContainEqual(
      expect.stringMatching(new RegExp(`provider=ollama model=${QWEN35.name} kind=error-line`)),
    );
    expect(logged).toContainEqual(expect.stringContaining("kind=not-json"));
    expect(logged.join("\n")).not.toContain("SECRET-DOCUMENT-TEXT");
  });

  test("after some text keeps the text", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const ollama = await startOllamaServer({
      models: [QWEN35],
      reply: answering(() => ({ content: "Spring tides.", unreadable: ["garbage"] })),
    });
    const { core, client, mind } = await setUpLocalModel(ollama, QWEN35.name);

    const answerId = await askNew(core, client, mind.id, "Anything?");

    expect(await answerEnded(core, answerId)).toMatchObject({ event: "finished" });
    expect(answerText(client, answerId)).toBe("Spring tides.");
  });
});
