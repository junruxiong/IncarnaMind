/**
 * What an Answer's meta line says it is doing ("answer.phase" events):
 * waiting for the User to allow its Question to go to a cloud service, the
 * local model loading, searching the Documents, writing. Against a stub of
 * Ollama's API, which takes a while to load a model, and a scripted cloud model.
 */
import { describe, expect, test } from "vitest";
import type { Core, CoreEvents } from "../../src/core";
import { translate } from "../../src/shared/i18n";
import { answerEnded } from "../helpers/citations";
import { createTempDataFolder, nextEvent, startCore } from "../helpers/core";
import { answering, askAndEnd, QWEN35, setUpLocalModel, TIDES } from "../helpers/localModels";
import { connectToMind } from "../helpers/mindClient";
import { question, writeMind } from "../helpers/minds";
import { scriptedModels, streamingModel } from "../helpers/models";
import { startOllamaServer } from "../helpers/ollama";

/** A Mind with a Question asked of a cloud model whose chat flow the User hasn't allowed yet. */
async function askACloudModel() {
  const models = scriptedModels(streamingModel("A notebook."));
  const core = startCore(await createTempDataFolder(), { createChatModel: models.createChatModel });
  await core.saveChatProvider({ kind: "openai", apiKey: "sk-test", modelId: "gpt-test" });
  const mind = await core.createMind({ title: "Cloud" });
  const client = await connectToMind(core, mind.id);
  const asked = question("What is a Mind?");
  writeMind(client, [asked]);
  await client.settled();
  const phases = phasesOf(core);
  const requested = nextEvent(core, "consent.requested");
  const result = await core.askQuestion({ mindId: mind.id, questionId: asked.attrs.id });
  if (!result.asked) throw new Error(`The Question wasn't asked: ${JSON.stringify(result)}`);
  return { core, models, phases, request: await requested, answerId: result.answerId };
}

/** The phases Answers go through, in order. */
const phasesOf = (core: Core) => {
  const phases: CoreEvents["answer.phase"]["phase"][] = [];
  core.on("answer.phase", ({ phase }) => phases.push(phase));
  return phases;
};

describe("What the meta line shows while an Answer is written", () => {
  test("loading while Ollama loads the model, searching the Documents, then writing", async () => {
    const ollama = await startOllamaServer({
      models: [QWEN35],
      reply: answering((_body, index) =>
        index === 0
          ? { toolCalls: [{ name: "search_documents", arguments: { query: "spring tides" } }] }
          : { content: "At new and full moon." },
      ),
      // A request waits for its model to load.
      loadMs: 300,
    });
    const { core, mind, client } = await setUpLocalModel(ollama, QWEN35.name, {
      documents: [{ name: "Tides.md", contents: TIDES }],
    });
    // Tagging the Document loaded the model; since then, Ollama has unloaded it.
    ollama.loaded.splice(0);
    const phases = phasesOf(core);
    const asked = question("When are spring tides?");
    writeMind(client, [asked]);

    await askAndEnd(core, client, mind.id, asked.attrs.id);

    // Nothing was loaded: the model loads, starts, searches, then writes.
    expect(phases).toEqual(["loading", "writing", "searching", "writing"]);
  });

  test("a model Ollama has loaded already goes straight to writing", async () => {
    const ollama = await startOllamaServer({
      models: [QWEN35],
      reply: () => ({ content: "A notebook." }),
    });
    ollama.loaded.push({ name: QWEN35.name, context_length: 16_384 });
    const { core, mind, client } = await setUpLocalModel(ollama, QWEN35.name);
    const phases = phasesOf(core);
    const asked = question("What is a Mind?");
    writeMind(client, [asked]);

    await askAndEnd(core, client, mind.id, asked.attrs.id);

    expect(phases).toEqual(["writing"]);
  });

  test("while the User is asked whether to send the Question to a cloud service, the Answer waits for their permission", async () => {
    const { core, models, phases, request, answerId } = await askACloudModel();

    // Nothing has gone to the service: the Answer waits for the User.
    expect(phases).toEqual(["waiting-for-consent"]);
    expect(models.model.doStreamCalls).toHaveLength(0);

    const ended = answerEnded(core, answerId);
    await core.respondToConsent(request.requestId, true);

    expect((await ended).event).toBe("finished");
    expect(phases).toEqual(["waiting-for-consent", "writing"]);
  });

  test("declined, the Answer stops waiting and says nothing was sent", async () => {
    const { core, models, phases, request, answerId } = await askACloudModel();

    const ended = answerEnded(core, answerId);
    await core.respondToConsent(request.requestId, false);

    expect(await ended).toMatchObject({
      event: "failed",
      payload: { error: { kind: "consent-declined" } },
    });
    expect(phases).toEqual(["waiting-for-consent"]);
    expect(models.model.doStreamCalls).toHaveLength(0);
  });

  test("each phase is said in English and in Chinese", () => {
    expect(translate("en", "answer.phase.waiting-for-consent")).toBe(
      "Waiting for your permission to send",
    );
    expect(translate("en", "answer.phase.loading")).toBe("Loading the model…");
    expect(translate("en", "answer.phase.searching")).toBe("Searching your Documents…");
    expect(translate("en", "answer.phase.writing")).toBe("Writing…");
    expect(translate("zh-CN", "answer.phase.waiting-for-consent")).toBe("等待你允许发送");
    expect(translate("zh-CN", "answer.phase.loading")).toBe("正在加载模型…");
    expect(translate("zh-CN", "answer.phase.searching")).toBe("正在搜索你的文档…");
    expect(translate("zh-CN", "answer.phase.writing")).toBe("正在回答…");
  });
});
