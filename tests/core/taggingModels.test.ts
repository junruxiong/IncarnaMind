/**
 * The models for the jobs that aren't Answers (#180): the quick-tasks model for
 * an Answer's search query, its translation and automatic tagging, the image
 * model for a scan's pages, and the one Tagging setting, Automatic unless the
 * User picks a model, which Organize uses too. Through the core's public
 * interface, with scripted models standing in for every provider: no network.
 */
import { describe, expect, test, vi } from "vitest";
import type { ChatModelFactory, ChatModelSpec, Core, CoreAdapters } from "../../src/core";
import { answerEnded } from "../helpers/citations";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { note, question, writeMind } from "../helpers/minds";
import { scriptedModel } from "../helpers/models";
import { buildPdf } from "../helpers/pdf";
import { waitForTagging } from "../helpers/tags";

const OPENAI = "https://api.openai.com";
const ANTHROPIC = "https://api.anthropic.com";

/** What one call that doesn't write the Answer asked of which model. */
interface SmallCall {
  kind: string;
  modelId: string;
  thinking: "off" | undefined;
  /** What it was for, from its instructions. */
  purpose: "rewrite" | "translate" | "tags" | "organize" | "other";
}

/**
 * Every provider's models, scripted: a rewrite gives a query, a translation
 * a Chinese one, tagging no Tags, Organize no Folder; an Answer (streamed)
 * a plain one, as JSON for structured output.
 */
function scriptedProviders() {
  const specs: ChatModelSpec[] = [];
  const calls: SmallCall[] = [];
  const answers: string[] = [];
  const createChatModel: ChatModelFactory = (spec) => {
    specs.push(spec);
    return scriptedModel(
      (call) => {
        answers.push(spec.modelId);
        return call.json
          ? { text: JSON.stringify({ answer: "Cities stay warm at night.", citations: [] }) }
          : { text: "Cities stay warm at night." };
      },
      {
        modelId: spec.modelId,
        generate: ({ system, options }) => {
          const schema =
            options.responseFormat?.type === "json"
              ? (options.responseFormat.schema as { properties?: Record<string, unknown> })
              : undefined;
          const purpose: SmallCall["purpose"] = schema?.properties?.groupId
            ? "organize"
            : schema?.properties?.tags
              ? "tags"
              : system.startsWith("You turn a Question")
                ? "rewrite"
                : system.startsWith("You translate a query")
                  ? "translate"
                  : "other";
          calls.push({ kind: spec.kind, modelId: spec.modelId, thinking: spec.thinking, purpose });
          const text = {
            organize: JSON.stringify({ groupId: "__unsorted__", tags: [] }),
            tags: JSON.stringify({ tags: [] }),
            rewrite: "urban heat islands at night",
            translate: "城市热岛 夜间",
            other: "OK",
          }[purpose];
          return { text };
        },
      },
    );
  };
  return { createChatModel, specs, calls, answers };
}

async function start(adapters: Partial<CoreAdapters> = {}) {
  const models = scriptedProviders();
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir, { createChatModel: models.createChatModel, ...adapters });
  return { core, dataDir, ...models };
}

async function addText(core: Core, name: string, text: string) {
  const [document] = await addAndProcess(core, [
    await writeSourceFile(await createTempDataFolder(), name, text),
  ]);
  if (!document) throw new Error("Nothing was added.");
  return document;
}

const automaticOf = async (core: Core) => (await core.getTaggingModels()).automatic;

describe("Automatic picks the model that tags", () => {
  test("with no model at all, nothing: Documents wait for one, and nothing is sent", async () => {
    const { core, calls } = await start();
    expect((await core.getLibrary()).settings.classifier).toEqual({ kind: "automatic" });
    expect(await automaticOf(core)).toBeNull();
    const document = await addText(core, "Notes.md", "Notes from the first meeting.");
    expect(document.tagging).toBe("waiting-for-provider");
    expect(calls).toEqual([]);
  });

  test("1. a Jev key comes first", async () => {
    const { core } = await start();
    await core.saveChatProvider({ kind: "openai", apiKey: "sk-test", modelId: "gpt-6.1-sol" });
    await core.saveJevSettings({ apiKey: "jev-test" });
    expect(await automaticOf(core)).toMatchObject({
      classifier: { kind: "jev" },
      reason: "jev",
      likelihoods: true,
    });
  });

  test.each([
    ["anthropic", "claude-sonnet-5-5", "claude-haiku-5-5", "Claude Haiku 5.5", "quick-model"],
    ["openai", "gpt-6.1-sol", "gpt-6-luna", "GPT-6 Luna", "quick-model"],
    ["mistral", "mistral-medium-2604", "mistral-small-2603", "Mistral Small 4", "tagging-model"],
  ] as const)(
    "2. a %s key: its tagging model, %s → %s",
    async (kind, answers, tagging, name, reason) => {
      const { core } = await start();
      const provider = await core.saveChatProvider({ kind, apiKey: "key", modelId: answers });
      expect(await automaticOf(core)).toMatchObject({
        classifier: { kind: "chat", choice: { providerId: provider.id, modelId: tagging } },
        name,
        reason,
        group: "keys",
        goodForTagging: true,
        likelihoods: false,
        cost: { kind: "estimate", currency: "USD" },
      });
    },
  );

  test("2. a server that names no quick model tags with its model for Answers", async () => {
    const { core } = await start();
    const provider = await core.saveChatProvider({
      kind: "openai-compatible",
      baseUrl: "https://llm.example.com/v1",
      apiKey: "key",
      modelId: "house-model",
    });
    expect(await automaticOf(core)).toMatchObject({
      classifier: { kind: "chat", choice: { providerId: provider.id, modelId: "house-model" } },
      reason: "answers-model",
      goodForTagging: false,
    });
  });

  test("2. never an Ollama cloud model: Ollama's cloud doesn't hold a reply to a schema", async () => {
    const { core } = await start();
    await core.saveChatProvider({ kind: "ollama", modelId: "gpt-oss:20b-cloud" });
    expect(await automaticOf(core)).toBeNull();
    const models = await core.getTaggingModels();
    expect(models.options.map((option) => option.name)).not.toContain("gpt-oss:20b-cloud");
  });

  test("3. a local decision model in Ollama, before the chat model on this computer", async () => {
    const { core } = await start({ localModels: async () => ["qwen3.5:4b", "tev1:4b"] });
    await core.saveChatProvider({ kind: "ollama", modelId: "qwen3.5:4b" });
    expect(await automaticOf(core)).toMatchObject({
      classifier: { kind: "auto" },
      reason: "decision-models",
      decisionModel: true,
      likelihoods: true,
      cost: { kind: "free" },
    });
  });

  test("3. a User whose only model is outside IncarnaMind gets an installed decision model", async () => {
    const { core } = await start({ localModels: async () => ["clef-flash:latest"] });
    expect(await automaticOf(core)).toMatchObject({
      classifier: { kind: "ollama", modelId: "clef-flash", usePageImages: true },
      reason: "decision-models",
    });
  });

  test("4. otherwise the chat model on this computer", async () => {
    const { core } = await start({ localModels: async () => ["qwen3.5:4b"] });
    const provider = await core.saveChatProvider({ kind: "ollama", modelId: "qwen3.5:4b" });
    expect(await automaticOf(core)).toMatchObject({
      classifier: { kind: "chat", choice: { providerId: provider.id, modelId: "qwen3.5:4b" } },
      reason: "local-chat",
      group: "local",
      cost: { kind: "free" },
    });
  });

  test("in local mode only a model on this computer", async () => {
    const { core } = await start({ localModels: async () => ["qwen3.5:4b"] });
    const ollama = await core.saveChatProvider({ kind: "ollama", modelId: "qwen3.5:4b" });
    await core.saveChatProvider({ kind: "anthropic", apiKey: "key", modelId: "claude-sonnet-5-5" });
    expect(await automaticOf(core)).toMatchObject({ reason: "quick-model" });

    await core.setLocalOnly(true);
    expect(await automaticOf(core)).toMatchObject({
      classifier: { kind: "chat", choice: { providerId: ollama.id, modelId: "qwen3.5:4b" } },
      reason: "local-chat",
    });
    expect((await core.getTaggingModels()).localMode).toBe(true);
  });

  test("the hand-picked list groups models as the model chip does, with what 100 Documents cost", async () => {
    const { core } = await start({ localModels: async () => ["tev1:4b", "tev1:0.8b"] });
    await core.saveChatProvider({ kind: "ollama", modelId: "qwen3.5:4b" });
    await core.saveChatProvider({ kind: "anthropic", apiKey: "key", modelId: "claude-sonnet-5-5" });
    const { options } = await core.getTaggingModels();
    expect(options.map((option) => [option.group, option.name])).toEqual([
      ["keys", "Claude Haiku 5.5"],
      ["keys", "Claude Sonnet 5.5"],
      ["local", "Tev1, Clef-Flash"],
      ["local", "Tev1 4B"],
      ["local", "Tev1 0.8B"],
    ]);
    const [haiku, sonnet] = options;
    expect(haiku).toMatchObject({ goodForTagging: true, provider: { en: "Anthropic" } });
    expect(sonnet).toMatchObject({ goodForTagging: false });
    // About 1,500 tokens in and 25 out a Document, at the catalog's price.
    if (haiku?.cost.kind !== "estimate" || sonnet?.cost.kind !== "estimate")
      throw new Error("No estimate.");
    expect(haiku.cost.amount).toBeCloseTo(0.016, 3);
    expect(sonnet.cost.amount).toBeGreaterThan(10 * haiku.cost.amount);
    expect(options.filter((option) => option.decisionModel)).toHaveLength(3);
  });
});

describe("Which model each quick call uses", () => {
  test("an Answer's search query and its translation go to the quick-tasks model; the Answer to its own", async () => {
    const { core, calls, answers, specs } = await start();
    await core.saveChatProvider({ kind: "openai", apiKey: "sk-test", modelId: "gpt-3.5-turbo" });
    await core.allowDataFlow("chat", OPENAI);
    await core.allowDataFlow("tagging", OPENAI);
    await addText(
      core,
      "城市热岛.md",
      "城市热岛效应在夜间最强。公园可以降低周边的气温。".repeat(10),
    );
    const mind = await core.createMind({ title: "Heat" });
    const client = await connectToMind(core, mind.id);
    const asked = question("How do they change at night?");
    writeMind(client, [
      note("Urban heat islands are cities warmer than the land around them."),
      asked,
    ]);
    await client.settled();

    const result = await core.askQuestion({ mindId: mind.id, questionId: asked.attrs.id });
    if (!result.asked) throw new Error("Not asked.");
    const ended = await answerEnded(core, result.answerId);

    expect(ended.event).toBe("finished");
    // GPT-3.5 Turbo can't call Tools: it answers in one go, after one search, rewritten and translated.
    const small = calls.filter(
      (each) => each.purpose === "rewrite" || each.purpose === "translate",
    );
    expect(small).toEqual([
      { kind: "openai", modelId: "gpt-6-luna", thinking: "off", purpose: "rewrite" },
      { kind: "openai", modelId: "gpt-6-luna", thinking: "off", purpose: "translate" },
    ]);
    expect(answers).toEqual(["gpt-3.5-turbo"]);
    expect(specs.find((spec) => spec.modelId === "gpt-3.5-turbo")?.thinking).toBeUndefined();
    // Counted as the Answer's: its own call and the two small ones.
    if (ended.event !== "finished") throw new Error("Not finished.");
    expect(ended.payload.usage.inputTokens).toBe(15);
  });

  test("Tag suggestions, and Organize once there are Folders, go to the provider's tagging model", async () => {
    const { core, calls } = await start();
    await core.saveChatProvider({ kind: "openai", apiKey: "sk-test", modelId: "gpt-6.1-sol" });
    await core.allowDataFlow("tagging", OPENAI);
    await core.allowDataFlow("classification", OPENAI);
    const first = await addText(core, "Report.md", "Quarterly report on sales.");
    await waitForTagging(core, [first.id]);
    expect(calls.filter((each) => each.purpose === "tags")).toEqual([
      { kind: "openai", modelId: "gpt-6-luna", thinking: "off", purpose: "tags" },
    ]);

    await core.createLibraryGroup({ name: "Reports", description: "Reports and reviews" });
    const second = await addText(core, "Review.md", "Annual review.");
    await vi.waitFor(async () =>
      expect(
        (await core.getLibrary()).assignments.find((each) => each.documentId === second.id)?.status,
      ).toBe("classified"),
    );
    expect(calls.filter((each) => each.purpose === "organize")).toEqual([
      { kind: "openai", modelId: "gpt-6-luna", thinking: "off", purpose: "organize" },
    ]);
  });

  test("a quick-tasks model the User sets does the small calls", async () => {
    const { core, calls } = await start();
    const provider = await core.saveChatProvider({
      kind: "openai",
      apiKey: "sk-test",
      modelId: "gpt-6.1-sol",
    });
    await core.updateSettings({
      user: { quickTasksModel: { providerId: provider.id, modelId: "gpt-5.4-nano" } },
    });
    await core.allowDataFlow("tagging", OPENAI);
    const document = await addText(core, "Report.md", "Quarterly report on sales.");
    await waitForTagging(core, [document.id]);
    expect(calls.map((each) => each.modelId)).toEqual(["gpt-5.4-nano"]);
    expect(await automaticOf(core)).toMatchObject({ name: "GPT-5.4 nano", reason: "quick-model" });
  });

  test("with no quick-tasks model, the model for Answers tags", async () => {
    const { core, calls } = await start();
    await core.saveChatProvider({
      kind: "openai-compatible",
      baseUrl: "https://llm.example.com/v1",
      apiKey: "key",
      modelId: "house-model",
    });
    await core.allowDataFlow("tagging", "https://llm.example.com");
    const document = await addText(core, "Report.md", "Quarterly report on sales.");
    await waitForTagging(core, [document.id]);
    expect(calls.map((each) => each.modelId)).toEqual(["house-model"]);
  });
});

describe("Local mode", () => {
  test("makes the quick-tasks, image and tagging models local or off, with the existing explanation", async () => {
    const { core, calls } = await start();
    const anthropic = await core.saveChatProvider({
      kind: "anthropic",
      apiKey: "key",
      modelId: "claude-sonnet-5-5",
    });
    await core.createLibraryGroup({ name: "Scans", description: "Scanned papers" });
    // Chosen by hand: a cloud model, which can't be used in local mode.
    await core.saveLibrarySettings({
      classifier: {
        kind: "chat",
        choice: { providerId: anthropic.id, modelId: "claude-haiku-5-5" },
      },
      automatic: true,
    });
    await core.setLocalOnly(true);
    const [text] = await addAndProcess(core, [
      await writeSourceFile(await createTempDataFolder(), "Notes.md", "Notes on scans."),
    ]);
    if (!text) throw new Error("Not added.");
    await vi.waitFor(async () =>
      expect(
        (await core.getLibrary()).assignments.find((each) => each.documentId === text.id),
      ).toMatchObject({
        status: "waiting",
        error: { message: "Cloud classification is paused in local mode." },
      }),
    );
    // Automatic: no model on this computer, so nothing tags, and no small call leaves it.
    await core.saveLibrarySettings({ classifier: { kind: "automatic" }, automatic: true });
    expect(await automaticOf(core)).toBeNull();
    expect(calls).toEqual([]);
    expect(await core.listConsentRequests()).toEqual([]);
  });
});

describe("The image model reads scans the model that tags can't", () => {
  const SCAN = buildPdf([{ image: true }]);

  test("a scan goes to the image model the User set; in local mode, to none, so it waits", async () => {
    const { core, calls, specs } = await start();
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const anthropic = await core.saveChatProvider({
      kind: "anthropic",
      apiKey: "key",
      modelId: "claude-sonnet-5-5",
    });
    const ollama = (await core.listChatProviders()).find((each) => each.kind === "ollama");
    if (!ollama) throw new Error("No Ollama.");
    await core.updateSettings({
      user: {
        chatModel: { providerId: ollama.id, modelId: "local-model" },
        imageModel: { providerId: anthropic.id, modelId: "claude-haiku-5-5" },
      },
    });
    await core.allowDataFlow("classification", ANTHROPIC);
    await core.createLibraryGroup({ name: "Scans", description: "Scanned papers" });
    const [scan] = await addAndProcess(core, [
      await writeSourceFile(await createTempDataFolder(), "scan.pdf", SCAN),
    ]);
    if (!scan) throw new Error("Not added.");
    await vi.waitFor(
      async () =>
        expect(
          (await core.getLibrary()).assignments.find((each) => each.documentId === scan.id),
        ).toMatchObject({ status: "classified", model: { id: "claude-haiku-5-5", images: true } }),
      { timeout: 15_000 },
    );
    expect(calls.filter((each) => each.purpose === "organize").map((each) => each.modelId)).toEqual(
      ["claude-haiku-5-5"],
    );
    expect(specs.some((spec) => spec.modelId === "local-model")).toBe(true);

    await core.setLocalOnly(true);
    const [second] = await addAndProcess(core, [
      await writeSourceFile(
        await createTempDataFolder(),
        "scan2.pdf",
        buildPdf([{ image: true }, { image: true }]),
      ),
    ]);
    if (!second) throw new Error("Not added.");
    await core.classifyDocuments([second.id]);
    expect(
      (await core.getLibrary()).assignments.find((each) => each.documentId === second.id)?.status,
    ).toBe("waiting");
    expect(calls.filter((each) => each.purpose === "organize")).toHaveLength(1);
  });
});

describe("Moving to one Tagging setting", () => {
  test("a classifier chosen before is kept; a Library that chose none, or Organize manually, gets Automatic", async () => {
    const dataDir = await createTempDataFolder();
    const before = startCore(dataDir);
    const kept = {
      classifier: {
        kind: "ollama" as const,
        baseUrl: "http://127.0.0.1:11434",
        modelId: "tev1:0.8b",
      },
      automatic: false,
    };
    await before.saveLibrarySettings(kept);
    before.close();
    const after = startCore(dataDir);
    expect((await after.getLibrary()).settings).toEqual(kept);
    after.close();

    // As the version before stored "Organize manually".
    queryDatabase(
      dataDir,
      "UPDATE device_settings SET value = ? WHERE key = 'library' AND deleted_at IS NULL RETURNING key",
      [JSON.stringify({ classifier: null, automatic: true })],
    );
    const upgraded = startCore(dataDir);
    expect((await upgraded.getLibrary()).settings).toEqual({
      classifier: { kind: "automatic" },
      automatic: true,
    });
  });

  test("null still saves as Automatic, for a caller that knew Organize manually", async () => {
    const { core } = await start();
    await core.saveLibrarySettings({ classifier: null, automatic: false });
    expect((await core.getLibrary()).settings).toEqual({
      classifier: { kind: "automatic" },
      automatic: false,
    });
  });
});
