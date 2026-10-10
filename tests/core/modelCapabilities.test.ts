import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { describe, expect, onTestFinished, test, vi } from "vitest";
import { type Core, DATABASE_FILE } from "../../src/core";
import { ANSWER_TEMPERATURE } from "../../src/core/answers/engine";
import { givesStructuredOutput, readsImages } from "../../src/core/providers/capabilities";
import { anthropicModelFacts } from "../../src/core/providers/modelLists";
import { migrate, openDatabase } from "../../src/core/storage";
import { migrations } from "../../src/core/storage/migrations";
import { askAndFinish, setUpWithDocuments } from "../helpers/citations";
import {
  createMemoryKeychain,
  createTempDataFolder,
  queryDatabase,
  startCore,
} from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { note, question, writeMind } from "../helpers/minds";
import {
  promptOf,
  replyingModel,
  scriptedModel,
  scriptedModels,
  streamingModel,
} from "../helpers/models";

/** Starts a core with a cloud provider the User has allowed Questions to go to. */
async function startWithCloudModel(
  provider: { kind: "openai" | "anthropic" | "google"; modelId: string },
  model = streamingModel("Twice a day."),
) {
  const models = scriptedModels(model);
  const core = startCore(await createTempDataFolder(), { createChatModel: models.createChatModel });
  const saved = await core.saveChatProvider({ ...provider, apiKey: "sk-test" });
  if (saved.service) await core.allowDataFlow("chat", saved.service.id);
  return { core, models, model, provider: saved };
}

/** Answers `fetch` for one URL with a JSON body; every other request goes through. */
function answerFetch(url: string, body: unknown) {
  const original = globalThis.fetch;
  const fetch = vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
    const asked = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    return asked === url
      ? Promise.resolve(Response.json(body))
      : original(input as Parameters<typeof original>[0], init);
  });
  onTestFinished(() => fetch.mockRestore());
  return fetch;
}

/** Asks a Question below `above` in a new Mind and waits for its Answer. */
async function askBelow(core: Core, above: string[], text: string) {
  const mind = await core.createMind({ title: "Tides" });
  const client = await connectToMind(core, mind.id);
  const asked = question(text);
  writeMind(client, [...above.map((each) => note(each)), asked]);
  await client.settled();
  const result = await core.askQuestion({ mindId: mind.id, questionId: asked.attrs.id });
  if (!result.asked) throw new Error("The Question wasn't asked.");
  const answerId = result.answerId;
  await new Promise<void>((resolve, reject) => {
    const stops = [
      core.on("answer.finished", (payload) => {
        if (payload.answerId !== answerId) return;
        for (const stop of stops) stop();
        resolve();
      }),
      core.on("answer.failed", (payload) => {
        if (payload.answerId !== answerId) return;
        for (const stop of stops) stop();
        reject(new Error(JSON.stringify(payload.error)));
      }),
    ];
  });
}

describe("Models in the app, by the catalog", () => {
  test("image input, structured output and context come from the catalog when a model is prepared", async () => {
    const { core } = await startWithCloudModel({ kind: "anthropic", modelId: "claude-sonnet-5-5" });
    const prepared = await core.prepareChatModel();
    expect(prepared.capabilities.known).toBe(true);
    expect(readsImages(prepared.capabilities)).toBe(true);
    expect(givesStructuredOutput(prepared.capabilities)).toBe(true);
    expect(prepared.support).toBe("tools");
    expect(prepared.window).toEqual({ tokens: 1_000_000, outputTokens: 128_000 });
    // A cloud model's quotes are never asked for again (ADR-0007).
    expect(prepared.retryQuotes).toBeUndefined();
  });

  test("an unknown model id on a known provider works, and is reported as capabilities unknown", async () => {
    const { core, model } = await startWithCloudModel({
      kind: "openai",
      modelId: "gpt-7-preview",
    });
    const prepared = await core.prepareChatModel();
    expect(prepared.capabilities).toEqual({ known: false, facts: {} });
    expect(prepared.window).toBeUndefined();
    expect(prepared.support).toBeUndefined();

    await askBelow(core, [], "How often are high tides?");
    expect(model.doStreamCalls[0]?.temperature).toBe(ANSWER_TEMPERATURE);

    answerFetch("https://api.openai.com/v1/models", { data: [] });
    const [group] = await core.listChatModels();
    expect(group).toMatchObject({ models: ["gpt-7-preview"], unknown: ["gpt-7-preview"] });
  });

  test("a model's requests are kept within the context window the catalog gives it", async () => {
    const oldest = `Oldest note. ${"tide ".repeat(1_400)}`;
    const newest = `Newest note. ${"moon ".repeat(1_400)}`;

    // GPT-4 has an 8,192-token window: the oldest note doesn't fit in it.
    const small = await startWithCloudModel({ kind: "openai", modelId: "gpt-4" });
    await askBelow(small.core, [oldest, newest], "How often are high tides?");
    const sent = promptOf(small.model)
      .map((message) => message.text)
      .join("\n");
    expect(sent).toContain("How often are high tides?");
    expect(sent).not.toContain("Oldest note.");

    // A model whose window isn't known gets the Question context as it is.
    const unknown = await startWithCloudModel({ kind: "openai", modelId: "gpt-7-preview" });
    await askBelow(unknown.core, [oldest, newest], "How often are high tides?");
    const all = promptOf(unknown.model)
      .map((message) => message.text)
      .join("\n");
    expect(all).toContain("Oldest note.");
    expect(all).toContain("Newest note.");
  });

  test("a model the catalog says gives no structured output isn't tried with it when it refuses Tools", async () => {
    // GPT-4 Turbo calls Tools but has no JSON schema output; here its provider refuses the Tools.
    const model = scriptedModel((call) =>
      call.tools.length > 0
        ? { error: { status: 400, message: "This model does not support tools" } }
        : { text: "Twice a day." },
    );
    const { core, mind, client } = await setUpWithDocuments(model, [
      { name: "Tides.md", contents: "# Tides\n\nHigh tides come twice a day.\n" },
    ]);
    const provider = await core.saveChatProvider({
      kind: "openai",
      apiKey: "sk-test",
      modelId: "gpt-4-turbo",
    });
    if (provider.service) await core.allowDataFlow("chat", provider.service.id);

    const { finished } = await askAndFinish(core, client, mind.id, "How often are high tides?");

    expect(finished.citationSupport).toBe("none");
    expect(model.doStreamCalls.some((call) => call.responseFormat?.type === "json")).toBe(false);
  });
});

describe("A provider's model list", () => {
  test("leaves out what the catalog knows isn't a chat model, and marks what nothing says anything about", async () => {
    const { core } = await startWithCloudModel({ kind: "openai", modelId: "gpt-6.1-sol" });
    answerFetch("https://api.openai.com/v1/models", {
      data: [
        { id: "gpt-6.1-sol" },
        { id: "gpt-6-luna" },
        { id: "text-embedding-3-small" },
        { id: "whisper-1" },
        { id: "gpt-7-preview" },
      ],
    });
    const [listed] = await core.listChatModels();
    expect(listed).toMatchObject({
      models: ["gpt-6.1-sol", "gpt-6-luna", "gpt-7-preview"],
      unknown: ["gpt-7-preview"],
    });
  });

  test("Anthropic's says what a model can do, which counts before the catalog", async () => {
    expect(
      anthropicModelFacts({
        id: "claude-next",
        max_input_tokens: 1_000_000,
        max_tokens: 128_000,
        capabilities: {
          image_input: { supported: true },
          pdf_input: { supported: true },
          structured_outputs: { supported: false },
        },
      }),
    ).toEqual({
      context: 1_000_000,
      maxOutput: 128_000,
      input: ["text", "image", "pdf"],
      structuredOutput: "none",
    });
    expect(anthropicModelFacts({ id: "claude-old" })).toEqual({});

    // A model the catalog doesn't know yet, as Anthropic describes it.
    const { core, provider } = await startWithCloudModel({
      kind: "anthropic",
      modelId: "claude-sonnet-5-5",
    });
    answerFetch("https://api.anthropic.com/v1/models?limit=100", {
      data: [
        { id: "claude-sonnet-5-5" },
        {
          id: "claude-next",
          max_input_tokens: 2_000_000,
          max_tokens: 128_000,
          capabilities: { image_input: { supported: true } },
        },
      ],
    });
    const [group] = await core.listChatModels();
    expect(group).toMatchObject({ models: ["claude-sonnet-5-5", "claude-next"], unknown: [] });
    const next = await core.prepareChatModel({ providerId: provider.id, modelId: "claude-next" });
    expect(readsImages(next.capabilities)).toBe(true);
    expect(next.window).toEqual({ tokens: 2_000_000, outputTokens: 128_000 });
  });
});

describe("Providers saved before the catalog", () => {
  test("keep working after the migration, keys included", async () => {
    const dataDir = await createTempDataFolder();
    const at = "2026-10-01T09:00:00.000Z";
    const ids = {
      openai: randomUUID(),
      anthropic: randomUUID(),
      google: randomUUID(),
      compatible: randomUUID(),
      ollama: randomUUID(),
      chatgpt: randomUUID(),
    };
    // The database as the version before the catalog left it.
    const db = openDatabase(join(dataDir, DATABASE_FILE));
    migrate(
      db,
      migrations.filter((migration) => migration.version < 29),
    );
    const rows: [string, string, string | null][] = [
      [ids.openai, "openai", null],
      [ids.anthropic, "anthropic", null],
      [ids.google, "google", null],
      [ids.compatible, "openai-compatible", "https://api.deepseek.com/v1"],
      [ids.ollama, "ollama", "http://127.0.0.1:11434"],
      [ids.chatgpt, "chatgpt", null],
    ];
    for (const [id, kind, baseUrl] of rows) {
      db.run(
        `INSERT INTO chat_providers (id, kind, base_url, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?)`,
        [id, kind, baseUrl, at, at],
      );
    }
    db.run(
      `INSERT INTO user_settings (id, key, value, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
      [
        randomUUID(),
        "chatModel",
        JSON.stringify({ providerId: ids.openai, modelId: "gpt-6.1-sol" }),
        at,
        at,
      ],
    );
    db.close();
    const keychain = createMemoryKeychain();
    await keychain.set(`chat-provider:${ids.openai}:api-key`, "sk-openai");
    await keychain.set(`chat-provider:${ids.anthropic}:api-key`, "sk-ant");
    await keychain.set(`chat-provider:${ids.google}:api-key`, "google-key");
    await keychain.set(`chat-provider:${ids.compatible}:api-key`, "sk-deepseek");

    const models = scriptedModels(replyingModel());
    const core = startCore(dataDir, { keychain, createChatModel: models.createChatModel });

    expect(await core.listChatProviders()).toEqual([
      {
        id: ids.openai,
        kind: "openai",
        catalogId: "openai",
        endpoint: "global",
        baseUrl: null,
        hasApiKey: true,
        service: { id: "https://api.openai.com", name: "OpenAI" },
      },
      {
        id: ids.anthropic,
        kind: "anthropic",
        catalogId: "anthropic",
        endpoint: "global",
        baseUrl: null,
        hasApiKey: true,
        service: { id: "https://api.anthropic.com", name: "Anthropic" },
      },
      {
        id: ids.google,
        kind: "google",
        catalogId: "google",
        endpoint: "global",
        baseUrl: null,
        hasApiKey: true,
        service: { id: "https://generativelanguage.googleapis.com", name: "Google" },
      },
      {
        id: ids.compatible,
        kind: "openai-compatible",
        catalogId: "openai-compatible",
        endpoint: null,
        baseUrl: "https://api.deepseek.com/v1",
        hasApiKey: true,
        service: { id: "https://api.deepseek.com", name: "api.deepseek.com" },
      },
      {
        id: ids.ollama,
        kind: "ollama",
        catalogId: "ollama",
        endpoint: null,
        baseUrl: "http://127.0.0.1:11434",
        hasApiKey: false,
        service: null,
      },
    ]);
    // The ChatGPT plan, off here, has no provider in the catalog.
    expect(
      queryDatabase<{ catalog_id: string | null; endpoint: string | null }>(
        dataDir,
        "SELECT catalog_id, endpoint FROM chat_providers WHERE id = ?",
        [ids.chatgpt],
      ),
    ).toEqual([{ catalog_id: null, endpoint: null }]);

    // The default model is ready, and its key goes with it.
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, modelId: "gpt-6.1-sol" });
    await core.allowDataFlow("chat", "https://api.openai.com");
    const prepared = await core.prepareChatModel();
    expect(models.specs.at(-1)).toEqual({
      kind: "openai",
      baseUrl: "https://api.openai.com/v1",
      apiKey: "sk-openai",
      modelId: "gpt-6.1-sol",
    });
    expect(prepared.capabilities.known).toBe(true);
    const deepseek = { providerId: ids.compatible, modelId: "deepseek-chat" };
    await core.allowDataFlow("chat", "https://api.deepseek.com");
    await core.prepareChatModel(deepseek);
    expect(models.specs.at(-1)).toMatchObject({
      kind: "openai-compatible",
      baseUrl: "https://api.deepseek.com/v1",
      apiKey: "sk-deepseek",
    });

    // Saving one of them again finds its row, and keeps its key.
    const anthropic = await core.saveChatProvider({
      kind: "anthropic",
      modelId: "claude-sonnet-5-5",
    });
    expect(anthropic).toMatchObject({ id: ids.anthropic, hasApiKey: true });
    const compatible = await core.saveChatProvider({
      kind: "openai-compatible",
      baseUrl: "https://api.deepseek.com/v1",
      modelId: "deepseek-chat",
    });
    expect(compatible).toMatchObject({ id: ids.compatible, hasApiKey: true });
    expect(await core.listChatProviders()).toHaveLength(5);
  });
});
