/**
 * Presets on this computer (LM Studio, Unsloth Studio) are on the shared OpenAI-compatible path at
 * loopback addresses, which count as on this computer. `fetch` is replaced by a stand-in; no port is bound.
 */
import { afterEach, describe, expect, test, vi } from "vitest";
import { InvalidInputError } from "../../src/core";
import { serviceFor, serviceForUrl } from "../../src/core/providers/kinds";
import { createLocalServers } from "../../src/core/providers/localServers";
import { createAiSdkChatModel } from "../../src/core/providers/models";
import { askAndFinish, citationsIn, setUpWithDocuments } from "../helpers/citations";
import { fakeProvider, type RecordedRequest } from "../helpers/fakeProvider";
import { replyingModel } from "../helpers/models";
import { buildPdf } from "../helpers/pdf";

afterEach(() => vi.unstubAllGlobals());

const TIDES = buildPdf([
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);
const SPRING = "Spring tides happen at new moon and at full moon.";

const LM_STUDIO = "127.0.0.1:1234";
const UNSLOTH = "127.0.0.1:8888";

const QWEN = "qwen/qwen3.5-4b";
const GEMMA = "google/gemma-4-12b";

/** LM Studio's `/api/v1/models`, beside the OpenAI-shaped `/v1/models` one fake serves both from. */
const LM_STUDIO_MODELS = {
  data: [{ id: QWEN }, { id: GEMMA }],
  models: [
    {
      type: "llm",
      key: QWEN,
      display_name: "Qwen3.5 4B",
      max_context_length: 262_144,
      loaded_instances: [{ id: QWEN, config: { context_length: 8_192 } }],
      capabilities: { vision: false, trained_for_tool_use: true },
    },
    {
      type: "llm",
      key: GEMMA,
      display_name: "Gemma 4 12B",
      max_context_length: 131_072,
      loaded_instances: [],
      capabilities: { vision: true, trained_for_tool_use: false },
    },
    {
      type: "embedding",
      key: "text-embedding-nomic-embed-text-v1.5",
      max_context_length: 2_048,
      loaded_instances: [],
    },
  ],
};

const serve = (host: string, models: unknown) =>
  fakeProvider(host, {
    protocol: "chat",
    models,
    query: "spring tides",
    records: (passages) => [
      { marker: 1, passage: passages[0]?.id, pageFrom: 1, pageTo: 1, quote: SPRING },
    ],
    answer: "Spring tides come at new and full moon [^1].",
  });

const posts = (requests: RecordedRequest[]) => requests.filter((each) => each.method === "POST");

const setUp = (files: { name: string; contents: Uint8Array }[] = []) =>
  setUpWithDocuments(replyingModel(), files, { createChatModel: createAiSdkChatModel });

describe("What is on this computer", () => {
  test.each([
    "http://127.0.0.1:1234/v1",
    "http://127.1.2.3:8888",
    "http://localhost:1234/v1",
    "http://lm.localhost:1234",
    "http://[::1]:1234/v1",
  ])("%s is, and sends nothing to a service", (address) => {
    expect(serviceForUrl(address)).toBeNull();
  });

  test.each([
    "http://192.168.1.20:1234/v1",
    "http://10.0.0.5:8888",
    "https://studio.example.com/v1",
    "http://127.0.0.1.example.com:1234",
    "http://localhost.example.com:1234",
  ])("%s isn't: it is a service of its own, by its origin", (address) => {
    const url = new URL(address);
    expect(serviceForUrl(address)).toEqual({ id: url.origin, name: url.host });
  });

  test("a preset is on this computer by the address it is set up at", () => {
    const at = (address: string, catalogId = "lmstudio") =>
      serviceFor("openai-compatible", `http://${address}/v1`, "local", catalogId);
    expect(at("127.0.0.1:1234")).toBeNull();
    expect(at("localhost:8888", "unsloth")).toBeNull();
    expect(at("[::1]:1234")).toBeNull();
    expect(at("192.168.1.20:1234")).toEqual({ id: "http://192.168.1.20:1234", name: "LM Studio" });
    expect(serviceFor("openai-compatible", null, "global", "groq")).toEqual({
      id: "https://api.groq.com",
      name: "Groq",
    });
    // Ollama and a server the User gives follow the same rule.
    expect(serviceFor("ollama", "http://127.0.0.1:11434")).toBeNull();
    expect(serviceFor("openai-compatible", "http://localhost:5000/v1")).toBeNull();
    expect(serviceFor("openai-compatible", "http://192.168.1.9:5000/v1")).toEqual({
      id: "http://192.168.1.9:5000",
      name: "192.168.1.9:5000",
    });
  });
});

describe("LM Studio", () => {
  test("is saved with no key, at its own address, as a model on this computer", async () => {
    const { core } = await setUp();
    const saved = await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "lmstudio",
      modelId: QWEN,
    });

    expect(saved).toMatchObject({
      kind: "openai-compatible",
      catalogId: "lmstudio",
      endpoint: "local",
      baseUrl: "http://127.0.0.1:1234/v1",
      hasApiKey: false,
      service: null,
    });
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "not-required" });
    expect(await core.listDataFlows()).toEqual([]);
  });

  test("another address, or another port, is a provider of its own; only a host and port is taken", async () => {
    const { core } = await setUp();
    const save = (endpointValue?: string) =>
      core.saveChatProvider({
        kind: "openai-compatible",
        catalogId: "lmstudio",
        ...(endpointValue !== undefined && { endpointValue }),
        modelId: QWEN,
      });
    const first = await save();
    const other = await save("127.0.0.1:1235");

    expect(other.id).not.toBe(first.id);
    expect(other).toMatchObject({ baseUrl: "http://127.0.0.1:1235/v1", service: null });
    expect((await save("127.0.0.1:1234")).id).toBe(first.id);
    for (const wrong of ["evil.example.com/x", "user@host:1234", "host:12345678", "http://host"]) {
      await expect(save(wrong)).rejects.toThrow(InvalidInputError);
    }
  });

  test("lists its language models from its own list, and reads what each can do from it", async () => {
    const { core } = await setUp();
    const saved = await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "lmstudio",
      modelId: QWEN,
    });
    const requests = serve(LM_STUDIO, LM_STUDIO_MODELS);

    const group = (await core.listChatModels()).find((each) => each.provider.id === saved.id);
    expect(group?.models).toEqual([QWEN, GEMMA]);
    expect(group?.unknown).toEqual([]);
    const listing = requests.filter((request) => request.method === "GET");
    expect(listing.map((request) => request.url)).toEqual(["http://127.0.0.1:1234/api/v1/models"]);
    expect(listing[0]?.headers.authorization).toBeUndefined();

    // Loaded: its window is the one it is loaded with, and it calls Tools.
    const loaded = await core.prepareChatModel({ providerId: saved.id, modelId: QWEN });
    expect(loaded.capabilities.facts).toMatchObject({ tools: true, context: 8_192 });
    expect(loaded.window?.tokens).toBe(8_192);
    expect(loaded.support).toBe("tools");
    // Not loaded: the longest window it supports; it reads images but isn't trained for Tools.
    const unloaded = await core.prepareChatModel({ providerId: saved.id, modelId: GEMMA });
    expect(unloaded.capabilities.facts).toMatchObject({
      tools: false,
      context: 131_072,
      input: ["text", "image"],
    });
    expect(unloaded.support).toBe("structured-output");
  });

  test("a Question is answered at its endpoint, with no key, and Local mode allows it", async () => {
    const { core, client, mind } = await setUp([{ name: "Tides.pdf", contents: TIDES }]);
    await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "lmstudio",
      modelId: QWEN,
    });
    await core.setLocalOnly(true);
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "not-required" });

    const requests = serve(LM_STUDIO, LM_STUDIO_MODELS);
    const { answerId } = await askAndFinish(core, client, mind.id, "When are spring tides?");

    expect(citationsIn(client, answerId)[0]).toMatchObject({ pageFrom: 1, check: "found" });
    expect(posts(requests).length).toBeGreaterThan(0);
    for (const request of requests) expect(request.headers.authorization).toBeUndefined();
    for (const request of posts(requests)) {
      expect(request.url).toBe("http://127.0.0.1:1234/v1/chat/completions");
    }
  });

  test("an LM Studio that wants a token is given the one saved for it", async () => {
    const { core } = await setUp();
    const saved = await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "lmstudio",
      apiKey: "lm-token",
      modelId: QWEN,
    });
    expect(saved).toMatchObject({ hasApiKey: true, service: null });
    const requests = serve(LM_STUDIO, LM_STUDIO_MODELS);

    await core.listChatModels();

    expect(requests[0]?.headers.authorization).toBe("Bearer lm-token");
  });
});

describe("Unsloth Studio", () => {
  test("needs a key, which goes with every request, and is on this computer", async () => {
    const { core, client, mind } = await setUp([{ name: "Tides.pdf", contents: TIDES }]);
    await expect(
      core.saveChatProvider({ kind: "openai-compatible", catalogId: "unsloth", modelId: "qwen" }),
    ).rejects.toThrow(InvalidInputError);

    const saved = await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "unsloth",
      apiKey: "sk-unsloth-test",
      modelId: "qwen",
    });
    expect(saved).toMatchObject({
      catalogId: "unsloth",
      endpoint: "local",
      baseUrl: "http://127.0.0.1:8888/v1",
      hasApiKey: true,
      service: null,
    });
    await core.setLocalOnly(true);
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "not-required" });

    const requests = serve(UNSLOTH, { data: [{ id: "qwen" }] });
    const group = (await core.listChatModels()).find((each) => each.provider.id === saved.id);
    expect(group?.models).toContain("qwen");
    const { answerId } = await askAndFinish(core, client, mind.id, "When are spring tides?");

    expect(citationsIn(client, answerId)[0]).toMatchObject({ pageFrom: 1, check: "found" });
    expect(requests.find((request) => request.method === "GET")?.url).toBe(
      "http://127.0.0.1:8888/v1/models",
    );
    for (const request of requests) {
      expect(request.headers.authorization).toBe("Bearer sk-unsloth-test");
    }
    for (const request of posts(requests)) {
      expect(request.url).toBe("http://127.0.0.1:8888/v1/chat/completions");
    }
  });
});

describe("Local mode", () => {
  const LAN = { id: "http://192.168.1.20:1234", name: "LM Studio" };

  test("allows a preset set up on loopback and refuses one set up at a network address", async () => {
    const { core } = await setUp();
    const lan = await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "lmstudio",
      endpointValue: "192.168.1.20:1234",
      modelId: QWEN,
    });
    expect(lan.service).toEqual(LAN);
    await core.setLocalOnly(true);
    expect(await core.getChatReadiness()).toMatchObject({ ready: false, reason: "local-mode" });

    await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "lmstudio",
      modelId: QWEN,
    });
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "not-required" });

    await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "groq",
      apiKey: "key-groq",
      modelId: "openai/gpt-oss-120b",
    });
    expect(await core.getChatReadiness()).toMatchObject({ ready: false, reason: "local-mode" });
  });

  test("a preset on loopback asks for no consent to send anywhere, one at a network address does", async () => {
    const { core } = await setUp();
    await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "lmstudio",
      modelId: QWEN,
    });
    await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "lmstudio",
      endpointValue: "192.168.1.20:1234",
      modelId: QWEN,
    });

    const services = (await core.listDataFlows()).map((each) => each.flow.service.id);
    expect([...new Set(services)]).toEqual([LAN.id]);
  });
});

describe("Looking for the server", () => {
  const detecting = () =>
    setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
      localServers: createLocalServers(),
    });

  test("LM Studio running is found with its language models, asked the way Ollama is asked", async () => {
    const { core } = await detecting();
    const requests = serve(LM_STUDIO, LM_STUDIO_MODELS);

    const status = await core.detectLocalServer({ catalogId: "lmstudio" });

    expect(status).toEqual({ running: true, address: LM_STUDIO, models: [GEMMA, QWEN] });
    expect(requests.map((request) => request.url)).toEqual([
      "http://127.0.0.1:1234/v1/models",
      "http://127.0.0.1:1234/api/v1/models",
    ]);
    expect(posts(requests)).toEqual([]);
  });

  test("nothing answering is said once, with no retry and nothing sent", async () => {
    const { core } = await detecting();
    const refused = vi.fn(async () => {
      throw new TypeError("fetch failed");
    });
    vi.stubGlobal("fetch", refused);

    expect(await core.detectLocalServer({ catalogId: "lmstudio" })).toEqual({
      running: false,
      address: LM_STUDIO,
    });
    expect(refused).toHaveBeenCalledTimes(1);
  });

  test("a server that answers 401 is running; the key typed is sent, and then it lists its models", async () => {
    const { core } = await detecting();
    const asked: (string | null)[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: unknown, init?: RequestInit) => {
        const authorization = new Headers(init?.headers).get("authorization");
        asked.push(authorization);
        return authorization === "Bearer sk-unsloth-test"
          ? Response.json({ data: [{ id: "qwen" }] })
          : new Response("{}", { status: 401 });
      }),
    );

    expect(await core.detectLocalServer({ catalogId: "unsloth" })).toEqual({
      running: true,
      address: UNSLOTH,
      models: [],
    });
    expect(
      await core.detectLocalServer({ catalogId: "unsloth", apiKey: "sk-unsloth-test" }),
    ).toEqual({ running: true, address: UNSLOTH, models: ["qwen"] });
    expect(asked).toEqual([null, "Bearer sk-unsloth-test", "Bearer sk-unsloth-test"]);
  });

  test("the address typed is the one asked, and only a host and port is taken", async () => {
    const { core } = await detecting();
    const requests = serve("127.0.0.1:1235", LM_STUDIO_MODELS);

    expect(
      await core.detectLocalServer({ catalogId: "lmstudio", address: "127.0.0.1:1235" }),
    ).toEqual({ running: true, address: "127.0.0.1:1235", models: [GEMMA, QWEN] });
    expect(requests[0]?.url).toBe("http://127.0.0.1:1235/v1/models");
    await expect(
      core.detectLocalServer({ catalogId: "lmstudio", address: "evil.example.com/x" }),
    ).rejects.toThrow(InvalidInputError);
  });

  test("only a preset on this computer can be looked for", async () => {
    const { core } = await detecting();
    const fetched = vi.fn();
    vi.stubGlobal("fetch", fetched);

    for (const catalogId of ["groq", "ollama", "openai-compatible", "nothing"]) {
      await expect(core.detectLocalServer({ catalogId })).rejects.toThrow(InvalidInputError);
    }
    expect(fetched).not.toHaveBeenCalled();
  });
});
