import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, test } from "vitest";
import { type ConsentRequest, type Core, createOllamaModels } from "../../src/core";
import { en } from "../../src/shared/i18n/en";
import { zhCN } from "../../src/shared/i18n/zh-CN";
import { createTempDataFolder, startCore } from "../helpers/core";
import { replyingModel, scriptedModels } from "../helpers/models";

/**
 * Ollama can run cloud models through the local server: the request goes to
 * localhost and Ollama forwards it to ollama.com. These tests use a fake
 * Ollama whose `/api/tags` lists one model of each kind.
 */
const CLOUD_SERVICE = { id: "https://ollama.com", name: "ollama.com" };

const LOCAL_MODEL = "qwen3:4b";
/** Listed with `remote_host`, and named like a local model: only Ollama's word marks it. */
const REMOTE_BY_HOST = "glm-next:latest";
const REMOTE_BY_NAME = "gpt-oss:120b-cloud";
/** Listed as remote, but Ollama then fails to describe it. */
const REMOTE_NO_SHOW = "kimi-next:latest";

const servers: Server[] = [];
afterEach(() => {
  for (const server of servers.splice(0)) server.close();
});

async function fakeOllama(): Promise<string> {
  const server = createServer((request, response) => {
    response.setHeader("content-type", "application/json");
    if (request.url === "/api/tags") {
      response.end(
        JSON.stringify({
          models: [
            { name: LOCAL_MODEL, model: LOCAL_MODEL, digest: "d1", size: 2_500_000_000 },
            {
              name: REMOTE_BY_HOST,
              model: REMOTE_BY_HOST,
              digest: "d2",
              size: 400,
              remote_model: "glm-next",
              remote_host: "https://ollama.com:443",
            },
            { name: REMOTE_BY_NAME, model: REMOTE_BY_NAME, digest: "d3", size: 400 },
            {
              name: REMOTE_NO_SHOW,
              model: REMOTE_NO_SHOW,
              digest: "d4",
              size: 400,
              remote_host: "https://ollama.com:443",
            },
          ],
        }),
      );
    } else if (request.url === "/api/show") {
      let body = "";
      request.on("data", (chunk) => {
        body += String(chunk);
      });
      request.on("end", () => {
        if (body.includes(REMOTE_NO_SHOW)) {
          response.statusCode = 500;
          response.end("{}");
        } else response.end(JSON.stringify({ capabilities: ["completion", "tools"] }));
      });
    } else {
      response.end(JSON.stringify({ capabilities: ["completion", "tools"] }));
    }
  });
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

async function setUp() {
  const baseUrl = await fakeOllama();
  const models = scriptedModels(replyingModel());
  const core = startCore(await createTempDataFolder(), {
    createChatModel: models.createChatModel,
    ollamaModels: createOllamaModels(),
  });
  const requests: ConsentRequest[] = [];
  core.on("consent.requested", (request) => requests.push(request));
  const use = async (modelId: string) => {
    const provider = await core.saveChatProvider({ kind: "ollama", baseUrl, modelId });
    return { providerId: provider.id, modelId };
  };
  return { core, models, requests, baseUrl, use };
}

const answerConsent = async (core: Core, request: ConsentRequest, accept: boolean) =>
  core.respondToConsent(request.requestId, accept);

describe("Ollama cloud models", () => {
  test("a model that runs on this computer needs no consent", async () => {
    const { core, requests, use } = await setUp();
    const choice = await use(LOCAL_MODEL);

    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "not-required" });
    const prepared = await core.prepareChatModel(choice);

    expect(prepared.provider.service).toBeNull();
    expect(requests).toEqual([]);
    expect((await core.listChatProviders())[0]?.service).toBeNull();
    expect(await core.listDataFlows()).toEqual([]);
  });

  test.each([
    ["remote_host in /api/tags", REMOTE_BY_HOST],
    ["a name ending in -cloud", REMOTE_BY_NAME],
    ["remote_host in /api/tags when /api/show fails", REMOTE_NO_SHOW],
  ])("a model with %s asks for consent to send to ollama.com first", async (_how, modelId) => {
    const { core, models, requests, use } = await setUp();
    const choice = await use(modelId);

    expect(await core.getChatReadiness()).toMatchObject({
      ready: true,
      consent: "needed",
      provider: { service: CLOUD_SERVICE },
    });

    const prepared = core.prepareChatModel(choice);
    await expect.poll(() => requests.length).toBe(1);
    expect(requests[0]?.flow).toMatchObject({ id: "chat", service: CLOUD_SERVICE });
    // Nothing is made, let alone sent, before the answer.
    expect(models.specs).toEqual([]);
    await answerConsent(core, requests[0] as ConsentRequest, true);
    expect((await prepared).provider.service).toEqual(CLOUD_SERVICE);
    expect(models.specs).toHaveLength(1);
  });

  test("declining sends nothing and leaves the model unused", async () => {
    const { core, models, requests, use } = await setUp();
    const choice = await use(REMOTE_BY_HOST);

    const prepared = core.prepareChatModel(choice);
    const declined = expect(prepared).rejects.toThrow(/declined sending data to ollama\.com/);
    await expect.poll(() => requests.length).toBe(1);
    await answerConsent(core, requests[0] as ConsentRequest, false);
    await declined;

    expect(models.specs).toEqual([]);
    expect(await core.getChatReadiness()).toMatchObject({
      ready: false,
      reason: "consent-declined",
    });
  });

  test("Settings → Privacy lists ollama.com while a cloud model is the default", async () => {
    const { core, use } = await setUp();
    await use(REMOTE_BY_HOST);

    const chat = (await core.listRegisteredDataFlows()).find((flow) => flow.id === "chat");
    expect(chat?.services.map((each) => each.flow.service)).toEqual([CLOUD_SERVICE]);
    expect(chat?.services[0]?.consent).toBe("not-asked");
  });

  test("local mode refuses it with the usual explanation", async () => {
    const { core, requests, use } = await setUp();
    await use(REMOTE_BY_HOST);
    await core.setLocalOnly(true);
    await core.updateSettings({ user: { suggestWhileWriting: true } });

    expect((await core.getSuggestionStatus()).paused).toEqual({ reason: "local-mode" });
    expect(requests).toEqual([]);
  });

  test("the model list marks the cloud models, and a local one isn't marked", async () => {
    const { core, use } = await setUp();
    await use(LOCAL_MODEL);

    const [group] = await core.listChatModels();
    expect(group?.models).toEqual(
      expect.arrayContaining([LOCAL_MODEL, REMOTE_BY_HOST, REMOTE_BY_NAME, REMOTE_NO_SHOW]),
    );
    expect([...(group?.cloud ?? [])].sort()).toEqual(
      [REMOTE_BY_HOST, REMOTE_BY_NAME, REMOTE_NO_SHOW].sort(),
    );
  });

  test("the name decides when Ollama can't be asked", async () => {
    const { core, requests, use } = await setUp();
    const provider = await core.saveChatProvider({
      kind: "ollama",
      baseUrl: "http://127.0.0.1:1",
      modelId: REMOTE_BY_NAME,
    });
    expect(await core.getChatReadiness()).toMatchObject({
      ready: true,
      consent: "needed",
      provider: { id: provider.id, service: CLOUD_SERVICE },
    });
    expect(requests).toEqual([]);
    await use(LOCAL_MODEL);
  });

  test("the one-click local suggestions never offer a cloud model", async () => {
    const { core, baseUrl } = await setUp();

    const status = await core.detectOllama({ baseUrl });

    expect(status).toMatchObject({ running: true, models: [LOCAL_MODEL] });
    expect(status.running && [...status.cloudModels].sort()).toEqual(
      [REMOTE_BY_HOST, REMOTE_BY_NAME, REMOTE_NO_SHOW].sort(),
    );
    await expect(core.selectOllama({ baseUrl, model: REMOTE_BY_NAME })).rejects.toThrow(
      /runs on ollama\.com/,
    );
    await expect(core.selectOllama({ baseUrl, model: REMOTE_BY_HOST })).rejects.toThrow(
      /runs on ollama\.com/,
    );
  });
});

describe("Ollama cloud labels", () => {
  const keys = [
    "composer.model.group.cloud",
    "composer.model.hint.cloud",
    "composer.model.via.cloud",
    "providers.settings.cloud",
  ] as const;

  test.each(keys)("%s is written in English and in Chinese", (key) => {
    expect(en[key]).toMatch(/cloud/i);
    expect(en[key]).toMatch(/ollama/i);
    expect(zhCN[key]).toMatch(/云端/);
  });

  test("the catalog no longer says every Ollama model keeps Questions on the computer", async () => {
    const { catalogProvider } = await import("../../src/core/providers/catalog/providers");
    const summary = catalogProvider("ollama")?.dataUse.summary;
    expect(summary?.en).toMatch(/ollama\.com/);
    expect(summary?.zh).toMatch(/ollama\.com/);
  });
});
