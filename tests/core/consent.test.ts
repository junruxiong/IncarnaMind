import { describe, expect, test, vi } from "vitest";
import {
  ChatNotReadyError,
  ConsentDeclinedError,
  type Core,
  type SaveChatProviderInput,
  type TestChatConnectionInput,
} from "../../src/core";
import { createMemoryKeychain, createTempDataFolder, nextEvent, startCore } from "../helpers/core";
import { replyingModel, scriptedModels } from "../helpers/models";

/** Works both for saving a provider and for testing it. */
type ProviderInput = SaveChatProviderInput & TestChatConnectionInput;

const OPENAI: ProviderInput = {
  kind: "openai",
  apiKey: "sk-test-openai",
  modelId: "gpt-5.4-mini",
};
const OPENAI_SERVICE = { id: "https://api.openai.com", name: "OpenAI" };

async function startWithModel(dataDir?: string, keychain = createMemoryKeychain()) {
  const models = scriptedModels(replyingModel());
  const core = startCore(dataDir ?? (await createTempDataFolder()), {
    keychain,
    createChatModel: models.createChatModel,
  });
  return { core, models };
}

/** Starts a connection test and returns it with the consent request it raised. */
async function testAndWaitForConsent(core: Core, input: ProviderInput = OPENAI) {
  const requested = nextEvent(core, "consent.requested");
  const result = core.testChatConnection(input);
  return { result, request: await requested };
}

describe("Data-flow consent", () => {
  test("nothing is sent until the User accepts, and the dialog says what goes where", async () => {
    const { core, models } = await startWithModel();
    await core.saveChatProvider(OPENAI);

    const { result, request } = await testAndWaitForConsent(core);

    expect(request).toEqual({
      requestId: expect.any(String),
      flow: { id: "chat", service: OPENAI_SERVICE, sends: ["blocks", "passages"] },
      newKinds: ["blocks", "passages"],
    });
    expect(await core.listConsentRequests()).toEqual([request]);
    expect(models.specs).toEqual([]);
    expect(models.model.doGenerateCalls).toEqual([]);

    const resolved = nextEvent(core, "consent.resolved");
    await core.respondToConsent(request.requestId, true);

    expect(await result).toEqual({ ok: true });
    expect(await resolved).toEqual({ requestId: request.requestId, accepted: true });
    expect(models.model.doGenerateCalls).toHaveLength(1);
    expect(await core.listConsentRequests()).toEqual([]);
  });

  test("an accepted flow isn't asked again, even after a restart", async () => {
    const dataDir = await createTempDataFolder();
    const keychain = createMemoryKeychain();
    const first = await startWithModel(dataDir, keychain);
    await first.core.saveChatProvider(OPENAI);
    const { result, request } = await testAndWaitForConsent(first.core);
    await first.core.respondToConsent(request.requestId, true);
    await result;
    first.core.close();

    const { core, models } = await startWithModel(dataDir, keychain);
    let asked = false;
    core.on("consent.requested", () => {
      asked = true;
    });

    expect(await core.testChatConnection(OPENAI)).toEqual({ ok: true });
    expect(asked).toBe(false);
    expect(models.model.doGenerateCalls).toHaveLength(1);
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "accepted" });
    expect(await core.listDataFlows()).toEqual([
      {
        flow: { id: "chat", service: OPENAI_SERVICE, sends: ["blocks", "passages"] },
        consent: "accepted",
        decidedAt: expect.any(String),
      },
    ]);
  });

  test("declining sends nothing and keeps Questions disabled until another provider is chosen", async () => {
    const { core, models } = await startWithModel();
    await core.saveChatProvider(OPENAI);
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "needed" });

    const { result, request } = await testAndWaitForConsent(core);
    await core.respondToConsent(request.requestId, false);

    expect(await result).toMatchObject({ ok: false, error: { kind: "consent-declined" } });
    expect(models.model.doGenerateCalls).toEqual([]);
    expect(await core.getChatReadiness()).toMatchObject({
      ready: false,
      reason: "consent-declined",
      provider: { kind: "openai" },
    });

    // The service stays unused: asking again doesn't raise another dialog or send anything.
    let askedAgain = false;
    core.on("consent.requested", () => {
      askedAgain = true;
    });
    expect(await core.testChatConnection(OPENAI)).toMatchObject({
      ok: false,
      error: { kind: "consent-declined" },
    });
    expect(askedAgain).toBe(false);
    expect(models.model.doGenerateCalls).toEqual([]);

    // A local model makes Questions available again.
    await core.saveChatProvider({
      kind: "ollama",
      baseUrl: "http://127.0.0.1:11434",
      modelId: "qwen3:4b",
    });
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "not-required" });

    // So does another cloud provider, which asks for its own consent.
    await core.saveChatProvider({ kind: "anthropic", apiKey: "k", modelId: "claude-sonnet-4-6" });
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "needed" });
  });

  test("consent is recorded per service", async () => {
    const { core } = await startWithModel();
    await core.saveChatProvider(OPENAI);
    const openai = await testAndWaitForConsent(core);
    await core.respondToConsent(openai.request.requestId, true);
    await openai.result;

    const anthropicInput: ProviderInput = {
      kind: "anthropic",
      apiKey: "sk-ant-test",
      modelId: "claude-sonnet-4-6",
    };
    await core.saveChatProvider(anthropicInput);
    const anthropic = await testAndWaitForConsent(core, anthropicInput);

    expect(anthropic.request.flow.service).toEqual({
      id: "https://api.anthropic.com",
      name: "Anthropic",
    });
    await core.respondToConsent(anthropic.request.requestId, false);
    await anthropic.result;
    expect(await core.listDataFlows()).toEqual([
      expect.objectContaining({
        flow: expect.objectContaining({ service: OPENAI_SERVICE }),
        consent: "accepted",
      }),
      expect.objectContaining({
        flow: expect.objectContaining({
          service: { id: "https://api.anthropic.com", name: "Anthropic" },
        }),
        consent: "declined",
      }),
    ]);
  });

  test("revoking forgets the decision: the next request asks again", async () => {
    const { core, models } = await startWithModel();
    await core.saveChatProvider(OPENAI);
    const first = await testAndWaitForConsent(core);
    await core.respondToConsent(first.request.requestId, true);
    await first.result;

    const readinessChanged = nextEvent(core, "chatReadiness.changed");
    await core.revokeConsent("chat", OPENAI_SERVICE.id);

    expect(await readinessChanged).toMatchObject({ ready: true, consent: "needed" });
    expect(await core.listDataFlows()).toEqual([
      expect.objectContaining({ consent: "not-asked", decidedAt: null }),
    ]);
    const second = await testAndWaitForConsent(core);
    expect(models.model.doGenerateCalls).toHaveLength(1);
    await core.respondToConsent(second.request.requestId, true);
    expect(await second.result).toEqual({ ok: true });
  });

  test("revoking a decline lets the User change their mind", async () => {
    const { core } = await startWithModel();
    await core.saveChatProvider(OPENAI);
    const declined = await testAndWaitForConsent(core);
    await core.respondToConsent(declined.request.requestId, false);
    await declined.result;

    await core.revokeConsent("chat", OPENAI_SERVICE.id);

    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "needed" });
  });

  test("a decision on a provider that was only tested is listed, so it can be revoked", async () => {
    const { core } = await startWithModel();
    const declined = await testAndWaitForConsent(core);
    await core.respondToConsent(declined.request.requestId, false);
    await declined.result;
    expect(await core.listChatProviders()).toEqual([]);

    expect(await core.listDataFlows()).toEqual([
      {
        flow: { id: "chat", service: OPENAI_SERVICE, sends: ["blocks", "passages"] },
        consent: "declined",
        decidedAt: expect.any(String),
      },
    ]);
    await core.revokeConsent("chat", OPENAI_SERVICE.id);
    expect(await core.listDataFlows()).toEqual([]);
  });

  test("a flow that starts sending a new kind of data asks again, for the new kind", async () => {
    const { core, models } = await startWithModel();
    await core.saveChatProvider(OPENAI);
    const first = await testAndWaitForConsent(core);
    await core.respondToConsent(first.request.requestId, true);
    await first.result;

    // E.g. once Connectors exist, Answers also send Tool results.
    const chat = core.dataFlows.get("chat");
    if (!chat) throw new Error("The chat flow isn't registered.");
    core.dataFlows.register({ ...chat, sends: [...chat.sends, "tool-results"] });

    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "needed" });
    const second = await testAndWaitForConsent(core);
    expect(second.request.flow.sends).toEqual(["blocks", "passages", "tool-results"]);
    expect(second.request.newKinds).toEqual(["tool-results"]);
    expect(models.model.doGenerateCalls).toHaveLength(1);

    await core.respondToConsent(second.request.requestId, true);
    expect(await second.result).toEqual({ ok: true });
    expect(await core.listDataFlows()).toEqual([expect.objectContaining({ consent: "accepted" })]);
  });

  test("requests made while a dialog is open wait for the same answer", async () => {
    const { core, models } = await startWithModel();
    await core.saveChatProvider(OPENAI);
    const requests: string[] = [];
    core.on("consent.requested", (request) => requests.push(request.requestId));

    const first = core.testChatConnection(OPENAI);
    const second = core.testChatConnection(OPENAI);
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    await core.respondToConsent(requests[0] as string, true);

    expect(await Promise.all([first, second])).toEqual([{ ok: true }, { ok: true }]);
    expect(requests).toHaveLength(1);
    expect(models.model.doGenerateCalls).toHaveLength(2);
  });

  test("answering a request twice, or one that doesn't exist, changes nothing", async () => {
    const { core } = await startWithModel();
    await core.saveChatProvider(OPENAI);
    const { result, request } = await testAndWaitForConsent(core);
    await core.respondToConsent(request.requestId, true);
    await result;

    await core.respondToConsent(request.requestId, false);
    await core.respondToConsent("no-such-request", false);

    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "accepted" });
  });

  test("Answers get a model only once the User has accepted its flow", async () => {
    const { core, models } = await startWithModel();
    await expect(core.prepareChatModel()).rejects.toThrow(ChatNotReadyError);
    await core.saveChatProvider(OPENAI);

    const requested = nextEvent(core, "consent.requested");
    const prepared = core.prepareChatModel();
    const request = await requested;
    expect(models.specs).toEqual([]);
    await core.respondToConsent(request.requestId, true);

    expect(await prepared).toMatchObject({ model: models.model, modelId: "gpt-5.4-mini" });
    expect(models.specs).toEqual([
      { kind: "openai", baseUrl: null, apiKey: "sk-test-openai", modelId: "gpt-5.4-mini" },
    ]);
  });

  test("Answers get no model after the User declined", async () => {
    const { core, models } = await startWithModel();
    await core.saveChatProvider(OPENAI);
    const requested = nextEvent(core, "consent.requested");
    const prepared = core.prepareChatModel();
    await core.respondToConsent((await requested).requestId, false);

    await expect(prepared).rejects.toThrow(ConsentDeclinedError);
    await expect(core.prepareChatModel()).rejects.toThrow(ChatNotReadyError);
    expect(models.specs).toEqual([]);
  });
});
