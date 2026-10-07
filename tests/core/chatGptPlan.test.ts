import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { generateText, streamText } from "ai";
import { describe, expect, test } from "vitest";
import {
  ChatNotReadyError,
  type Core,
  type CoreAdapters,
  type CoreEventName,
  type CoreEvents,
  createAiSdkChatModel,
  InvalidInputError,
} from "../../src/core";
import { createFileKeychain, SECRETS_FILE } from "../../src/main/secretsFile";
import {
  ACCOUNT_ID,
  ANSWER,
  createFakeBrowser,
  EMAIL,
  type FakeBrowser,
  type FakeOpenAI,
  occupiedPort,
  startFakeOpenAI,
} from "../helpers/chatgpt";
import { createFakeCipher } from "../helpers/cipher";
import {
  createMemoryKeychain,
  createTempDataFolder,
  type MemoryKeychain,
  manualClock,
  nextEvent,
  startCore,
} from "../helpers/core";

const CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const CHATGPT_SERVICE = { id: "https://chatgpt.com", name: "ChatGPT" };
const MODEL = "gpt-5.5";

interface Setup {
  core: Core;
  fake: FakeOpenAI;
  browser: FakeBrowser;
  keychain: MemoryKeychain;
  clock: ReturnType<typeof manualClock>;
  dataDir: string;
}

/** A core whose ChatGPT plan provider talks to a fresh fake OpenAI through the real adapter. */
async function setUp(overrides: Partial<CoreAdapters> = {}): Promise<Setup> {
  const fake = await startFakeOpenAI();
  const browser = createFakeBrowser(fake);
  const keychain = createMemoryKeychain();
  const clock = manualClock();
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir, {
    keychain,
    browser,
    now: clock.now,
    createChatModel: createAiSdkChatModel,
    chatGptPlan: fake.endpoints,
    ...overrides,
  });
  return { core, fake, browser, keychain, clock, dataDir };
}

function acceptConsentAutomatically(core: Core): void {
  core.on("consent.requested", (request) => void core.respondToConsent(request.requestId, true));
}

/** Turns the provider on, signs in, accepts the chat flow and makes the plan the default chat model. */
async function signedIn(overrides: Partial<CoreAdapters> = {}): Promise<Setup> {
  const setup = await setUp(overrides);
  await setup.core.setChatGptPlanEnabled(true);
  const result = await setup.core.signInToChatGpt();
  expect(result.ok).toBe(true);
  acceptConsentAutomatically(setup.core);
  await setup.core.saveChatProvider({ kind: "chatgpt", modelId: MODEL });
  return setup;
}

/** Records every payload of an event from now on. */
function collect<E extends CoreEventName>(core: Core, event: E): CoreEvents[E][] {
  const payloads: CoreEvents[E][] = [];
  core.on(event, (payload) => payloads.push(payload));
  return payloads;
}

const refreshes = (fake: FakeOpenAI) =>
  fake.tokenRequests.filter((request) => request.grant_type === "refresh_token");

/** Asks one Question-sized request through the model Answers get. */
async function ask(core: Core) {
  const { model } = await core.prepareChatModel();
  return generateText({ model, prompt: "What is attention?", maxRetries: 0 });
}

const fiveMinutes = 5 * 60_000;

describe("The experimental switch", () => {
  test("is off by default, and while it is off nothing can be signed in, saved or tested", async () => {
    const { core, browser } = await setUp();

    const status = await core.getChatGptPlan();

    expect(status).toEqual({
      enabled: false,
      account: { state: "signed-out" },
      signingIn: false,
      models: expect.arrayContaining([{ id: MODEL, name: "GPT-5.5" }]),
    });
    await expect(core.signInToChatGpt()).rejects.toThrow(InvalidInputError);
    await expect(core.saveChatProvider({ kind: "chatgpt", modelId: MODEL })).rejects.toThrow(
      InvalidInputError,
    );
    await expect(core.testChatConnection({ kind: "chatgpt", modelId: MODEL })).rejects.toThrow(
      InvalidInputError,
    );
    expect(browser.opened).toEqual([]);
  });

  test("the plan offers only the models its endpoint accepts", async () => {
    const { core } = await setUp();
    await core.setChatGptPlanEnabled(true);

    const { models } = await core.getChatGptPlan();

    expect(models.length).toBeGreaterThan(0);
    for (const model of models) expect(model.id).toMatch(/^gpt-/);
    await expect(core.saveChatProvider({ kind: "chatgpt", modelId: "gpt-4o" })).rejects.toThrow(
      InvalidInputError,
    );
    await expect(
      core.saveChatProvider({ kind: "chatgpt", apiKey: "sk-x", modelId: MODEL }),
    ).rejects.toThrow(InvalidInputError);
  });

  test("turning it off signs out and removes the provider", async () => {
    const { core, keychain } = await signedIn();
    const changes = collect(core, "chatGptPlan.changed");

    const status = await core.setChatGptPlanEnabled(false);

    expect(status).toMatchObject({ enabled: false, account: { state: "signed-out" } });
    expect(keychain.secrets.size).toBe(0);
    expect(await core.listChatProviders()).toEqual([]);
    expect((await core.getSettings()).user.chatModel).toBeNull();
    expect(await core.getChatReadiness()).toEqual({ ready: false, reason: "no-provider" });
    await expect.poll(() => changes.at(-1)).toMatchObject({ enabled: false });
  });
});

describe("Signing in", () => {
  test("goes through the browser and a loopback redirect, with PKCE and the Codex CLI's registration", async () => {
    const { core, fake, browser, keychain } = await setUp();
    await core.setChatGptPlanEnabled(true);
    const changes = collect(core, "chatGptPlan.changed");

    const result = await core.signInToChatGpt();

    expect(browser.opened).toHaveLength(1);
    const authorize = new URL(browser.opened[0] ?? "");
    expect(`${authorize.origin}${authorize.pathname}`).toBe(fake.endpoints.authorizeUrl);
    const params = Object.fromEntries(authorize.searchParams);
    expect(params).toEqual({
      response_type: "code",
      client_id: CODEX_CLIENT_ID,
      redirect_uri: expect.stringMatching(/^http:\/\/127\.0\.0\.1:\d+\/auth\/callback$/),
      scope: "openid profile email offline_access",
      code_challenge: expect.stringMatching(/^[\w-]{43}$/),
      code_challenge_method: "S256",
      state: expect.any(String),
      id_token_add_organizations: "true",
      codex_cli_simplified_flow: "true",
      originator: "incarnamind",
    });
    // The code went back with the verifier; the fake checks it against the challenge.
    expect(fake.tokenRequests).toEqual([
      {
        grant_type: "authorization_code",
        client_id: CODEX_CLIENT_ID,
        code: "code-1",
        redirect_uri: params.redirect_uri,
        code_verifier: expect.stringMatching(/^[\w-]{43}$/),
      },
    ]);

    // The browser tab says it can be closed.
    const visit = await browser.visits[0];
    expect(visit?.status).toBe(200);
    expect(visit?.html).toContain("You can close this tab");

    expect(result).toEqual({
      ok: true,
      status: {
        enabled: true,
        account: { state: "signed-in", email: EMAIL, plan: "plus" },
        signingIn: false,
        models: expect.any(Array),
      },
    });
    await expect.poll(() => changes.map((status) => status.signingIn)).toEqual([true, false]);
    expect(changes.at(-1)?.account.state).toBe("signed-in");

    // The tokens are in the secret store, and the loopback port is free again.
    const stored = [...keychain.secrets.values()].join("\n");
    const issued = fake.issued[0];
    for (const token of [issued?.accessToken, issued?.refreshToken, issued?.idToken]) {
      expect(stored).toContain(token);
    }
    await expect(fetch(params.redirect_uri ?? "")).rejects.toThrow();
  });

  test("declining in the browser is reported, and nothing is stored", async () => {
    const { core, browser, keychain } = await setUp();
    await core.setChatGptPlanEnabled(true);
    browser.behaviour = "decline";

    const result = await core.signInToChatGpt();

    expect(result).toMatchObject({ ok: false, error: { kind: "denied" } });
    expect((await browser.visits[0])?.html).toContain("Close this tab and try again");
    expect(keychain.secrets.size).toBe(0);
    expect((await core.getChatGptPlan()).account).toEqual({ state: "signed-out" });
  });

  test("a busy port gives a plain-language error, and no browser opens", async () => {
    const port = await occupiedPort();
    const fake = await startFakeOpenAI();
    const { core, browser } = await setUp({
      chatGptPlan: { ...fake.endpoints, callbackPort: port },
    });
    await core.setChatGptPlanEnabled(true);

    const result = await core.signInToChatGpt();

    expect(result).toEqual({
      ok: false,
      error: { kind: "port-in-use", message: expect.stringContaining(`Port ${port}`) },
    });
    expect(browser.opened).toEqual([]);
    expect((await core.getChatGptPlan()).signingIn).toBe(false);
  });

  test("a sign-in nobody finishes times out", async () => {
    const fake = await startFakeOpenAI();
    const { core, browser } = await setUp({
      chatGptPlan: { ...fake.endpoints, signInTimeoutMs: 100 },
    });
    await core.setChatGptPlanEnabled(true);
    browser.behaviour = "ignore";

    const result = await core.signInToChatGpt();

    expect(result).toMatchObject({ ok: false, error: { kind: "timed-out" } });
    expect((await core.getChatGptPlan()).signingIn).toBe(false);
  });

  test("a waiting sign-in can be cancelled, which frees its port", async () => {
    const { core, browser } = await setUp();
    await core.setChatGptPlanEnabled(true);
    browser.behaviour = "ignore";

    const result = core.signInToChatGpt();
    await expect.poll(() => browser.opened).toHaveLength(1);
    expect((await core.getChatGptPlan()).signingIn).toBe(true);
    await core.cancelChatGptSignIn();

    expect(await result).toMatchObject({ ok: false, error: { kind: "cancelled" } });
    expect((await core.getChatGptPlan()).signingIn).toBe(false);
    const redirect = new URL(browser.opened[0] ?? "").searchParams.get("redirect_uri") ?? "";
    await expect(fetch(redirect)).rejects.toThrow();
  });

  test("starting again replaces a sign-in still waiting, e.g. after the User closed the tab", async () => {
    const { core, browser } = await setUp();
    await core.setChatGptPlanEnabled(true);
    browser.behaviour = "ignore";
    const abandoned = core.signInToChatGpt();
    await expect.poll(() => browser.opened).toHaveLength(1);

    browser.behaviour = "approve";
    const retried = await core.signInToChatGpt();

    expect(await abandoned).toMatchObject({ ok: false, error: { kind: "cancelled" } });
    expect(retried).toMatchObject({ ok: true, status: { account: { state: "signed-in" } } });
    expect(browser.opened).toHaveLength(2);
  });
});

describe("Answers with the ChatGPT plan", () => {
  test("use the plan through the model factory, with the endpoint's headers and request shape", async () => {
    const { core, fake } = await signedIn();
    expect(await core.getChatReadiness()).toMatchObject({
      ready: true,
      provider: { kind: "chatgpt", service: CHATGPT_SERVICE },
      modelId: MODEL,
    });

    const { model } = await core.prepareChatModel();
    const generated = await generateText({
      model,
      system: "Answer briefly.",
      prompt: "What is attention?",
      maxOutputTokens: 200,
      maxRetries: 0,
    });
    const streamed = streamText({ model, prompt: "And transformers?", maxRetries: 0 });
    let streamedText = "";
    for await (const delta of streamed.textStream) streamedText += delta;

    expect(generated.text).toBe(ANSWER);
    expect(streamedText).toBe(ANSWER);
    expect(fake.codexRequests).toHaveLength(2);
    const [first, second] = fake.codexRequests;
    expect(first?.headers).toMatchObject({
      authorization: `Bearer ${fake.issued[0]?.accessToken}`,
      "chatgpt-account-id": ACCOUNT_ID,
      "openai-beta": "responses=experimental",
      originator: "incarnamind",
      accept: "text/event-stream",
    });
    expect(first?.body).toMatchObject({
      model: MODEL,
      instructions: "Answer briefly.",
      store: false,
      stream: true,
      include: ["reasoning.encrypted_content"],
      input: [{ role: "user", content: [{ type: "input_text", text: "What is attention?" }] }],
    });
    expect(first?.body).not.toHaveProperty("max_output_tokens");
    // Without a system prompt, the endpoint still gets instructions.
    expect(second?.body).toMatchObject({ instructions: expect.any(String), stream: true });
    expect(second?.body.instructions).not.toBe("");
  });

  test("the connection test sends one small request and reports success", async () => {
    const { core, fake } = await signedIn();

    expect(await core.testChatConnection({ kind: "chatgpt", modelId: MODEL })).toEqual({
      ok: true,
    });
    expect(fake.codexRequests).toHaveLength(1);
  });
});

describe("Tokens", () => {
  test("an access token is refreshed shortly before it expires, and the new tokens are stored", async () => {
    const { core, fake, clock, keychain } = await signedIn();
    const first = fake.issued[0];

    await ask(core);
    clock.advance(fake.expiresIn * 1000 - fiveMinutes + 1000);
    await ask(core);
    await ask(core);

    expect(refreshes(fake)).toEqual([
      {
        grant_type: "refresh_token",
        client_id: CODEX_CLIENT_ID,
        refresh_token: first?.refreshToken,
      },
    ]);
    const second = fake.issued[1];
    expect(fake.codexRequests.map((request) => request.headers.authorization)).toEqual([
      `Bearer ${first?.accessToken}`,
      `Bearer ${second?.accessToken}`,
      `Bearer ${second?.accessToken}`,
    ]);
    const stored = [...keychain.secrets.values()].join("\n");
    expect(stored).toContain(second?.refreshToken);
    expect(stored).not.toContain(first?.refreshToken);
  });

  test("concurrent requests share one refresh", async () => {
    const { core, fake, clock } = await signedIn();
    fake.refreshDelayMs = 50;
    clock.advance(fake.expiresIn * 1000);

    await Promise.all([ask(core), ask(core), ask(core)]);

    expect(refreshes(fake)).toHaveLength(1);
    const fresh = `Bearer ${fake.issued[1]?.accessToken}`;
    expect(fake.codexRequests.map((request) => request.headers.authorization)).toEqual([
      fresh,
      fresh,
      fresh,
    ]);
  });

  test("a token the endpoint refuses early is refreshed once and the request retried", async () => {
    const { core, fake } = await signedIn();
    fake.revokeAccessTokens();

    const answer = await ask(core);

    expect(answer.text).toBe(ANSWER);
    expect(refreshes(fake)).toHaveLength(1);
    expect(fake.codexRequests).toHaveLength(2);
  });

  test("a refused refresh asks the User to sign in again, and Questions wait for it", async () => {
    const { core, fake, clock, keychain } = await signedIn();
    fake.refreshRefusal = {
      status: 400,
      body: { error: "invalid_grant", error_description: "refresh_token_expired" },
    };
    clock.advance(fake.expiresIn * 1000);
    const readinessChanged = collect(core, "chatReadiness.changed");
    const planChanged = collect(core, "chatGptPlan.changed");

    const result = await core.testChatConnection({ kind: "chatgpt", modelId: MODEL });

    expect(result).toMatchObject({ ok: false, error: { kind: "not-signed-in" } });
    expect(fake.codexRequests).toEqual([]);
    expect(keychain.secrets.size).toBe(0);
    expect((await core.getChatGptPlan()).account).toEqual({ state: "expired" });
    await expect.poll(() => planChanged.at(-1)?.account).toEqual({ state: "expired" });
    const readiness = await core.getChatReadiness();
    expect(readiness).toMatchObject({ ready: false, reason: "sign-in-required" });
    await expect.poll(() => readinessChanged.at(-1)).toEqual(readiness);
    await expect(core.prepareChatModel()).rejects.toThrow(ChatNotReadyError);

    // Signing in again brings Questions back.
    fake.refreshRefusal = null;
    expect((await core.signInToChatGpt()).ok).toBe(true);
    expect(await core.getChatReadiness()).toMatchObject({ ready: true });
  });

  test("signing out deletes the tokens, and nothing is sent afterwards", async () => {
    const { core, fake, keychain } = await signedIn();

    const status = await core.signOutOfChatGpt();

    expect(status.account).toEqual({ state: "signed-out" });
    expect(keychain.secrets.size).toBe(0);
    expect(await core.getChatReadiness()).toMatchObject({
      ready: false,
      reason: "sign-in-required",
    });
    expect(await core.testChatConnection({ kind: "chatgpt", modelId: MODEL })).toEqual({
      ok: false,
      error: { kind: "not-signed-in", message: expect.any(String) },
    });
    expect(fake.codexRequests).toEqual([]);
  });

  test("tokens never reach the SQLite database, only the encrypted secrets file", async () => {
    const dataDir = await createTempDataFolder();
    const keychain = createFileKeychain(join(dataDir, SECRETS_FILE), createFakeCipher());
    const { core, fake, clock } = await signedIn({ keychain, paths: { dataDir } });
    clock.advance(fake.expiresIn * 1000);
    await ask(core); // refreshes, so two sets of tokens have been stored
    core.close();

    const tokens = fake.issued.flatMap((set) => [set.accessToken, set.refreshToken, set.idToken]);
    expect(tokens).toHaveLength(6);
    const files = (await readdir(dataDir, { withFileTypes: true, recursive: true }))
      .filter((entry) => entry.isFile())
      .map((entry) => join(entry.parentPath, entry.name));
    expect(files).toContain(join(dataDir, "incarnamind.db"));
    expect(files).toContain(join(dataDir, SECRETS_FILE));
    for (const file of files) {
      const bytes = await readFile(file);
      for (const token of tokens) {
        expect(bytes.includes(token), `${file} contains a token`).toBe(false);
      }
    }
    // The tokens are there, encrypted: a restart is still signed in.
    const restarted = startCore(dataDir, {
      keychain: createFileKeychain(join(dataDir, SECRETS_FILE), createFakeCipher()),
    });
    expect((await restarted.getChatGptPlan()).account).toMatchObject({ state: "signed-in" });
  });
});

describe("Consent", () => {
  test("nothing goes to chatgpt.com before the User accepts the chat flow", async () => {
    const { core, fake } = await setUp();
    await core.setChatGptPlanEnabled(true);
    await core.signInToChatGpt();
    await core.saveChatProvider({ kind: "chatgpt", modelId: MODEL });
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "needed" });

    const requested = nextEvent(core, "consent.requested");
    const result = core.testChatConnection({ kind: "chatgpt", modelId: MODEL });
    const request = await requested;

    expect(request.flow).toEqual({
      id: "chat",
      service: CHATGPT_SERVICE,
      sends: ["blocks", "passages"],
    });
    expect(fake.codexRequests).toEqual([]);

    await core.respondToConsent(request.requestId, false);

    expect(await result).toMatchObject({ ok: false, error: { kind: "consent-declined" } });
    expect(fake.codexRequests).toEqual([]);
    expect(await core.getChatReadiness()).toMatchObject({
      ready: false,
      reason: "consent-declined",
    });
    await expect(core.prepareChatModel()).rejects.toThrow(ChatNotReadyError);
    expect(fake.codexRequests).toEqual([]);
  });
});

describe("Connection test errors", () => {
  test.each([
    {
      name: "a reached usage limit",
      status: 429,
      body: { error: { type: "usage_limit_reached", plan_type: "plus", resets_at: 1_790_003_600 } },
      kind: "plan-limit",
    },
    {
      name: "a plan without this use",
      status: 429,
      body: { error: { type: "usage_not_included", message: "Upgrade to use Codex." } },
      kind: "plan-limit",
    },
    {
      name: "a 403 after a valid sign-in",
      status: 403,
      body: { error: { message: "This client isn't allowed." } },
      kind: "blocked",
    },
    {
      name: "an unknown model",
      status: 404,
      body: { error: { message: "Model not found", type: "invalid_request_error" } },
      kind: "model",
    },
    {
      name: "a server error",
      status: 500,
      body: { error: { message: "Internal error", type: "server_error" } },
      kind: "provider",
    },
  ])("$name is reported as $kind", async ({ status, body, kind }) => {
    const { core, fake } = await signedIn();
    fake.codexReply = { kind: "error", status, body };

    const result = await core.testChatConnection({ kind: "chatgpt", modelId: MODEL });

    expect(result).toMatchObject({ ok: false, error: { kind } });
  });

  test("a 401 that a fresh token doesn't fix means OpenAI blocked the sign-in", async () => {
    const { core, fake } = await signedIn();
    // Every token is refused from now on, including the refreshed one.
    fake.codexReply = { kind: "error", status: 401, body: { error: { message: "Unauthorized" } } };

    const result = await core.testChatConnection({ kind: "chatgpt", modelId: MODEL });

    expect(result).toMatchObject({ ok: false, error: { kind: "blocked" } });
    expect(refreshes(fake)).toHaveLength(1);
    expect(fake.codexRequests.map((request) => request.headers.authorization)).toEqual([
      `Bearer ${fake.issued[0]?.accessToken}`,
      `Bearer ${fake.issued[1]?.accessToken}`,
    ]);
    // The sign-in itself is still good: only the endpoint refused it.
    expect((await core.getChatGptPlan()).account.state).toBe("signed-in");
  });

  test("the plan limit message says when it resets", async () => {
    const { core, fake } = await signedIn();
    fake.codexReply = {
      kind: "error",
      status: 429,
      body: { error: { type: "usage_limit_reached", plan_type: "plus", resets_at: 1_790_003_600 } },
    };

    const result = await core.testChatConnection({ kind: "chatgpt", modelId: MODEL });

    expect(result).toMatchObject({
      ok: false,
      error: {
        kind: "plan-limit",
        message: expect.stringContaining(new Date(1_790_003_600_000).toISOString()),
      },
    });
  });
});
