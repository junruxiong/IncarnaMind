import { generateText, streamText } from "ai";
import { describe, expect, test, vi } from "vitest";
import { version } from "../../package.json";
import { type CopilotProfileId, type Core, createAiSdkChatModel } from "../../src/core";
import type { SignInRouteProblem } from "../../src/core/api";
import { authorizeWithDevice } from "../../src/core/oauth";
import { COPILOT_SERVICE } from "../../src/core/providers/copilot/service";
import { classifyProviderError } from "../../src/core/providers/providerErrors";
import {
  COPILOT_SIGN_IN_PROBLEMS,
  SIGN_IN_ROUTES,
  wayBackLabel,
} from "../../src/core/providers/signInRoutes";
import { translate } from "../../src/shared/i18n";
import { askAndFinish, citationsIn, shownPassages } from "../helpers/citations";
import {
  createMemoryKeychain,
  createTempDataFolder,
  type MemoryKeychain,
  manualClock,
  startCore,
  type TestAdapters,
} from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";
import {
  ANSWER,
  createRecordingBrowser,
  EDITOR_HEADER,
  type FakeGitHub,
  startFakeGitHub,
  USER_CODE,
} from "../helpers/fakeGitHub";
import { connectToMind } from "../helpers/mindClient";

const MODEL = "gpt-5.5";

interface Setup {
  core: Core;
  fake: FakeGitHub;
  browser: ReturnType<typeof createRecordingBrowser>;
  keychain: MemoryKeychain;
  clock: ReturnType<typeof manualClock>;
}

async function setUp(overrides: TestAdapters = {}): Promise<Setup> {
  const clock = manualClock();
  const fake = await startFakeGitHub();
  const browser = createRecordingBrowser();
  const keychain = createMemoryKeychain();
  const core = startCore(await createTempDataFolder(), {
    keychain,
    browser,
    now: clock.now,
    createChatModel: createAiSdkChatModel,
    copilotProfiles: { own: fake.profile("own"), editor: fake.profile("editor") },
    ...overrides,
  });
  return { core, fake, browser, keychain, clock };
}

/**
 * Starts a sign-in and waits until it shows GitHub's code. The sign-in, still
 * waiting for the User, comes back in an object: awaiting it here would wait
 * for the User.
 */
async function codeShown(core: Core, profile?: CopilotProfileId) {
  const result = core.signInToCopilot(profile ? { profile } : {});
  await vi.waitFor(async () => expect((await core.getCopilot()).deviceCode).not.toBeNull());
  return { result };
}

async function signIn(setup: Setup, profile?: CopilotProfileId) {
  const { result } = await codeShown(setup.core, profile);
  setup.fake.approve();
  expect(await result).toMatchObject({ ok: true });
}

async function signedIn(profile?: CopilotProfileId, setup?: Setup): Promise<Setup> {
  const ready = setup ?? (await setUp());
  await signIn(ready, profile);
  await ready.core.saveChatProvider({ kind: "copilot", modelId: MODEL });
  return ready;
}

async function ask(core: Core) {
  const { model } = await core.prepareChatModel();
  return generateText({ model, prompt: "Hi", maxRetries: 0 });
}

async function failure(core: Core) {
  try {
    await ask(core);
  } catch (error) {
    return classifyProviderError(error);
  }
  throw new Error("The request didn't fail.");
}

const chats = (fake: FakeGitHub) => fake.requests.filter((r) => r.path === "/chat/completions");
const stored = (keychain: MemoryKeychain) => [...keychain.secrets.values()].join("\n");

describe("The device flow", () => {
  test("polls at the server's interval, and after a slow_down five seconds longer or as long as it says", async () => {
    const answers = [
      { error: "authorization_pending" },
      { error: "slow_down" },
      { error: "authorization_pending" },
      { error: "slow_down", interval: 20 },
      { access_token: "test-access" },
    ];
    const waits: number[] = [];
    const tokens = await authorizeWithDevice({
      deviceCodeUrl: "https://github.example.test/login/device/code",
      tokenUrl: "https://github.example.test/login/oauth/access_token",
      clientId: "test-app",
      onCode: () => {},
      openBrowser: async () => {},
      wait: async (ms) => {
        waits.push(ms);
      },
      fetch: async (input) =>
        Response.json(
          String(input).endsWith("/device/code")
            ? {
                device_code: "device",
                user_code: USER_CODE,
                verification_uri: "https://github.example.test/login/device",
                expires_in: 900,
                interval: 5,
              }
            : answers.shift(),
        ),
    });

    expect(tokens.accessToken).toBe("test-access");
    expect(waits).toEqual([5000, 5000, 10_000, 10_000, 20_000]);
  });

  test("gives up once the code's lifetime has been waited", async () => {
    const waits: number[] = [];
    const signIn = authorizeWithDevice({
      deviceCodeUrl: "https://github.example.test/login/device/code",
      tokenUrl: "https://github.example.test/login/oauth/access_token",
      clientId: "test-app",
      onCode: () => {},
      openBrowser: async () => {},
      wait: async (ms) => {
        waits.push(ms);
      },
      fetch: async (input) =>
        Response.json(
          String(input).endsWith("/device/code")
            ? {
                device_code: "device",
                user_code: USER_CODE,
                verification_uri: "https://github.example.test/login/device",
                expires_in: 12,
                interval: 5,
              }
            : { error: "authorization_pending" },
        ),
    });

    await expect(signIn).rejects.toMatchObject({ reason: "timed-out" });
    expect(waits).toEqual([5000, 5000, 5000]);
  });
  test("cancelling during a poll is a cancelled sign-in, not a failed one", async () => {
    const controller = new AbortController();
    const signIn = authorizeWithDevice({
      deviceCodeUrl: "https://github.example.test/login/device/code",
      tokenUrl: "https://github.example.test/login/oauth/access_token",
      clientId: "test-app",
      onCode: () => {},
      openBrowser: async () => {},
      wait: async () => {},
      signal: controller.signal,
      fetch: async (input, init) => {
        if (String(input).endsWith("/device/code")) {
          return Response.json({
            device_code: "device",
            user_code: USER_CODE,
            verification_uri: "https://github.example.test/login/device",
            interval: 5,
          });
        }
        controller.abort();
        throw init?.signal?.reason;
      },
    });

    await expect(signIn).rejects.toMatchObject({ reason: "cancelled" });
  });
});

describe("Sign in with GitHub for Copilot", () => {
  test("shows GitHub's code, opens its page, and signs in once the User approves there", async () => {
    const { core, fake, browser, keychain } = await setUp();
    const { result } = await codeShown(core);

    expect((await core.getCopilot()).deviceCode).toEqual({
      userCode: USER_CODE,
      verificationUri: `${fake.origin}/login/device`,
      expiresIn: 900,
    });
    expect(browser.opened).toEqual([`${fake.origin}/login/device`]);
    expect(fake.requests[0]?.body).toEqual({
      client_id: "test-own-app",
      scope: "read:user offline_access",
    });

    fake.approve();
    expect(await result).toMatchObject({
      ok: true,
      status: { account: { state: "signed-in", profile: "own" }, deviceCode: null },
    });
    const poll = fake.requests.find((request) => request.body.grant_type !== undefined);
    expect(poll?.body).toEqual({
      client_id: "test-own-app",
      device_code: "device-1",
      grant_type: "urn:ietf:params:oauth:grant-type:device_code",
    });
    expect(stored(keychain)).toContain(fake.issued[0]?.accessToken);
  });

  test("a slow_down from GitHub is waited out, and the sign-in still finishes", async () => {
    const { core, fake } = await setUp();
    fake.slowDown = true;
    const { result } = await codeShown(core);
    fake.approve();

    expect(await result).toMatchObject({ ok: true });
  });

  test("cancelling while the code shows keeps nothing", async () => {
    const { core, keychain } = await setUp();
    const { result } = await codeShown(core);

    await core.cancelCopilotSignIn();

    expect(await result).toMatchObject({ ok: false, error: { kind: "sign-in-cancelled" } });
    expect(keychain.secrets.size).toBe(0);
    expect(await core.getCopilot()).toMatchObject({
      account: { state: "signed-out" },
      deviceCode: null,
    });
  });

  test("declining on GitHub's page says the User declined, and a code that expires says so", async () => {
    const { core, fake } = await setUp();
    const { result: declined } = await codeShown(core);
    fake.deny();
    expect(await declined).toMatchObject({ ok: false, error: { kind: "declined" } });

    const { result: expired } = await codeShown(core);
    fake.expire();
    expect(await expired).toMatchObject({ ok: false, error: { kind: "code-expired" } });
  });

  test("offers IncarnaMind's own app first, and says which way to sign in is experimental", async () => {
    const { core } = await setUp();

    expect((await core.getCopilot()).profiles).toEqual([
      { id: "own", experimental: false },
      { id: "editor", experimental: true },
    ]);
  });
});

describe("Questions with IncarnaMind's own app", () => {
  test("go to Copilot with the GitHub token and IncarnaMind's User-Agent, and nothing of another app", async () => {
    const { core, fake } = await signedIn();

    const { text } = await ask(core);

    expect(text).toBe(ANSWER);
    const [request] = chats(fake);
    expect(request?.headers).toMatchObject({
      authorization: `Bearer ${fake.issued[0]?.accessToken}`,
      "user-agent": `IncarnaMind/${version}`,
    });
    expect(request?.headers["editor-version"]).toBeUndefined();
    expect(request?.headers["copilot-integration-id"]).toBeUndefined();
    expect(fake.requests.some((r) => r.path === "/copilot_internal/v2/token")).toBe(false);
  });

  test("a streamed Answer comes through", async () => {
    const { core } = await signedIn();
    const { model } = await core.prepareChatModel();

    expect(await streamText({ model, prompt: "Hi", maxRetries: 0 }).text).toBe(ANSWER);
  });

  test("the model list is what Copilot lists for the token, for pickers and allowed by policy", async () => {
    const { core } = await signedIn();

    expect((await core.getCopilot()).models).toEqual([
      { id: "gpt-5.5", name: "GPT-5.5" },
      { id: "claude-sonnet-5", name: "Claude Sonnet 5" },
    ]);
  });

  test("an expiring token is refreshed with the client ID alone, and the new one is used", async () => {
    const setup = await setUp();
    setup.fake.expiring = true;
    const { core, fake, clock, keychain } = await signedIn(undefined, setup);
    clock.advance(28_800 * 1000);

    await ask(core);

    const refresh = fake.requests.find((r) => r.body.grant_type === "refresh_token");
    expect(refresh?.body).toEqual({
      grant_type: "refresh_token",
      client_id: "test-own-app",
      refresh_token: fake.issued[0]?.refreshToken,
    });
    expect(chats(fake).at(-1)?.headers.authorization).toBe(`Bearer ${fake.issued[1]?.accessToken}`);
    expect(stored(keychain)).not.toContain(fake.issued[0]?.refreshToken);
  });

  test("a refused refresh, answered in a 200 as GitHub does, ends the sign-in: Sign in again", async () => {
    const setup = await setUp();
    setup.fake.expiring = true;
    const { core, fake, clock } = await signedIn(undefined, setup);
    fake.refuseRefresh = "bad_refresh_token";
    clock.advance(28_800 * 1000);

    expect(await failure(core)).toMatchObject({
      kind: "not-signed-in",
      signInRoute: "sign-in-expired",
    });
    expect((await core.getCopilot()).account).toEqual({ state: "expired" });
  });

  test("a refresh refused with a code IncarnaMind doesn't know asks to sign in again, not to wait", async () => {
    const setup = await setUp();
    setup.fake.expiring = true;
    const { core, fake, clock } = await signedIn(undefined, setup);
    fake.refuseRefresh = "refresh_not_allowed";
    clock.advance(28_800 * 1000);

    expect(await failure(core)).toMatchObject({ signInRoute: "sign-in-expired" });
  });

  test("a token that doesn't expire lasts until GitHub refuses it, then asks to sign in again", async () => {
    const { core, fake, clock } = await signedIn();
    clock.advance(365 * 24 * 3600 * 1000);
    expect((await ask(core)).text).toBe(ANSWER);

    fake.revoke();

    expect(await failure(core)).toMatchObject({ signInRoute: "sign-in-expired" });
    expect((await core.getCopilot()).account).toEqual({ state: "expired" });
  });

  test("signing out forgets the tokens, and nothing is sent afterwards", async () => {
    const { core, fake, keychain } = await signedIn();
    await ask(core);
    const sent = chats(fake).length;

    expect((await core.signOutOfCopilot()).account).toEqual({ state: "signed-out" });

    expect(stored(keychain)).not.toContain(fake.issued[0]?.accessToken);
    await expect(core.prepareChatModel()).rejects.toThrow();
    expect(chats(fake)).toHaveLength(sent);
  });

  test("choosing the route allows its flows to GitHub Copilot, and nothing is sent before", async () => {
    const setup = await setUp();
    await signIn(setup);
    expect(chats(setup.fake)).toHaveLength(0);

    await setup.core.saveChatProvider({ kind: "copilot", modelId: MODEL });

    const chat = (await setup.core.listDataFlows()).find(
      (flow) => flow.flow.id === "chat" && flow.flow.service.id === COPILOT_SERVICE.id,
    );
    expect(chat).toMatchObject({ consent: "accepted", chosen: true });
  });

  test("the Privacy page lists the sign-in's traffic to GitHub once the User has started signing in", async () => {
    const setup = await setUp();
    const ids = async () => (await setup.core.listNetworkTraffic()).map((t) => t.id);
    expect(await ids()).not.toContain("copilot-sign-in");

    await signIn(setup);

    expect(await setup.core.listNetworkTraffic()).toContainEqual({
      id: "copilot-sign-in",
      service: { id: setup.fake.origin, name: new URL(setup.fake.origin).host },
      enabled: true,
    });
  });

  test("codes and tokens are never logged", async () => {
    const logged: unknown[] = [];
    for (const method of ["log", "info", "warn", "error", "debug"] as const) {
      vi.spyOn(console, method).mockImplementation((...args) => {
        logged.push(...args);
      });
    }
    const setup = await setUp();
    setup.fake.expiring = true;
    const { core, fake, clock } = await signedIn(undefined, setup);
    clock.advance(28_800 * 1000);
    await ask(core);
    vi.restoreAllMocks();

    const text = logged.map((each) => String(each instanceof Error ? each.stack : each)).join("\n");
    for (const tokens of fake.issued) {
      expect(text).not.toContain(tokens.accessToken);
      if (tokens.refreshToken) expect(text).not.toContain(tokens.refreshToken);
    }
    expect(text).not.toContain("device-1");
  });
});

describe("A Question asked with the Copilot plan", () => {
  test("gets an Answer whose Citation is checked", async () => {
    const spring = "Spring tides happen at new moon and at full moon.";
    const { core, fake } = await signedIn();
    const sources = await createSourceFolder();
    await addAndProcess(core, [
      await writeSourceFile(sources, "Tides.md", `# Tides\n\n${spring}\n`),
    ]);
    const mind = await core.createMind({ title: "Tides" });
    const client = await connectToMind(core, mind.id);
    fake.script = (body) => {
      const messages = Array.isArray(body.messages) ? body.messages : [];
      const results = messages.flatMap((message) =>
        typeof message === "object" &&
        message !== null &&
        "role" in message &&
        message.role === "tool" &&
        "content" in message
          ? [String(message.content)]
          : [],
      );
      if (results.length === 0) {
        return { calls: [{ name: "search_documents", arguments: { query: "spring tides" } }] };
      }
      if (results.length === 1) {
        const [passage] = shownPassages(results[0] ?? "");
        return {
          calls: [
            {
              name: "cite",
              arguments: { citations: [{ marker: 1, passage: passage?.id, quote: spring }] },
            },
          ],
        };
      }
      return { text: "Spring tides come at new and full moon [^1]." };
    };

    const { answerId } = await askAndFinish(core, client, mind.id, "When are spring tides?");

    expect(citationsIn(client, answerId)).toEqual([
      expect.objectContaining({ quote: spring, check: "found" }),
    ]);
    expect(
      chats(fake).every((request) => request.headers.authorization?.startsWith("Bearer ")),
    ).toBe(true);
  });
});

describe("The editor profile, an experimental sign-in route", () => {
  test("trades the GitHub token for a Copilot token, and its requests carry the profile's headers", async () => {
    const { core, fake } = await signedIn("editor");

    expect((await ask(core)).text).toBe(ANSWER);

    const exchange = fake.requests.find((r) => r.path === "/copilot_internal/v2/token");
    expect(exchange?.headers.authorization).toBe(`Bearer ${fake.issued[0]?.accessToken}`);
    const [request] = chats(fake);
    expect(request?.headers).toMatchObject({
      authorization: `Bearer ${fake.exchanged[0]}`,
      "editor-version": EDITOR_HEADER,
    });
    expect((await core.getCopilot()).account).toEqual({ state: "signed-in", profile: "editor" });
  });

  test("an expired Copilot token is traded for a new one with the kept GitHub token", async () => {
    const { core, fake, clock } = await signedIn("editor");
    clock.advance(25 * 60_000);

    await ask(core);

    expect(fake.exchanged).toHaveLength(2);
    expect(chats(fake).at(-1)?.headers.authorization).toBe(`Bearer ${fake.exchanged[1]}`);
  });

  test("a failed token exchange says what happened, and says no plan only when GitHub does", async () => {
    const cases = [
      { set: (fake: FakeGitHub) => (fake.exchangeStatus = 503), kind: "service-error" },
      { set: (fake: FakeGitHub) => (fake.exchangeStatus = 403), kind: "refused" },
      { set: (fake: FakeGitHub) => (fake.exchangeOffline = true), kind: "unreachable" },
    ];
    for (const { set, kind } of cases) {
      const setup = await setUp();
      set(setup.fake);
      const { result } = await codeShown(setup.core, "editor");
      setup.fake.approve();

      expect(await result, kind).toMatchObject({ ok: false, error: { kind } });
      expect(setup.keychain.secrets.size).toBe(0);
    }
  });

  test("a GitHub account without Copilot is told so at sign-in, and nothing is kept", async () => {
    const setup = await setUp();
    setup.fake.exchangeStatus = 404;
    const { result } = await codeShown(setup.core, "editor");
    setup.fake.approve();

    expect(await result).toMatchObject({ ok: false, error: { kind: "no-plan" } });
    expect(setup.keychain.secrets.size).toBe(0);
  });
});

describe("Failures", () => {
  const cases: {
    name: string;
    status: number;
    body: unknown;
    kind: string;
    problem: SignInRouteProblem;
  }[] = [
    {
      name: "no Copilot plan",
      status: 403,
      body: { error: { message: "You don't have access to GitHub Copilot." } },
      kind: "blocked",
      problem: "no-plan",
    },
    {
      name: "a refusal that doesn't say there's no plan",
      status: 403,
      body: { error: { message: "Forbidden" } },
      kind: "blocked",
      problem: "refused",
    },
    {
      name: "an organization's policy",
      status: 403,
      body: { error: { message: "Copilot is disabled by your organization's policy." } },
      kind: "blocked",
      problem: "policy-blocked",
    },
    {
      name: "a model the plan doesn't offer here",
      status: 400,
      body: {
        error: { message: "The requested model is not supported.", code: "model_not_supported" },
      },
      kind: "model",
      problem: "model-unavailable",
    },
    {
      name: "the plan's limit",
      status: 402,
      body: { error: { message: "You have no premium requests left." } },
      kind: "plan-limit",
      problem: "plan-limit",
    },
    {
      name: "a used-up quota",
      status: 429,
      body: { error: { message: "Quota exceeded", code: "quota_exceeded" } },
      kind: "plan-limit",
      problem: "plan-limit",
    },
  ];

  for (const { name, status, body, kind, problem } of cases) {
    test(`${name} is "${problem}"`, async () => {
      const { core, fake } = await signedIn();
      fake.replies.push({ kind: "error", status, body: JSON.stringify(body) });

      expect(await failure(core)).toMatchObject({ kind, signInRoute: problem });
    });
  }

  test("a passing rate limit is a rate limit, with its own line", async () => {
    const { core, fake } = await signedIn();
    fake.replies.push({
      kind: "error",
      status: 429,
      body: JSON.stringify({ error: { message: "Too many requests" } }),
    });

    expect(await failure(core)).toMatchObject({ kind: "rate-limit", signInRoute: "rate-limited" });
  });

  test("a page that isn't an answer is an unexpected response", async () => {
    const { core, fake } = await signedIn();
    fake.replies.push({
      kind: "error",
      status: 200,
      headers: { "content-type": "text/html" },
      body: "<html></html>",
    });

    expect(await failure(core)).toMatchObject({ signInRoute: "unexpected-response" });
  });

  test("each problem has its own line naming GitHub, and its ways back, in English and Chinese", () => {
    const { problems } = SIGN_IN_ROUTES.copilot;
    for (const language of ["en", "zh-CN"] as const) {
      const lines = Object.values(problems).map(({ line }) => translate(language, line));
      expect(new Set(lines).size).toBe(lines.length);
      for (const line of lines) expect(line).toMatch(/GitHub|Copilot|organization|组织/);
      for (const words of Object.values(problems)) {
        for (const wayBack of words.waysBack) {
          const key = wayBackLabel("copilot", wayBack);
          expect(translate(language, key)).not.toBe(key);
        }
      }
    }
    for (const problem of [
      "no-plan",
      "policy-blocked",
      "model-unavailable",
      "plan-limit",
    ] as const) {
      expect(problems[problem]?.waysBack).toContain("use-another-model");
    }
    expect(problems["sign-in-expired"]?.waysBack[0]).toBe("sign-in-again");
    // Every failure, of a sign-in or a request, has a way back to an API key.
    const signIn = Object.values(COPILOT_SIGN_IN_PROBLEMS);
    for (const words of [...Object.values(problems), ...signIn]) {
      expect(words.waysBack).toContain("paste-api-key");
    }
    for (const language of ["en", "zh-CN"] as const) {
      for (const { line } of signIn) expect(translate(language, line)).not.toBe(line);
    }
  });
});
