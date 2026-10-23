/**
 * A key from either region (#1572): Kimi, Qwen, GLM and SiliconFlow each make keys per region, and
 * the key form asks no region. It tests a new key with `anyRegion`, from the region it would
 * have picked, and saves on the region that took it, as here. The real AI SDK providers talk to a
 * fake `fetch` whose regions take only the key made there.
 */
import { afterEach, describe, expect, test, vi } from "vitest";
import type { Core } from "../../src/core";
import type { ChatProviderKind, TestChatConnectionInput } from "../../src/core/api";
import { catalogProvider, endpointOf } from "../../src/core/providers/catalog/providers";
import { createAiSdkChatModel } from "../../src/core/providers/models";
import { askAndFinish, citationsIn, setUpWithDocuments } from "../helpers/citations";
import { fakeProvider } from "../helpers/fakeProvider";
import { replyingModel } from "../helpers/models";
import { buildPdf } from "../helpers/pdf";

afterEach(() => vi.unstubAllGlobals());

const TIDES = buildPdf([
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);
const SPRING = "Spring tides happen at new moon and at full moon.";
const KEY = "sk-made-in-one-region";

type Region = "cn" | "intl";
const OTHER: Record<Region, Region> = { cn: "intl", intl: "cn" };

function hostOf(kind: ChatProviderKind, region: Region): URL {
  const provider = catalogProvider(kind);
  const endpoint = provider && endpointOf(provider, region);
  if (endpoint?.id !== region) throw new Error(`${kind} has no ${region} region.`);
  return new URL(endpoint.baseUrl);
}

/** A reply as these providers give it, and their refusal of a key they didn't make (HTTP 401). */
const completion = (model: string) =>
  Response.json({
    id: "chat_1",
    object: "chat.completion",
    created: 1_790_000_000,
    model,
    choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: "OK" } }],
    usage: { prompt_tokens: 12, completion_tokens: 1, total_tokens: 13 },
  });
const refusal = () =>
  Response.json(
    { error: { message: "Invalid Authentication", type: "invalid_authentication_error" } },
    { status: 401 },
  );

/**
 * In place of `fetch`: each region's API, `answer` deciding what it says to each request. Returns
 * the hosts the key went to, in order (the Ollama the set-up saved is asked too, without it).
 */
function regionsAnswering(
  answer: (host: string, key: string | null) => Response | "offline",
): string[] {
  const keyWentTo: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input instanceof Request ? input.url : input));
      const key = new Headers(init?.headers).get("authorization")?.replace(/^Bearer /, "") ?? null;
      if (key === KEY) keyWentTo.push(url.host);
      const reply = answer(url.host, key);
      if (reply === "offline") throw new TypeError("fetch failed");
      return reply;
    }),
  );
  return keyWentTo;
}

/** Only the region at `home` takes the key; every other refuses it. */
const takenOnlyAt = (home: URL, model: string) =>
  regionsAnswering((host, key) =>
    host === home.host && key === KEY ? completion(model) : refusal(),
  );

const formTest = (kind: ChatProviderKind, region: Region): TestChatConnectionInput => ({
  kind,
  endpoint: region,
  apiKey: KEY,
  modelId: catalogProvider(kind)?.roles.answers as string,
  anyRegion: true,
});

/** What the key form does with the test's result: it saves on the region that took the key. */
async function saveAsTheFormDoes(core: Core, input: TestChatConnectionInput) {
  const result = await core.testChatConnection(input);
  if (!result.ok) throw new Error(`The test failed: ${result.error.kind}`);
  const { anyRegion: _, ...request } = input;
  return core.saveChatProvider({
    ...request,
    ...(result.endpoint && { endpoint: result.endpoint }),
  });
}

const KINDS = ["kimi", "qwen", "glm", "siliconflow"] as const satisfies ChatProviderKind[];

describe.each(KINDS)("%s", (kind) => {
  test.each([
    { home: "intl", first: "cn" },
    { home: "cn", first: "intl" },
  ] as const)(
    "a $home key, tried first at $first, is saved on $home with no region asked, and a Question goes there alone",
    async ({ home, first }) => {
      const modelId = catalogProvider(kind)?.roles.answers as string;
      const homeUrl = hostOf(kind, home);
      const otherUrl = hostOf(kind, OTHER[home]);
      const { core, client, mind } = await setUpWithDocuments(
        replyingModel(),
        [{ name: "Tides.pdf", contents: TIDES }],
        { createChatModel: createAiSdkChatModel },
      );
      const keyWentTo = takenOnlyAt(homeUrl, modelId);

      expect(await core.testChatConnection(formTest(kind, first))).toEqual({
        ok: true,
        endpoint: home,
      });
      // The region tried first refused it, then the other took it: the key went to each once.
      expect(keyWentTo).toEqual([otherUrl.host, homeUrl.host]);

      const saved = await saveAsTheFormDoes(core, formTest(kind, first));
      expect(saved).toMatchObject({
        kind,
        endpoint: home,
        hasApiKey: true,
        service: { id: homeUrl.origin },
      });
      expect(await core.getChatReadiness()).toMatchObject({ ready: true, modelId });
      // Choosing it allowed its flows for the region that took the key, and none for the other.
      const flows = await core.listDataFlows();
      expect(flows.map((each) => each.flow.service.id)).toContain(homeUrl.origin);
      expect(flows.map((each) => each.flow.service.id)).not.toContain(otherUrl.origin);

      // A fake that fails any request to another host: the Answer goes to the key's region only.
      const requests = fakeProvider(homeUrl.host, {
        protocol: "chat",
        models: { data: [{ id: modelId }] },
        query: "spring tides",
        records: (passages) => [
          { marker: 1, passage: passages[0]?.id, pageFrom: 1, pageTo: 1, quote: SPRING },
        ],
        answer: "Spring tides come at new and full moon [^1].",
      });
      const { answerId } = await askAndFinish(core, client, mind.id, "When are spring tides?");
      expect(citationsIn(client, answerId)).toHaveLength(1);
      const posts = requests.filter((each) => each.method === "POST");
      expect(posts.length).toBeGreaterThan(0);
      for (const request of posts) expect(request.headers.authorization).toBe(`Bearer ${KEY}`);
    },
  );

  test("a key the region tried first takes goes nowhere else", async () => {
    const { core } = await setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
    });
    for (const region of ["cn", "intl"] as const) {
      const keyWentTo = takenOnlyAt(hostOf(kind, region), formTest(kind, region).modelId);
      expect(await core.testChatConnection(formTest(kind, region))).toEqual({
        ok: true,
        endpoint: region,
      });
      expect(keyWentTo).toEqual([hostOf(kind, region).host]);
    }
  });

  test("a key no region takes is refused as a wrong key, after one try at each, and nothing is saved", async () => {
    const { core } = await setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
    });
    const before = await core.listChatProviders();
    const keyWentTo = regionsAnswering(() => refusal());

    const result = await core.testChatConnection(formTest(kind, "cn"));

    expect(result).toMatchObject({ ok: false, error: { kind: "auth" } });
    expect(keyWentTo).toEqual([hostOf(kind, "cn").host, hostOf(kind, "intl").host]);
    expect(await core.listChatProviders()).toEqual(before);
    expect(await core.listDataFlows()).toEqual([]);
  });
});

describe("what ends the search, and what never starts it", () => {
  const kimi = (region: Region) => hostOf("kimi", region);

  test.each([
    {
      name: "a rate limit",
      reply: () => Response.json({ error: { message: "Too many requests" } }, { status: 429 }),
      kind: "rate-limit",
    },
    { name: "no connection", reply: "offline" as const, kind: "network" },
  ])(
    "$name at the first region is that region's failure: the key goes nowhere else",
    async ({ reply, kind }) => {
      const { core } = await setUpWithDocuments(replyingModel(), [], {
        createChatModel: createAiSdkChatModel,
      });
      const keyWentTo = regionsAnswering(() => (reply === "offline" ? "offline" : reply()));

      const result = await core.testChatConnection(formTest("kimi", "intl"));

      expect(result).toMatchObject({ ok: false, error: { kind } });
      expect(keyWentTo).toEqual([kimi("intl").host]);
    },
  );

  test("without anyRegion, a key is tried only at the region given", async () => {
    const { core } = await setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
    });
    const keyWentTo = takenOnlyAt(kimi("cn"), "kimi-k2.6");
    const { anyRegion: _, ...input } = formTest("kimi", "intl");

    expect(await core.testChatConnection(input)).toMatchObject({
      ok: false,
      error: { kind: "auth" },
    });
    expect(keyWentTo).toEqual([kimi("intl").host]);
  });

  test("a stored key belongs to its region: tested without a new key, it goes nowhere else", async () => {
    const { core } = await setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
    });
    await core.saveChatProvider({
      kind: "kimi",
      endpoint: "intl",
      apiKey: KEY,
      modelId: "kimi-k2.6",
    });
    const keyWentTo = takenOnlyAt(kimi("cn"), "kimi-k2.6");
    const { apiKey: _, ...input } = formTest("kimi", "intl");

    expect(await core.testChatConnection(input)).toMatchObject({
      ok: false,
      error: { kind: "auth" },
    });
    expect(keyWentTo).toEqual([kimi("intl").host]);
  });

  test("a provider with one region is tried there alone", async () => {
    const { core } = await setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
    });
    const keyWentTo = regionsAnswering(() => completion("deepseek-v4-flash"));
    const modelId = catalogProvider("deepseek")?.roles.answers as string;

    expect(
      await core.testChatConnection({ kind: "deepseek", apiKey: KEY, modelId, anyRegion: true }),
    ).toEqual({ ok: true });
    expect(keyWentTo).toHaveLength(1);
  });
});
