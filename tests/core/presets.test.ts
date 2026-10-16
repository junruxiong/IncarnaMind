/**
 * Presets: hosted services that speak OpenAI's API, each one catalog row on
 * the shared "openai-compatible" kind. Every hosted preset in the catalog goes
 * through the same path; a row of the test's own is added only to show that a
 * row is all a preset needs. The ones on this computer are in localPresets.test.ts.
 */
import { afterEach, describe, expect, test, vi } from "vitest";
import { InvalidInputError } from "../../src/core";
import type { CatalogProvider } from "../../src/core/providers/catalog";
import { PRESETS as CATALOG_PRESETS } from "../../src/core/providers/catalog/presets";
import { catalogProvider, fieldUrl } from "../../src/core/providers/catalog/providers";
import { createAiSdkChatModel } from "../../src/core/providers/models";
import { askAndFinish, citationsIn, setUpWithDocuments } from "../helpers/citations";
import { fakeProvider, type RecordedRequest } from "../helpers/fakeProvider";
import { replyingModel } from "../helpers/models";
import { buildPdf } from "../helpers/pdf";

vi.mock("../../src/core/providers/catalog/presets", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/core/providers/catalog/presets")>();
  const tideCloud: CatalogProvider = {
    id: "tide-cloud",
    name: { en: "Tide Cloud", zh: "潮汐云" },
    support: "best-effort",
    group: "international",
    adapter: "openai-compatible",
    endpoints: [
      { id: "global", label: { en: "Global", zh: "全球" }, baseUrl: "https://api.tide.test/v1" },
    ],
    apiKey: "required",
    keyUrl: "https://tide.test/keys",
    roles: { answers: "tide-1" },
    dataUse: { training: "unknown", summary: { en: "Tide Cloud.", zh: "潮汐云。" } },
    checked: "2026-10-14",
  };
  return { PRESETS: [...actual.PRESETS, tideCloud] };
});

afterEach(() => vi.unstubAllGlobals());

const TIDES = buildPdf([
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);
const SPRING = "Spring tides happen at new moon and at full moon.";

const FIELD_VALUES: Record<string, string> = {
  region: "eu-west-1",
  account: "0123456789abcdef0123456789abcdef",
};

const PRESETS = CATALOG_PRESETS.filter((each) => each.group !== "local").map((each) => {
  const [endpoint] = each.endpoints;
  const value = endpoint?.field ? FIELD_VALUES[endpoint.field.name] : undefined;
  const base = endpoint ? fieldUrl(endpoint, value ?? "") : "";
  return { catalogId: each.id, origin: new URL(base).origin, base, value: value ?? null };
});

const posts = (requests: RecordedRequest[]) => requests.filter((each) => each.method === "POST");

describe("Presets on the shared OpenAI-compatible path", () => {
  test.each(PRESETS)(
    "$catalogId: saved as the shared kind with its own service, its models listed and a Question answered at its endpoint",
    async ({ catalogId, origin, base, value }) => {
      const modelId = catalogProvider(catalogId)?.roles.answers as string;
      const { core, client, mind } = await setUpWithDocuments(
        replyingModel(),
        [{ name: "Tides.pdf", contents: TIDES }],
        { createChatModel: createAiSdkChatModel },
      );
      const saved = await core.saveChatProvider({
        kind: "openai-compatible",
        catalogId,
        ...(value && { endpointValue: value }),
        apiKey: `key-for-${catalogId}`,
        modelId,
      });
      expect(saved).toMatchObject({
        kind: "openai-compatible",
        catalogId,
        endpoint: "global",
        baseUrl: value ? base : null,
        hasApiKey: true,
        service: { id: origin, name: catalogProvider(catalogId)?.name.en },
      });
      // Choosing it is the consent for its own origin (ADR-0022).
      expect(await core.getChatReadiness()).toMatchObject({ ready: true, consent: "accepted" });

      const requests = fakeProvider(new URL(origin).host, {
        protocol: "chat",
        models: { data: [{ id: modelId }, { id: "another-model" }] },
        query: "spring tides",
        records: (passages) => [
          { marker: 1, passage: passages[0]?.id, pageFrom: 1, pageTo: 1, quote: SPRING },
        ],
        answer: "Spring tides come at new and full moon [^1].",
      });
      const group = (await core.listChatModels()).find((each) => each.provider.id === saved.id);
      expect(group?.models).toContain(modelId);
      const listing = requests.find((request) => request.method === "GET");
      expect(listing?.url).toBe(`${base}/models`);
      expect(listing?.headers.authorization).toBe(`Bearer key-for-${catalogId}`);

      const { answerId } = await askAndFinish(core, client, mind.id, "When are spring tides?");
      expect(citationsIn(client, answerId)[0]).toMatchObject({ pageFrom: 1, check: "found" });
      expect(posts(requests).length).toBeGreaterThan(0);
      for (const request of posts(requests)) {
        expect(request.url).toBe(`${base}/chat/completions`);
        expect(request.headers.authorization).toBe(`Bearer key-for-${catalogId}`);
      }
    },
  );

  test("two presets are two providers with their own keys and their own consent", async () => {
    const { core } = await setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
    });
    const chatConsent = async (origin: string) =>
      (await core.listDataFlows()).find(
        (each) => each.flow.id === "chat" && each.flow.service.id === origin,
      )?.consent ?? "not-asked";
    const groq = await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "groq",
      apiKey: "key-groq",
      modelId: "openai/gpt-oss-120b",
    });
    expect(await chatConsent("https://api.groq.com")).toBe("accepted");
    expect(await chatConsent("https://api.tide.test")).toBe("not-asked");

    const tide = await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "tide-cloud",
      apiKey: "key-tide",
      modelId: "tide-1",
    });
    expect(tide.id).not.toBe(groq.id);
    expect(tide.service).toEqual({ id: "https://api.tide.test", name: "Tide Cloud" });
    await core.revokeConsent("chat", "https://api.groq.com");
    expect(await chatConsent("https://api.tide.test")).toBe("accepted");
    expect(await core.getChatReadiness()).toMatchObject({
      ready: true,
      consent: "accepted",
      provider: { id: tide.id },
    });

    const again = await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "groq",
      modelId: "openai/gpt-oss-20b",
    });
    expect(again).toMatchObject({ id: groq.id, hasApiKey: true });
    const saved = await core.listChatProviders();
    expect(saved.filter((each) => each.catalogId === "groq")).toHaveLength(1);
    expect(saved.filter((each) => each.catalogId === "tide-cloud")).toHaveLength(1);
  });

  test("testing a preset's connection without a key uses the key saved for that preset, and no other", async () => {
    const { core } = await setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
    });
    await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "groq",
      apiKey: "key-groq",
      modelId: "openai/gpt-oss-120b",
    });
    await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "tide-cloud",
      apiKey: "key-tide",
      modelId: "tide-1",
    });
    const requests = fakeProvider("api.groq.com", {
      protocol: "chat",
      models: { data: [] },
      query: "",
      records: () => [],
      answer: "OK",
    });
    const result = await core.testChatConnection({
      kind: "openai-compatible",
      catalogId: "groq",
      modelId: "openai/gpt-oss-120b",
    });
    expect(result.ok).toBe(true);
    expect(posts(requests).length).toBeGreaterThan(0);
    for (const request of posts(requests)) {
      expect(request.url).toBe("https://api.groq.com/openai/v1/chat/completions");
      expect(request.headers.authorization).toBe("Bearer key-groq");
    }
  });

  test("each value of an address field is a provider of its own, with its own origin and consent; a missing or wrong value is refused", async () => {
    const { core } = await setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
    });
    const chatConsent = async (origin: string) =>
      (await core.listDataFlows()).find(
        (each) => each.flow.id === "chat" && each.flow.service.id === origin,
      )?.consent ?? "not-asked";
    const bedrock = (region?: string) =>
      core.saveChatProvider({
        kind: "openai-compatible",
        catalogId: "bedrock",
        ...(region !== undefined && { endpointValue: region }),
        apiKey: "key-bedrock",
        modelId: "openai.gpt-oss-120b",
      });
    const ireland = await bedrock("eu-west-1");
    expect(ireland).toMatchObject({
      baseUrl: "https://bedrock-mantle.eu-west-1.api.aws/v1",
      service: { id: "https://bedrock-mantle.eu-west-1.api.aws", name: "Amazon Bedrock" },
    });
    expect(await chatConsent("https://bedrock-mantle.us-east-1.api.aws")).toBe("not-asked");
    const virginia = await bedrock("us-east-1");
    expect(virginia.id).not.toBe(ireland.id);
    expect(await chatConsent("https://bedrock-mantle.eu-west-1.api.aws")).toBe("accepted");
    expect(await chatConsent("https://bedrock-mantle.us-east-1.api.aws")).toBe("accepted");
    expect((await bedrock("eu-west-1")).id).toBe(ireland.id);

    await expect(bedrock()).rejects.toThrow(InvalidInputError);
    await expect(bedrock("evil.example.com/x")).rejects.toThrow(InvalidInputError);
    await expect(
      core.saveChatProvider({
        kind: "openai-compatible",
        catalogId: "cloudflare",
        endpointValue: "not-an-account",
        apiKey: "key",
        modelId: "@cf/zai-org/glm-5.3-flash",
      }),
    ).rejects.toThrow(InvalidInputError);
  });

  test("a server the User gives is still its own provider, beside a preset", async () => {
    const { core } = await setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
    });
    const groq = await core.saveChatProvider({
      kind: "openai-compatible",
      catalogId: "groq",
      apiKey: "key-groq",
      modelId: "openai/gpt-oss-120b",
    });
    const server = await core.saveChatProvider({
      kind: "openai-compatible",
      baseUrl: "https://llm.example.com/v1",
      modelId: "local-model",
    });
    expect(server.id).not.toBe(groq.id);
    expect(server).toMatchObject({
      catalogId: "openai-compatible",
      baseUrl: "https://llm.example.com/v1",
      service: { id: "https://llm.example.com", name: "llm.example.com" },
    });
  });

  test("a preset takes no server address, only the shared kind takes one, and an unknown one is refused", async () => {
    const { core } = await setUpWithDocuments(replyingModel(), [], {
      createChatModel: createAiSdkChatModel,
    });
    const groq = { catalogId: "groq", apiKey: "key", modelId: "openai/gpt-oss-120b" };
    await expect(
      core.saveChatProvider({
        kind: "openai-compatible",
        ...groq,
        baseUrl: "https://elsewhere.example.com/v1",
      }),
    ).rejects.toThrow(InvalidInputError);
    await expect(core.saveChatProvider({ kind: "openai", ...groq })).rejects.toThrow(
      InvalidInputError,
    );
    await expect(
      core.saveChatProvider({ kind: "openai-compatible", ...groq, catalogId: "nowhere" }),
    ).rejects.toThrow(InvalidInputError);
    await expect(
      core.saveChatProvider({ kind: "openai-compatible", catalogId: "groq", modelId: "x" }),
    ).rejects.toThrow();
  });
});
