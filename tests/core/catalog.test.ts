import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { describe, expect, test } from "vitest";
import { RECOMMENDED_OLLAMA_MODEL } from "../../src/core";
import { type ChatProviderKind, chatProviderKinds } from "../../src/core/api";
import {
  cloudWindow,
  givesStructuredOutput,
  readsImages,
  resolveCapabilities,
  startingSupport,
} from "../../src/core/providers/capabilities";
import {
  CATALOG_PROVIDERS,
  catalogFacts,
  catalogModel,
  catalogModels,
  isNonChatModel,
  LOCAL_MODEL_TIERS,
  LOCAL_MODELS,
  modelsProviderOfKind,
  providersWithModels,
  recommendedLocalModels,
} from "../../src/core/providers/catalog";
import { OVERRIDES } from "../../src/core/providers/catalog/overrides";
import { catalogProvider } from "../../src/core/providers/catalog/providers";
import {
  baseUrlFor,
  endpointUrl,
  requiresApiKey,
  serviceFor,
} from "../../src/core/providers/kinds";
import { en } from "../../src/shared/i18n/en";
import { zhCN } from "../../src/shared/i18n/zh-CN";

const GIB = 1024 ** 3;
const REPO_ROOT = resolve(__dirname, "../..");

/** What the catalog alone says of a model on a provider kind. */
const fromCatalog = (kind: ChatProviderKind, modelId: string) =>
  resolveCapabilities({ catalog: catalogFacts(modelsProviderOfKind(kind), modelId) });

describe("The provider catalog", () => {
  test("every provider has both names, a default for each role it offers, and a data-use line", () => {
    expect(CATALOG_PROVIDERS.map((provider) => provider.id)).toEqual([
      "anthropic",
      "openai",
      "google",
      "openrouter",
      "mistral",
      "xai",
      "deepseek",
      "qwen",
      "kimi",
      "glm",
      "siliconflow",
      "ollama",
      "openai-compatible",
    ]);
    for (const provider of CATALOG_PROVIDERS) {
      const where = provider.id;
      expect(provider.name.en.trim(), where).not.toBe("");
      expect(provider.name.zh.trim(), where).not.toBe("");
      expect(provider.dataUse.summary.en.trim(), where).not.toBe("");
      expect(provider.dataUse.summary.zh.trim(), where).not.toBe("");
      expect(["no", "opt-out", "free-tier", "unknown"], where).toContain(provider.dataUse.training);
      expect(provider.checked, where).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      for (const note of provider.notes ?? []) {
        expect(note.en.trim() && note.zh.trim(), where).not.toBe("");
      }
      // A hosted provider has its API; a server the User gives has a URL rule instead.
      if (provider.endpoints.length > 0) {
        for (const endpoint of provider.endpoints) {
          expect(new URL(endpoint.baseUrl).protocol, where).toBe("https:");
          expect(endpoint.label.en && endpoint.label.zh, where).toBeTruthy();
        }
      } else {
        expect(provider.serverUrl, where).toBeDefined();
      }
      // Each role names a model the catalog knows: a cloud model, or one recommended for Ollama.
      for (const [role, modelId] of Object.entries(provider.roles)) {
        const known =
          provider.id === "ollama"
            ? LOCAL_MODELS.some((model) => model.tag === modelId)
            : catalogModel(provider.id, modelId) !== undefined;
        expect(known, `${where} ${role}: ${modelId}`).toBe(true);
      }
    }
  });

  test("a hosted provider offers every role: a capable, cheaper Answers model, its strongest one click away, a cheap quick-tasks model and one that reads images", () => {
    for (const provider of CATALOG_PROVIDERS.filter((each) => each.endpoints.length > 0)) {
      const { answers, strongest, quickTasks, images } = provider.roles;
      const model = (id: string | undefined) => {
        const found = id ? catalogModel(provider.id, id) : undefined;
        if (!found) throw new Error(`${provider.id} has no model ${id}.`);
        return found;
      };
      expect(model(answers).tools, provider.id).toBe(true);
      expect(model(images).input, provider.id).toContain("image");
      // The Answers model costs less than the strongest, and a quick-tasks model no more than it
      // (where the strongest's price is known: SiliconFlow's yuan price for it is not listed).
      const output = (id: string | undefined) => model(id).price?.output;
      expect(output(answers), provider.id).toBeDefined();
      if (output(strongest) !== undefined) {
        expect(output(answers) ?? Number.NaN, provider.id).toBeLessThan(output(strongest) ?? 0);
      }
      expect(output(quickTasks) ?? Number.NaN, provider.id).toBeLessThanOrEqual(
        output(answers) ?? Number.NaN,
      );
      // Prices in the provider's own currency: yuan on a Chinese endpoint of a provider that bills in it.
      const yuan = provider.id === "deepseek" || provider.endpoints[0]?.id === "cn";
      expect(model(answers).price?.currency, provider.id).toBe(yuan ? "CNY" : "USD");
    }
  });

  test("the generated facts cover the hosted providers, with unique ids, and their sources' licences ship with the app", async () => {
    expect(providersWithModels().sort()).toEqual([
      "anthropic",
      "deepseek",
      "glm",
      "glm/intl",
      "google",
      "kimi",
      "kimi/intl",
      "mistral",
      "openai",
      "openrouter",
      "qwen",
      "qwen/intl",
      "siliconflow",
      "siliconflow/intl",
      "xai",
    ]);
    for (const providerId of providersWithModels()) {
      const ids = catalogModels(providerId).map((model) => model.id);
      expect(new Set(ids).size, providerId).toBe(ids.length);
      for (const model of catalogModels(providerId)) {
        expect(model.context, model.id).toBeGreaterThan(0);
        expect(model.input, model.id).toContain("text");
        // A chat model is never also listed as something else.
        expect(isNonChatModel(providerId, model.id), model.id).toBe(false);
      }
    }
    const notice = await readFile(join(REPO_ROOT, "resources/notices/model-catalog.txt"), "utf8");
    expect(notice).toContain("Copyright (c) 2025 models.dev");
    expect(notice).toContain("Copyright (c) 2023 Berri AI");
    expect(notice.match(/Permission is hereby granted/g)).toHaveLength(2);
    const builder = await readFile(join(REPO_ROOT, "electron-builder.yml"), "utf8");
    expect(builder).toMatch(/- from: resources\/notices\n\s+to: notices\n/);
  });

  test("every hand-written correction names a model the generated facts have", () => {
    for (const providerId of Object.keys(OVERRIDES.providers)) {
      expect(providersWithModels()).toContain(providerId);
    }
    for (const [providerId, models] of Object.entries(OVERRIDES.models)) {
      for (const modelId of Object.keys(models)) {
        expect(catalogModel(providerId, modelId)?.id, `${providerId}/${modelId}`).toBe(modelId);
      }
    }
    // Corrections apply: Claude Sonnet 5.5 refuses a forced Tool choice, and Gemini runs at its default.
    expect(catalogModel("anthropic", "claude-sonnet-5-5")?.forcedToolChoice).toBe(false);
    expect(catalogModel("google", "gemini-2.5-flash")?.temperature).toBe(false);
    expect(catalogModel("google", "gemini-3.8-flash")?.price?.note?.en).toContain("2027-01-01");
  });

  test("a model is found by its id, or a dated snapshot by the id without its date; nothing else is guessed from a name", () => {
    expect(catalogModel("openai", "gpt-6.1-sol")?.name).toBe("GPT-6.1 Sol");
    expect(catalogModel("openai", "gpt-5-2025-08-07")?.id).toBe("gpt-5");
    expect(catalogModel("anthropic", "claude-haiku-4-5-20251001")?.id).toBe(
      "claude-haiku-4-5-20251001",
    );
    expect(catalogModel("anthropic", "claude-sonnet-4-6-20991231")?.id).toBe("claude-sonnet-4-6");
    expect(catalogModel("openai", "gpt-6.1-sol-mini")).toBeUndefined();
    expect(catalogModel("openai", "openai/gpt-6.1-sol")).toBeUndefined();
    expect(catalogModel("google", "models/gemini-3.8-flash")).toBeUndefined();
    // Ids its sources know as something else than a chat model.
    expect(isNonChatModel("openai", "text-embedding-3-small")).toBe(true);
    expect(isNonChatModel("openai", "whisper-1")).toBe(true);
    expect(isNonChatModel("google", "gemini-embedding-001")).toBe(true);
    expect(isNonChatModel("openai", "gpt-7-preview")).toBe(false);
  });

  test.each<[ChatProviderKind, string, boolean]>([
    ["anthropic", "claude-sonnet-5-5", true],
    ["anthropic", "claude-haiku-4-5-20251001", true],
    ["anthropic", "claude-2.1", false],
    ["openai", "gpt-5.5", true],
    ["openai", "gpt-4o-mini", true],
    ["openai", "o4-mini", true],
    ["openai", "o3-mini", false],
    ["openai", "gpt-3.5-turbo", false],
    ["openai", "gpt-7-preview", false],
    ["chatgpt", "gpt-6-sol", true],
    ["google", "gemini-3.8-flash", true],
    ["google", "gemini-pro-latest", false],
    ["openai-compatible", "qwen-vl-max", false],
    ["ollama", "llava", false],
  ])("%s %s reads images by the catalog: %s", (kind, modelId, reads) => {
    expect(readsImages(fromCatalog(kind, modelId))).toBe(reads);
  });

  test("models recommended for Ollama follow this computer's memory", () => {
    const tags = (gb: number) => recommendedLocalModels(gb * GIB).map((model) => model.tag);
    expect(tags(4)).toEqual(["qwen3.5:4b", "qwen3.5:2b"]);
    expect(tags(8)).toEqual(["qwen3.5:4b", "qwen3.5:2b"]);
    expect(tags(16)).toEqual(["qwen3.5:4b", "qwen3.5:9b", "gemma4:12b"]);
    expect(tags(15.6)).toEqual(tags(16));
    expect(tags(32)[0]).toBe("qwen3.5:9b");
    expect(tags(64)[0]).toBe("qwen3.6:35b-a3b");
    expect(tags(128)).toEqual(tags(64));
    for (const tier of LOCAL_MODEL_TIERS) {
      for (const tag of tier.models) {
        expect(
          LOCAL_MODELS.some((model) => model.tag === tag),
          tag,
        ).toBe(true);
      }
    }
    // Only a model the evaluation measured counts as tested; the one-click model is one.
    expect(LOCAL_MODELS.filter((model) => model.evaluated).map((model) => model.tag)).toEqual([
      RECOMMENDED_OLLAMA_MODEL,
    ]);
  });
});

describe("What is known of a model", () => {
  test("the User's word wins over the server's, the server's over the catalog's, and what none says is unknown", () => {
    const known = resolveCapabilities({
      user: { input: ["text", "image"] },
      server: { context: 32_768, tools: false, input: ["text"] },
      catalog: {
        context: 128_000,
        tools: true,
        temperature: false,
        structuredOutput: "json_schema",
      },
    });
    expect(known.known).toBe(true);
    expect(known.facts).toEqual({
      input: ["text", "image"],
      context: 32_768,
      tools: false,
      temperature: false,
      structuredOutput: "json_schema",
    });
    expect(readsImages(known)).toBe(true);
    expect(startingSupport(known)).toBe("structured-output");

    // A source leaving a fact out doesn't hide the next one's.
    expect(
      resolveCapabilities({ server: { tools: undefined }, catalog: { tools: true } }).facts.tools,
    ).toBe(true);
  });

  test("a model nothing says anything about keeps today's behaviour: Tools first, a temperature, no images, no window", () => {
    const unknown = resolveCapabilities({});
    expect(unknown.known).toBe(false);
    expect(readsImages(unknown)).toBe(false);
    expect(startingSupport(unknown)).toBeUndefined();
    expect(givesStructuredOutput(unknown)).toBeUndefined();
    expect(cloudWindow(unknown)).toBeUndefined();
    expect(unknown.facts.temperature).toBeUndefined();
  });

  test("how a model cites starts from what it can do, and a measured or server-decided way wins", () => {
    const support = (facts: Parameters<typeof resolveCapabilities>[0]["catalog"]) =>
      startingSupport(resolveCapabilities({ catalog: facts }));
    expect(support({ tools: true })).toBe("tools");
    expect(support({ tools: false, structuredOutput: "json_object" })).toBe("structured-output");
    expect(support({ tools: false })).toBe("structured-output");
    expect(support({ tools: false, structuredOutput: "none" })).toBe("none");
    expect(support({ tools: true, citing: "structured-output" })).toBe("structured-output");
  });

  test("a cloud model's window is its context, less its input limit or the room for its longest reply", () => {
    const window = (facts: Parameters<typeof resolveCapabilities>[0]["catalog"]) =>
      cloudWindow(resolveCapabilities({ catalog: facts }));
    expect(window({ context: 1_050_000, maxInput: 922_000, maxOutput: 128_000 })).toEqual({
      tokens: 1_050_000,
      outputTokens: 128_000,
    });
    expect(window({ context: 1_000_000, maxOutput: 128_000 })).toEqual({
      tokens: 1_000_000,
      outputTokens: 128_000,
    });
    // The reply never takes more than half the window.
    expect(window({ context: 8_192, maxOutput: 8_192 })).toEqual({
      tokens: 8_192,
      outputTokens: 4_096,
    });
    expect(window({ maxOutput: 8_192 })).toBeUndefined();
  });
});

describe("The providers of #178", () => {
  const added = ["openrouter", "mistral", "xai", "deepseek", "qwen", "kimi", "glm", "siliconflow"];

  test("each is a chat provider kind with its own catalog entry, names in both languages, and a key link", () => {
    for (const id of added) {
      const provider = catalogProvider(id);
      expect(provider, id).toBeDefined();
      expect(chatProviderKinds as readonly string[], id).toContain(id);
      expect(provider?.apiKey, id).toBe("required");
      expect(provider?.keyUrl, id).toMatch(/^https:\/\//);
      for (const endpoint of provider?.endpoints ?? []) {
        expect(endpoint.keyUrl ?? provider?.keyUrl, `${id} ${endpoint.id}`).toMatch(/^https:/);
      }
      // The names the page shows are the catalog's, in both languages.
      expect(en[`providers.kind.${id as ChatProviderKind}`], id).toBe(provider?.name.en);
      expect(zhCN[`providers.kind.${id as ChatProviderKind}`], id).toBe(provider?.name.zh);
    }
  });

  test("each says whether it may train on API data; DeepSeek may (with an opt-out) and Kimi's pages disagree", () => {
    const training = Object.fromEntries(
      added.map((id) => [id, catalogProvider(id)?.dataUse.training]),
    );
    expect(training).toEqual({
      openrouter: "no",
      mistral: "opt-out",
      xai: "no",
      deepseek: "opt-out",
      qwen: "no",
      kimi: "unknown",
      glm: "unknown",
      siliconflow: "no",
    });
    expect(catalogProvider("kimi")?.dataUse.summary.en).toContain("disagree");
  });

  test("a provider with regions has an endpoint for each, the first the default, and a set of model facts for each", () => {
    const regions = (id: string) => catalogProvider(id)?.endpoints.map((each) => each.id);
    expect(regions("qwen")).toEqual(["cn", "intl"]);
    expect(regions("kimi")).toEqual(["cn", "intl"]);
    expect(regions("glm")).toEqual(["cn", "intl"]);
    expect(regions("siliconflow")).toEqual(["cn", "intl"]);
    expect(regions("deepseek")).toEqual(["global"]);
    expect(endpointUrl("qwen")).toBe("https://dashscope.aliyuncs.com/compatible-mode/v1");
    expect(endpointUrl("qwen", "intl")).toBe(
      "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    );
    expect(endpointUrl("kimi", "cn")).toBe("https://api.moonshot.cn/v1");
    expect(endpointUrl("kimi", "intl")).toBe("https://api.moonshot.ai/v1");
    expect(endpointUrl("glm", "cn")).toBe("https://open.bigmodel.cn/api/paas/v4");
    expect(endpointUrl("glm", "intl")).toBe("https://api.z.ai/api/paas/v4");
    expect(endpointUrl("siliconflow", "intl")).toBe("https://api.siliconflow.com/v1");
    // The region picks the model set; one without its own uses the provider's.
    expect(modelsProviderOfKind("qwen")).toBe("qwen");
    expect(modelsProviderOfKind("qwen", "cn")).toBe("qwen");
    expect(modelsProviderOfKind("qwen", "intl")).toBe("qwen/intl");
    expect(modelsProviderOfKind("deepseek", "global")).toBe("deepseek");
    // Each region is its own service for consent: data goes to its host.
    expect(serviceFor("qwen", null, "cn")?.id).toBe("https://dashscope.aliyuncs.com");
    expect(serviceFor("qwen", null, "intl")?.id).toBe("https://dashscope-intl.aliyuncs.com");
  });

  test("prices are in the provider's own currency and the defaults are as chosen", () => {
    const answers = (id: string, set = id) => {
      const model = catalogModel(set, catalogProvider(id)?.roles.answers ?? "");
      return [model?.id, model?.price?.currency, model?.price?.input, model?.price?.output];
    };
    expect(answers("deepseek")).toEqual(["deepseek-flash", "CNY", 2, 8]);
    expect(answers("qwen")).toEqual(["qwen3.7-plus", "CNY", 2, 8]);
    expect(answers("qwen", "qwen/intl")).toEqual(["qwen3.7-plus", "USD", 0.4, 1.6]);
    expect(answers("kimi")).toEqual(["kimi-k2.6", "CNY", 6.5, 27]);
    expect(answers("kimi", "kimi/intl")).toEqual(["kimi-k2.6", "USD", 0.95, 4]);
    expect(answers("glm")).toEqual(["glm-5.3-flash", "CNY", 0.8, 2.8]);
    expect(answers("glm", "glm/intl")).toEqual(["glm-5.3-flash", "USD", 0.15, 0.5]);
    expect(answers("siliconflow")).toEqual(["deepseek-ai/DeepSeek-V4-Flash", "CNY", 3, 9]);
    expect(answers("mistral")).toEqual(["mistral-small-2603", "USD", 0.15, 0.6]);
    expect(answers("xai")).toEqual(["grok-4.3", "USD", 1.25, 2.5]);
    expect(catalogProvider("kimi")?.roles.strongest).toBe("kimi-k3");
    expect(catalogProvider("glm")?.roles.quickTasks).toBe("glm-4.7-flash");
    // The corrections: Kimi fixes its sampling, GLM and DeepSeek take JSON mode only.
    expect(catalogModel("kimi", "kimi-k2.6")?.temperature).toBe(false);
    expect(catalogModel("glm", "glm-5.3")?.structuredOutput).toBe("json_object");
    expect(catalogModel("deepseek", "deepseek-flash")?.structuredOutput).toBe("json_object");
  });

  test("their requests carry what the provider needs", () => {
    expect(catalogProvider("xai")?.request?.store).toBe(false);
    expect(catalogProvider("openrouter")?.request?.body).toEqual({
      provider: { data_collection: "deny", require_parameters: true },
    });
    expect(catalogProvider("siliconflow")?.request?.openAiCompatible).toEqual({
      includeUsage: true,
      supportsStructuredOutputs: true,
    });
    // Every one that thinks by default says how a small call turns it off.
    for (const id of ["deepseek", "qwen", "kimi", "glm", "siliconflow", "openrouter", "xai"]) {
      expect(catalogProvider(id)?.request?.thinkingOff, id).toBeDefined();
    }
    expect(catalogProvider("glm")?.request?.thinkingOff?.unless).toEqual(["glm-5.3"]);
  });

  test("a provider needs a key and takes no server URL; a region comes with the endpoint", () => {
    for (const id of added) {
      expect(requiresApiKey(id as ChatProviderKind), id).toBe(true);
      expect(() => baseUrlFor(id as ChatProviderKind, "https://example.com"), id).toThrow(
        /doesn't take a server URL/,
      );
      expect(baseUrlFor(id as ChatProviderKind, undefined), id).toBeNull();
    }
  });
});
