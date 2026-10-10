/**
 * The model providers IncarnaMind offers, written by hand: their names,
 * endpoints, adapter, "get a key" link, request quirks, what they do with API
 * data, and their models for each role. Checked against each provider's own
 * pages on the date in `checked`.
 *
 * A provider's Answers default is its cheaper capable model; its strongest
 * stays one click away. A changed default needs the citing evaluation to pass
 * on the new model first.
 *
 * This module holds no model facts, so the renderer can import it for names
 * and defaults without the generated data (see ./index).
 */
import type { ChatProviderKind } from "../../api";
import type { CatalogProvider, Endpoint } from "./types";

const GLOBAL = { en: "Global", zh: "全球" };

const PROVIDERS = [
  {
    id: "anthropic",
    name: { en: "Anthropic", zh: "Anthropic" },
    group: "international",
    adapter: "anthropic",
    endpoints: [{ id: "global", label: GLOBAL, baseUrl: "https://api.anthropic.com/v1" }],
    apiKey: "required",
    keyUrl: "https://platform.claude.com/settings/keys",
    docsUrl: "https://docs.anthropic.com/en/docs/about-claude/models",
    roles: {
      answers: "claude-sonnet-5-5",
      strongest: "claude-fable-5-1",
      quickTasks: "claude-haiku-5-5",
      images: "claude-sonnet-5-5",
    },
    dataUse: {
      training: "no",
      summary: {
        en: "Doesn't train on API data by default, and keeps it for up to 30 days.",
        zh: "默认不用 API 数据训练模型，数据最多保留 30 天。",
      },
      url: "https://www.anthropic.com/legal/commercial-terms",
    },
    notes: [
      {
        en: "Not available in mainland China or Hong Kong.",
        zh: "中国大陆和香港无法使用。",
      },
    ],
    checked: "2026-10-10",
  },
  {
    id: "openai",
    name: { en: "OpenAI", zh: "OpenAI" },
    group: "international",
    adapter: "openai",
    endpoints: [{ id: "global", label: GLOBAL, baseUrl: "https://api.openai.com/v1" }],
    apiKey: "required",
    keyUrl: "https://platform.openai.com/api-keys",
    docsUrl: "https://platform.openai.com/docs/models",
    // The Responses API keeps each request on OpenAI's servers unless it says not to.
    request: { store: false },
    roles: {
      answers: "gpt-6.1-sol",
      strongest: "gpt-6-astra",
      quickTasks: "gpt-6-luna",
      images: "gpt-6.1-sol",
    },
    dataUse: {
      training: "no",
      summary: {
        en: "Doesn't train on API data unless you opt in. IncarnaMind asks it not to store requests; it may keep logs for abuse monitoring for up to 30 days.",
        zh: "除非你选择加入，否则不用 API 数据训练模型。IncarnaMind 会要求它不保存请求；它可能为防滥用保留日志，最多 30 天。",
      },
      url: "https://platform.openai.com/docs/guides/your-data",
    },
    notes: [
      {
        en: "Not available in mainland China or Hong Kong.",
        zh: "中国大陆和香港无法使用。",
      },
    ],
    checked: "2026-10-10",
  },
  {
    id: "google",
    name: { en: "Google", zh: "Google" },
    group: "international",
    adapter: "google",
    endpoints: [
      {
        id: "global",
        label: GLOBAL,
        baseUrl: "https://generativelanguage.googleapis.com/v1beta",
      },
    ],
    apiKey: "required",
    keyUrl: "https://aistudio.google.com/apikey",
    docsUrl: "https://ai.google.dev/gemini-api/docs/models",
    roles: {
      answers: "gemini-3.8-flash",
      strongest: "gemini-3.1-pro-preview",
      quickTasks: "gemini-3.1-flash-lite",
      images: "gemini-3.8-flash",
    },
    dataUse: {
      training: "free-tier",
      summary: {
        en: "On the free tier, Google uses API data to improve its products, with human review; on a paid plan it doesn't.",
        zh: "免费层级的 API 数据会被 Google 用于改进其产品，并可能经人工审阅；付费方案则不会。",
      },
      url: "https://ai.google.dev/gemini-api/terms",
    },
    notes: [
      {
        en: "Not available in mainland China or Hong Kong. In the EEA, the UK and Switzerland, only paid plans.",
        zh: "中国大陆和香港无法使用。欧洲经济区、英国和瑞士仅限付费方案。",
      },
    ],
    checked: "2026-10-10",
  },
  {
    id: "ollama",
    name: { en: "Ollama", zh: "Ollama" },
    group: "local",
    adapter: "ollama",
    endpoints: [],
    serverUrl: "optional",
    apiKey: "none",
    docsUrl: "https://ollama.com/library",
    // The model measured with IncarnaMind (#67); the loaded model also does the small jobs,
    // as a second local model would push it out of memory. Others by this computer's
    // memory are in ./local.
    roles: { answers: "qwen3.5:4b", quickTasks: "qwen3.5:4b" },
    dataUse: {
      training: "no",
      summary: {
        en: "Downloaded models run on this computer: what you ask stays on it.",
        zh: "下载的模型在这台电脑上运行：你的提问不会离开这台电脑。",
      },
    },
    checked: "2026-10-10",
  },
  {
    id: "openai-compatible",
    name: { en: "OpenAI-compatible server", zh: "OpenAI 兼容服务器" },
    group: "other",
    adapter: "openai-compatible",
    endpoints: [],
    serverUrl: "required",
    apiKey: "optional",
    roles: {},
    dataUse: {
      training: "unknown",
      summary: {
        en: "Depends on the server: check its terms. A server on this computer sends nothing out.",
        zh: "取决于服务器，请查看其条款。在这台电脑上运行的服务器不会向外发送任何内容。",
      },
    },
    checked: "2026-10-10",
  },
] as const satisfies readonly CatalogProvider[];

export type CatalogProviderId = (typeof PROVIDERS)[number]["id"];

/** The providers, in the order the catalog lists them. */
export const CATALOG_PROVIDERS: readonly CatalogProvider[] = PROVIDERS;

const byId = new Map<string, CatalogProvider>(CATALOG_PROVIDERS.map((each) => [each.id, each]));

/** The provider with this catalog id, if the catalog has it. */
export const catalogProvider = (id: string | null | undefined): CatalogProvider | undefined =>
  id ? byId.get(id) : undefined;

/**
 * The catalog provider a provider kind stands for: each kind is one today,
 * except the ChatGPT plan, which signs in instead (see ../chatgpt).
 */
export const catalogProviderOfKind = (kind: ChatProviderKind): CatalogProvider | undefined =>
  catalogProvider(kind);

/** The provider's endpoint with this id, else its first (the default); none for a server the User gives. */
export const endpointOf = (
  provider: CatalogProvider,
  id: string | null | undefined,
): Endpoint | undefined =>
  provider.endpoints.find((each) => each.id === id) ?? provider.endpoints[0];
