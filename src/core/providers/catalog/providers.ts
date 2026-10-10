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
    id: "openrouter",
    name: { en: "OpenRouter", zh: "OpenRouter" },
    group: "international",
    adapter: "openrouter",
    endpoints: [{ id: "global", label: GLOBAL, baseUrl: "https://openrouter.ai/api/v1" }],
    apiKey: "required",
    keyUrl: "https://openrouter.ai/settings/keys",
    docsUrl: "https://openrouter.ai/models",
    request: {
      // Route only to providers that don't keep or train on prompts, and only to those that
      // support everything a request asks for (Tools, a JSON schema).
      body: { provider: { data_collection: "deny", require_parameters: true } },
      thinkingOff: { options: { openrouter: { reasoning: { effort: "none" } } } },
    },
    roles: {
      answers: "anthropic/claude-sonnet-5.5",
      strongest: "anthropic/claude-fable-5.1",
      quickTasks: "openai/gpt-6-luna",
      images: "anthropic/claude-sonnet-5.5",
    },
    dataUse: {
      training: "no",
      summary: {
        en: "One key for many models. IncarnaMind asks OpenRouter to use only providers that don't keep or train on your prompts; that leaves out some models.",
        zh: "一个密钥可用多种模型。IncarnaMind 会要求 OpenRouter 只使用不保存、不用于训练的提供方，因此部分模型无法使用。",
      },
      url: "https://openrouter.ai/docs/guides/privacy/logging",
    },
    notes: [
      {
        en: "OpenRouter adds a fee of 5.5% to credits bought by card.",
        zh: "OpenRouter 对刷卡充值的额度收取 5.5% 的手续费。",
      },
    ],
    checked: "2026-10-10",
  },
  {
    id: "mistral",
    name: { en: "Mistral", zh: "Mistral" },
    group: "international",
    adapter: "mistral",
    endpoints: [{ id: "global", label: GLOBAL, baseUrl: "https://api.mistral.ai/v1" }],
    apiKey: "required",
    keyUrl: "https://console.mistral.ai/api-keys",
    docsUrl: "https://docs.mistral.ai/getting-started/models/models_overview/",
    roles: {
      answers: "mistral-small-2603",
      strongest: "mistral-medium-2604",
      quickTasks: "ministral-8b-2512",
      images: "mistral-small-2603",
    },
    dataUse: {
      training: "opt-out",
      summary: {
        en: "In free mode Mistral may train on API data unless you turn that off in its privacy settings; on pay-as-you-go you can opt out too. Preview models may be trained on regardless.",
        zh: "免费模式下，Mistral 可能用 API 数据训练模型，除非你在其隐私设置中关闭；按量付费同样可以选择退出。预览版模型无论如何都可能被用于训练。",
      },
      url: "https://help.mistral.ai/en/articles/347617-do-you-use-my-user-data-to-train-your-artificial-intelligence-models",
    },
    notes: [
      {
        en: "Mistral Large 4 is a preview: pick it only if its terms suit you.",
        zh: "Mistral Large 4 为预览版：请确认其条款适合你再选用。",
      },
    ],
    checked: "2026-10-10",
  },
  {
    id: "xai",
    name: { en: "xAI", zh: "xAI" },
    group: "international",
    adapter: "xai",
    endpoints: [{ id: "global", label: GLOBAL, baseUrl: "https://api.x.ai/v1" }],
    apiKey: "required",
    keyUrl: "https://console.x.ai",
    docsUrl: "https://docs.x.ai/docs/models",
    // The Responses API keeps each request unless it says not to.
    request: {
      store: false,
      thinkingOff: { options: { xai: { reasoningEffort: "none" } } },
    },
    roles: {
      answers: "grok-4.3",
      strongest: "grok-4.7",
      quickTasks: "grok-4.3",
      images: "grok-4.3",
    },
    dataUse: {
      training: "no",
      summary: {
        en: "Doesn't train on API data without your permission, and keeps it for up to 30 days. IncarnaMind asks it not to store requests.",
        zh: "未经你许可不用 API 数据训练模型，数据最多保留 30 天。IncarnaMind 会要求它不保存请求。",
      },
      url: "https://x.ai/legal/faq-enterprise",
    },
    checked: "2026-10-10",
  },
  {
    id: "deepseek",
    name: { en: "DeepSeek", zh: "DeepSeek 深度求索" },
    group: "china",
    adapter: "deepseek",
    endpoints: [{ id: "global", label: GLOBAL, baseUrl: "https://api.deepseek.com" }],
    apiKey: "required",
    keyUrl: "https://platform.deepseek.com/api_keys",
    docsUrl: "https://api-docs.deepseek.com/quick_start/pricing",
    request: { thinkingOff: { options: { deepseek: { thinking: { type: "disabled" } } } } },
    roles: {
      answers: "deepseek-flash",
      strongest: "deepseek-v4-pro",
      quickTasks: "deepseek-flash",
      images: "deepseek-flash",
    },
    dataUse: {
      training: "opt-out",
      summary: {
        en: "DeepSeek's privacy policy lets it use personal data to train and improve its technology, with a right to opt out. Data is stored in China.",
        zh: "DeepSeek 的隐私政策允许其使用个人数据训练和改进技术，用户有权选择退出。数据存储在中国。",
      },
      url: "https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy.html",
    },
    notes: [
      {
        en: "Weekdays 09:00–12:00 and 14:00–18:00 Beijing time cost double.",
        zh: "北京时间工作日 9:00–12:00 和 14:00–18:00 为高峰时段，价格翻倍。",
      },
    ],
    checked: "2026-10-10",
  },
  {
    id: "qwen",
    name: { en: "Qwen (Alibaba Cloud Model Studio)", zh: "通义千问（阿里云百炼）" },
    group: "china",
    adapter: "alibaba",
    // Each region has its own key and model list: a key works only in the region it was made in.
    endpoints: [
      {
        id: "cn",
        label: { en: "China (Beijing)", zh: "中国内地（北京）" },
        baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        keyUrl: "https://bailian.console.aliyun.com/?tab=model#/api-key",
      },
      {
        id: "intl",
        label: { en: "International (Singapore)", zh: "国际（新加坡）" },
        baseUrl: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        keyUrl: "https://modelstudio.console.alibabacloud.com/?tab=playground#/api-key",
      },
    ],
    apiKey: "required",
    keyUrl: "https://bailian.console.aliyun.com/?tab=model#/api-key",
    docsUrl: "https://www.alibabacloud.com/help/en/model-studio/regions",
    request: { thinkingOff: { options: { alibaba: { enableThinking: false } } } },
    roles: {
      answers: "qwen3.7-plus",
      strongest: "qwen3.8-max",
      quickTasks: "qwen3.7-flash",
      images: "qwen3.7-plus",
    },
    dataUse: {
      training: "no",
      summary: {
        en: "Alibaba Cloud says it will never use your data for model training. Data is stored in the region you choose.",
        zh: "阿里云承诺绝不会将你的数据用于模型训练。数据存储在你所选的地域。",
      },
      url: "https://www.alibabacloud.com/help/en/model-studio/",
    },
    notes: [
      {
        en: "A key works only in the region it was made in. Choose the region first.",
        zh: "密钥只能用于创建它的地域。请先选择地域。",
      },
    ],
    checked: "2026-10-10",
  },
  {
    id: "kimi",
    name: { en: "Kimi (Moonshot AI)", zh: "Kimi（月之暗面）" },
    group: "china",
    adapter: "moonshotai",
    endpoints: [
      {
        id: "cn",
        label: { en: "China", zh: "中国" },
        baseUrl: "https://api.moonshot.cn/v1",
        keyUrl: "https://platform.kimi.com/console/api-keys",
      },
      {
        id: "intl",
        label: { en: "International", zh: "国际" },
        baseUrl: "https://api.moonshot.ai/v1",
        keyUrl: "https://platform.kimi.ai/console/api-keys",
      },
    ],
    apiKey: "required",
    keyUrl: "https://platform.kimi.com/console/api-keys",
    docsUrl: "https://platform.kimi.ai/docs/pricing/chat",
    request: { thinkingOff: { options: { moonshotai: { thinking: { type: "disabled" } } } } },
    roles: {
      answers: "kimi-k2.6",
      strongest: "kimi-k3",
      quickTasks: "kimi-k2.6",
      images: "kimi-k2.6",
    },
    dataUse: {
      training: "unknown",
      summary: {
        en: "Moonshot's pages disagree: its help page says API data isn't used to train Kimi, but its international terms let it use content to improve the service unless agreed otherwise.",
        zh: "月之暗面的说明并不一致：帮助页面称不会用 API 数据训练 Kimi，但其国际版条款允许在未另行约定时使用内容改进服务。",
      },
      url: "https://platform.kimi.ai/docs/terms",
    },
    notes: [
      {
        en: "Until you top up, Kimi allows 3 requests a minute; a top-up of ¥50 lifts that to 100.",
        zh: "充值前每分钟只允许 3 次请求；充值 ¥50 后提高到 100 次。",
      },
    ],
    checked: "2026-10-10",
  },
  {
    id: "glm",
    name: { en: "GLM (Zhipu / Z.ai)", zh: "智谱 GLM" },
    group: "china",
    adapter: "zai",
    // BigModel (mainland China) and Z.ai (international) are separate platforms with separate keys.
    // Not the Coding Plan's endpoint: its terms forbid use in other apps.
    endpoints: [
      {
        id: "cn",
        label: { en: "China (BigModel)", zh: "中国（智谱开放平台）" },
        baseUrl: "https://open.bigmodel.cn/api/paas/v4",
        keyUrl: "https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
      },
      {
        id: "intl",
        label: { en: "International (Z.ai)", zh: "国际（Z.ai）" },
        baseUrl: "https://api.z.ai/api/paas/v4",
        keyUrl: "https://z.ai/manage-apikey/apikey-list",
      },
    ],
    apiKey: "required",
    keyUrl: "https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
    docsUrl: "https://docs.bigmodel.cn/cn/guide/start/pricing",
    request: {
      thinkingOff: {
        options: { zai: { thinking: { type: "disabled" } } },
        // GLM-5.3 and its Flash can't turn thinking off.
        unless: ["glm-5.3"],
      },
    },
    roles: {
      answers: "glm-5.3-flash",
      strongest: "glm-5.3",
      quickTasks: "glm-4.7-flash",
      images: "glm-5.3-flash",
    },
    dataUse: {
      training: "unknown",
      summary: {
        en: "Z.ai says it won't use your content unless you agree. BigModel (China) may train on anonymised data. Which applies depends on the region you choose.",
        zh: "Z.ai 表示未经你同意不会使用你的内容；智谱开放平台（中国）可能使用匿名化数据训练。具体取决于你选择的地域。",
      },
      url: "https://docs.z.ai/legal-agreement/privacy-policy",
    },
    notes: [
      {
        en: "BigModel asks for real-name verification before it gives a key.",
        zh: "智谱开放平台在发放密钥前需要实名认证。",
      },
    ],
    checked: "2026-10-10",
  },
  {
    id: "siliconflow",
    name: { en: "SiliconFlow", zh: "硅基流动" },
    group: "china",
    adapter: "openai-compatible",
    endpoints: [
      {
        id: "cn",
        label: { en: "China", zh: "中国" },
        baseUrl: "https://api.siliconflow.cn/v1",
        keyUrl: "https://cloud.siliconflow.cn/account/ak",
      },
      {
        id: "intl",
        label: { en: "International", zh: "国际" },
        baseUrl: "https://api.siliconflow.com/v1",
        keyUrl: "https://cloud.siliconflow.com/account/ak",
      },
    ],
    apiKey: "required",
    keyUrl: "https://cloud.siliconflow.cn/account/ak",
    docsUrl: "https://docs.siliconflow.cn/cn/userguide/quickstart",
    request: {
      openAiCompatible: { includeUsage: true, supportsStructuredOutputs: true },
      thinkingOff: { options: { siliconflow: { enable_thinking: false } } },
    },
    roles: {
      answers: "deepseek-ai/DeepSeek-V4-Flash",
      strongest: "deepseek-ai/DeepSeek-V4-Pro",
      quickTasks: "Qwen/Qwen3-8B",
      images: "Qwen/Qwen3-VL-32B-Instruct",
    },
    dataUse: {
      training: "no",
      summary: {
        en: "SiliconFlow says it won't use your business data for pre-training or fine-tuning any model.",
        zh: "硅基流动表示不会将你的业务数据用于任何大模型的预训练或微调。",
      },
      url: "https://docs.siliconflow.cn/cn/legals/privacy-policy",
    },
    notes: [
      {
        en: "Many models with one key. Mainland accounts need ID and face verification.",
        zh: "一个密钥可用多种模型。中国大陆账号需要身份证和人脸验证。",
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
