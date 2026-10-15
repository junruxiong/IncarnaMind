/**
 * Hosted services that speak OpenAI's API, one catalog row each (#863). They
 * share the "openai-compatible" kind and its code, told apart by their
 * catalog id, so adding one is a row here and nothing else.
 */
import type { CatalogProvider } from "./types";

export const PRESETS: readonly CatalogProvider[] = [
  {
    id: "groq",
    name: { en: "Groq", zh: "Groq" },
    support: "best-effort",
    group: "international",
    adapter: "openai-compatible",
    endpoints: [
      {
        id: "global",
        label: { en: "Global", zh: "全球" },
        baseUrl: "https://api.groq.com/openai/v1",
      },
    ],
    apiKey: "required",
    keyUrl: "https://console.groq.com/keys",
    docsUrl: "https://console.groq.com/docs/models",
    request: { openAiCompatible: { includeUsage: true, supportsStructuredOutputs: false } },
    roles: { answers: "openai/gpt-oss-120b", quickTasks: "openai/gpt-oss-20b" },
    dataUse: {
      training: "unknown",
      summary: {
        en: "Groq doesn't keep API requests by default. It may log some for up to 30 days to look into errors or abuse, which you can turn off.",
        zh: "Groq 默认不保留 API 请求。为排查错误或滥用，它可能记录部分请求最多 30 天，你可以关闭这项记录。",
      },
      url: "https://console.groq.com/docs/your-data",
    },
    checked: "2026-10-14",
  },
  {
    id: "together",
    name: { en: "Together AI", zh: "Together AI" },
    support: "best-effort",
    group: "international",
    adapter: "openai-compatible",
    endpoints: [
      { id: "global", label: { en: "Global", zh: "全球" }, baseUrl: "https://api.together.ai/v1" },
    ],
    apiKey: "required",
    keyUrl: "https://api.together.ai/settings/projects/~current/api-keys",
    docsUrl: "https://docs.together.ai/docs/serverless-models",
    request: { openAiCompatible: { includeUsage: true, supportsStructuredOutputs: false } },
    roles: {
      answers: "deepseek-ai/DeepSeek-V4-Flash-0731",
      strongest: "deepseek-ai/DeepSeek-V4-Pro-0813",
      quickTasks: "Qwen/Qwen3.5-9B",
    },
    dataUse: {
      training: "no",
      summary: {
        en: "Together doesn't train on your data unless you opt in. With Zero Data Retention on, what you send is removed once it's processed.",
        zh: "除非你同意，Together 不会用你的数据训练模型。开启零数据保留后，请求处理完即删除。",
      },
      url: "https://www.together.ai/privacy",
    },
    checked: "2026-10-15",
  },
  {
    id: "fireworks",
    name: { en: "Fireworks AI", zh: "Fireworks AI" },
    support: "best-effort",
    group: "international",
    adapter: "openai-compatible",
    endpoints: [
      {
        id: "global",
        label: { en: "Global", zh: "全球" },
        baseUrl: "https://api.fireworks.ai/inference/v1",
      },
    ],
    apiKey: "required",
    keyUrl: "https://app.fireworks.ai/settings/users/api-keys",
    docsUrl: "https://docs.fireworks.ai/getting-started/quickstart",
    request: { openAiCompatible: { includeUsage: true, supportsStructuredOutputs: false } },
    roles: {
      answers: "accounts/fireworks/models/glm-5p3-flash",
      strongest: "accounts/fireworks/models/glm-5p3",
      quickTasks: "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
    },
    dataUse: {
      training: "unknown",
      summary: {
        en: "Fireworks doesn't log or store prompts or replies for open models unless you opt in.",
        zh: "除非你同意，Fireworks 不会记录或保存开放模型的提示和回复。",
      },
      url: "https://docs.fireworks.ai/guides/security_compliance/data_handling",
    },
    checked: "2026-10-15",
  },
  {
    id: "cerebras",
    name: { en: "Cerebras", zh: "Cerebras" },
    support: "best-effort",
    group: "international",
    adapter: "openai-compatible",
    endpoints: [
      { id: "global", label: { en: "Global", zh: "全球" }, baseUrl: "https://api.cerebras.ai/v1" },
    ],
    apiKey: "required",
    keyUrl: "https://cloud.cerebras.ai",
    docsUrl: "https://inference-docs.cerebras.ai/models/overview",
    request: { openAiCompatible: { includeUsage: true, supportsStructuredOutputs: false } },
    roles: { answers: "gpt-oss-120b", quickTasks: "gpt-oss-120b" },
    dataUse: {
      training: "unknown",
      summary: {
        en: "Cerebras says it doesn't keep the inputs and outputs of its inference service.",
        zh: "Cerebras 表示不保留推理服务的输入和输出。",
      },
      url: "https://www.cerebras.ai/privacy-policy",
    },
    checked: "2026-10-15",
  },
  {
    id: "cohere",
    name: { en: "Cohere", zh: "Cohere" },
    support: "best-effort",
    group: "international",
    adapter: "openai-compatible",
    endpoints: [
      {
        id: "global",
        label: { en: "Global", zh: "全球" },
        baseUrl: "https://api.cohere.ai/compatibility/v1",
      },
    ],
    apiKey: "required",
    keyUrl: "https://dashboard.cohere.com/api-keys",
    docsUrl: "https://docs.cohere.com/docs/compatibility-api",
    // Its compatibility API doesn't document `stream_options`, so streamed usage isn't asked for.
    request: { openAiCompatible: { includeUsage: false, supportsStructuredOutputs: false } },
    roles: {
      answers: "command-r-08-2024",
      strongest: "command-a-plus-05-2026",
      quickTasks: "command-r7b-12-2024",
    },
    dataUse: {
      training: "opt-out",
      summary: {
        en: "Cohere may train on prompts and replies unless you turn that off in Data Controls. It deletes logged ones after 30 days.",
        zh: "除非你在“Data Controls”中关闭，Cohere 可能用提示和回复训练模型。记录的内容 30 天后删除。",
      },
      url: "https://cohere.com/data-usage-policy",
    },
    checked: "2026-10-15",
  },
  {
    id: "venice",
    name: { en: "Venice", zh: "Venice" },
    support: "best-effort",
    group: "international",
    adapter: "openai-compatible",
    endpoints: [
      {
        id: "global",
        label: { en: "Global", zh: "全球" },
        baseUrl: "https://api.venice.ai/api/v1",
      },
    ],
    apiKey: "required",
    keyUrl: "https://venice.ai/settings/api",
    docsUrl: "https://docs.venice.ai/overview/getting-started",
    request: { openAiCompatible: { includeUsage: true, supportsStructuredOutputs: false } },
    roles: { answers: "deepseek-v4-flash-0731", quickTasks: "qwen3-5-9b" },
    dataUse: {
      training: "unknown",
      summary: {
        en: "Venice doesn't store or log what you send or what it replies.",
        zh: "Venice 不保存也不记录你发送的内容和它的回复。",
      },
      url: "https://docs.venice.ai/overview/privacy",
    },
    checked: "2026-10-15",
  },
];
