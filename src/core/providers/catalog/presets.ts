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
];
