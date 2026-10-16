/**
 * Services that speak OpenAI's API, hosted or on this computer, one catalog
 * row each. They share the "openai-compatible" kind and its code, told apart
 * by their catalog id, so adding one is a row here and nothing else.
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
  {
    id: "bedrock",
    name: { en: "Amazon Bedrock", zh: "Amazon Bedrock" },
    support: "best-effort",
    group: "international",
    adapter: "openai-compatible",
    endpoints: [
      {
        id: "global",
        label: { en: "Global", zh: "全球" },
        baseUrl: "https://bedrock-mantle.{region}.api.aws/v1",
        field: {
          name: "region",
          label: { en: "Region", zh: "区域" },
          hint: {
            en: "Use a long-term API key made in this region. A short-term key stops working within 12 hours.",
            zh: "请使用在这个区域创建的长期 API 密钥。短期密钥最长 12 小时后失效。",
          },
          choices: [
            {
              value: "us-east-1",
              label: { en: "US East (N. Virginia)", zh: "美国东部（弗吉尼亚北部）" },
            },
            { value: "us-east-2", label: { en: "US East (Ohio)", zh: "美国东部（俄亥俄）" } },
            { value: "us-west-2", label: { en: "US West (Oregon)", zh: "美国西部（俄勒冈）" } },
            { value: "eu-central-1", label: { en: "Europe (Frankfurt)", zh: "欧洲（法兰克福）" } },
            { value: "eu-north-1", label: { en: "Europe (Stockholm)", zh: "欧洲（斯德哥尔摩）" } },
            { value: "eu-south-1", label: { en: "Europe (Milan)", zh: "欧洲（米兰）" } },
            { value: "eu-west-1", label: { en: "Europe (Ireland)", zh: "欧洲（爱尔兰）" } },
            { value: "eu-west-2", label: { en: "Europe (London)", zh: "欧洲（伦敦）" } },
            {
              value: "ap-northeast-1",
              label: { en: "Asia Pacific (Tokyo)", zh: "亚太地区（东京）" },
            },
            { value: "ap-south-1", label: { en: "Asia Pacific (Mumbai)", zh: "亚太地区（孟买）" } },
            {
              value: "ap-southeast-2",
              label: { en: "Asia Pacific (Sydney)", zh: "亚太地区（悉尼）" },
            },
            {
              value: "ap-southeast-3",
              label: { en: "Asia Pacific (Jakarta)", zh: "亚太地区（雅加达）" },
            },
            {
              value: "sa-east-1",
              label: { en: "South America (São Paulo)", zh: "南美洲（圣保罗）" },
            },
          ],
        },
      },
    ],
    apiKey: "required",
    keyUrl: "https://console.aws.amazon.com/bedrock/home#/api-keys",
    docsUrl: "https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html",
    request: { openAiCompatible: { includeUsage: false, supportsStructuredOutputs: false } },
    roles: { answers: "openai.gpt-oss-120b", quickTasks: "openai.gpt-oss-20b" },
    dataUse: {
      training: "no",
      summary: {
        en: "Amazon Bedrock doesn't use your prompts and replies to train models, and the models' makers can't see them.",
        zh: "Amazon Bedrock 不会用你的提示和回复训练模型，模型的开发方也看不到它们。",
      },
      url: "https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html",
    },
    checked: "2026-10-15",
  },
  {
    id: "cloudflare",
    name: { en: "Cloudflare Workers AI", zh: "Cloudflare Workers AI" },
    support: "best-effort",
    group: "international",
    adapter: "openai-compatible",
    endpoints: [
      {
        id: "global",
        label: { en: "Global", zh: "全球" },
        baseUrl: "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
        field: {
          name: "account",
          label: { en: "Account ID", zh: "账户 ID" },
          hint: {
            en: "On the Workers AI page of the Cloudflare dashboard, under Use REST API.",
            zh: "在 Cloudflare 控制台的 Workers AI 页面，“使用 REST API”下可以找到。",
          },
          pattern: "[0-9a-f]{32}",
        },
      },
    ],
    apiKey: "required",
    keyUrl: "https://dash.cloudflare.com/profile/api-tokens",
    docsUrl: "https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/",
    request: { openAiCompatible: { includeUsage: false, supportsStructuredOutputs: false } },
    roles: {
      answers: "@cf/zai-org/glm-5.3-flash",
      strongest: "@cf/zai-org/glm-5.3",
      quickTasks: "@cf/zai-org/glm-4.7-flash",
    },
    dataUse: {
      training: "no",
      summary: {
        en: "Cloudflare doesn't use what you send to Workers AI to train any model.",
        zh: "Cloudflare 不会用你发送给 Workers AI 的内容训练任何模型。",
      },
      url: "https://developers.cloudflare.com/workers-ai/platform/data-usage/",
    },
    checked: "2026-10-15",
  },
  {
    id: "lmstudio",
    name: { en: "LM Studio", zh: "LM Studio" },
    support: "best-effort",
    group: "local",
    adapter: "openai-compatible",
    endpoints: [
      {
        id: "local",
        label: { en: "On this computer", zh: "本机" },
        baseUrl: "http://{address}/v1",
        field: {
          name: "address",
          label: { en: "Address", zh: "地址" },
          pattern: "(?:[A-Za-z0-9.-]+|\\[[0-9A-Fa-f:]+\\])(?::[0-9]{1,5})?",
          default: "127.0.0.1:1234",
        },
      },
    ],
    // A token is needed only when the User turned on authentication in its server settings.
    apiKey: "optional",
    docsUrl: "https://lmstudio.ai/docs/developer/openai-compat",
    request: { openAiCompatible: { includeUsage: false, supportsStructuredOutputs: true } },
    roles: {},
    dataUse: {
      training: "no",
      summary: {
        en: "LM Studio runs on this computer, so what you ask stays on it.",
        zh: "LM Studio 在这台电脑上运行，你的提问不会离开这台电脑。",
      },
    },
    checked: "2026-10-15",
  },
  {
    id: "unsloth",
    name: { en: "Unsloth Studio", zh: "Unsloth Studio" },
    support: "best-effort",
    group: "local",
    adapter: "openai-compatible",
    endpoints: [
      {
        id: "local",
        label: { en: "On this computer", zh: "本机" },
        baseUrl: "http://{address}/v1",
        field: {
          name: "address",
          label: { en: "Address", zh: "地址" },
          pattern: "(?:[A-Za-z0-9.-]+|\\[[0-9A-Fa-f:]+\\])(?::[0-9]{1,5})?",
          default: "127.0.0.1:8888",
        },
      },
    ],
    apiKey: "required",
    keyUrl: "https://unsloth.ai/docs/basics/api",
    docsUrl: "https://unsloth.ai/docs/basics/api",
    request: { openAiCompatible: { includeUsage: false, supportsStructuredOutputs: false } },
    roles: {},
    dataUse: {
      training: "no",
      summary: {
        en: "Unsloth Studio runs on this computer, so what you ask stays on it.",
        zh: "Unsloth Studio 在这台电脑上运行，你的提问不会离开这台电脑。",
      },
    },
    checked: "2026-10-15",
  },
];
