/**
 * The shapes of the provider catalog (see ./index): what IncarnaMind knows
 * about model providers and their models before it sends them anything.
 */

/** Text in the app's two languages. */
export interface Localized {
  en: string;
  zh: string;
}

/**
 * How a provider's requests are made: the AI SDK provider package, or the
 * app's own adapter for Ollama (see ../models). Today each adapter is also a
 * provider kind; providers that share one differ in their endpoints and quirks.
 */
export type Adapter =
  | "openai"
  | "anthropic"
  | "google"
  | "deepseek"
  | "alibaba"
  | "moonshotai"
  | "zai"
  | "mistral"
  | "xai"
  | "openrouter"
  | "openai-compatible"
  | "ollama";

/** Where a provider's API is, for one region. */
export interface Endpoint {
  /** Stable, stored on the provider row: "global", "cn", "intl". */
  id: string;
  label: Localized;
  /** The API's base URL, as the AI SDK provider takes it (e.g. "https://api.openai.com/v1"). */
  baseUrl: string;
  /** Where the User gets a key for this region, when it isn't the provider's `keyUrl` (keys are per region). */
  keyUrl?: string;
}

/**
 * The jobs a provider's models are picked for:
 * - "answers": the default for Answers, the provider's cheaper capable model;
 * - "strongest": its strongest model for Answers, one click away from the default;
 * - "quickTasks": a small, cheap model for the calls that don't write the Answer;
 * - "images": the model that reads images and scanned pages.
 */
export type Role = "answers" | "strongest" | "quickTasks" | "images";

/** What a provider does with the data it is sent through its API. */
export interface DataUse {
  /**
   * Whether it may train its models on API data: "no"; "opt-out" (it may,
   * unless the User turns it off); "free-tier" (on a free plan only);
   * "unknown" (it depends on the server, or its terms don't say).
   */
  training: "no" | "opt-out" | "free-tier" | "unknown";
  /** One plain line, shown where the User picks the provider. */
  summary: Localized;
  /** The provider's own terms, when there is one place for them. */
  url?: string;
}

/** What every request to a provider carries, whatever the model. */
export interface RequestQuirks {
  /** Ask the provider not to keep requests (the Responses API of OpenAI and xAI keeps them otherwise). */
  store?: false;
  /**
   * For an OpenAI-compatible adapter: ask for token usage in streamed replies
   * (`stream_options.include_usage`), so that costs can be counted, and send a
   * JSON schema as a schema (`json_schema`) rather than as JSON mode.
   */
  openAiCompatible?: { includeUsage: boolean; supportsStructuredOutputs: boolean };
  /** Fields added to the body of every request (OpenRouter: do not route to providers that keep or train on prompts). */
  body?: Record<string, unknown>;
  /**
   * Provider options (by the AI SDK provider's key) that turn thinking off, for
   * the small calls that don't write the Answer. A model whose id starts with
   * one of `unless` can't turn it off.
   */
  thinkingOff?: { options: Record<string, Record<string, unknown>>; unless?: readonly string[] };
  /**
   * Provider options that ask for the provider's summary of a model's
   * reasoning, for the calls that write the Answer (#391). Only providers
   * whose reasoning comes back summarized have them: other providers' is raw
   * chain of thought, which is never kept. They never turn thinking on: with
   * `only`, they go to the models whose id starts with one of its prefixes,
   * the ones that think by default.
   */
  reasoningSummary?: { options: Record<string, Record<string, unknown>>; only?: readonly string[] };
}

/**
 * How far IncarnaMind stands behind a provider (ADR-0019). A first-class one is
 * in the first-run setup, has a key guide where it needs a key, and is tried by
 * hand before each release. A best-effort one stays in the catalog and keeps
 * working for Users who have a key, listed under "More providers", with a link
 * to the provider's key page and no more.
 */
export type Support = "first-class" | "best-effort";

export interface CatalogProvider {
  /** Stable, stored on the provider row (`chat_providers.catalog_id`). */
  id: string;
  name: Localized;
  support: Support;
  group: "international" | "china" | "local" | "other";
  adapter: Adapter;
  /**
   * Its API, one per region, the first the default. Empty for a server the
   * User gives the address of (an OpenAI-compatible server, Ollama).
   */
  endpoints: readonly Endpoint[];
  /** For a provider without endpoints: whether the User must give the server's URL. */
  serverUrl?: "required" | "optional";
  apiKey: "required" | "optional" | "none";
  /** Where the User gets a key ("Get a key ↗"). */
  keyUrl?: string;
  /**
   * "How do I get one?": the steps to a key, in plain words, for a first-class
   * provider that needs one. The provider's own page (`keyUrl`) has the screens.
   */
  keyGuide?: readonly Localized[];
  docsUrl?: string;
  request?: RequestQuirks;
  /** Model ids for the roles it offers. A provider whose models the User names has none. */
  roles: Partial<Record<Role, string>>;
  dataUse: DataUse;
  /** Short notes shown under the key field. */
  notes?: readonly Localized[];
  /**
   * Which of the provider's subscription plans include credits a key can
   * spend, in the provider's own words, and the page that says so: shown
   * under the key field. Left out where no plan does.
   */
  planCredits?: Localized & { url: string };
  /** When a person last checked this entry against the provider's own pages (YYYY-MM-DD). */
  checked: string;
}

/** The kinds of input a model reads. */
export type ModelInput = "text" | "image" | "pdf" | "audio" | "video";

/**
 * What a schema request may use: strict JSON schema, JSON mode (the schema
 * only in the prompt), or nothing.
 */
export type StructuredOutput = "json_schema" | "json_object" | "none";

/** How IncarnaMind gives Citations with a model: in the Tool loop, with structured output, or not at all. */
export type Citing = "tools" | "structured-output" | "none";

/** A price per million tokens, in the provider's own currency. */
export interface Price {
  currency: "USD" | "CNY";
  input: number;
  output: number;
  cacheRead?: number;
  /** E.g. "Prompts over 100K tokens: $0.50 / $2.50". */
  note?: Localized;
  /** Outside its peak hours the price is cut, by `factor`. */
  offPeak?: OffPeak;
}

/**
 * The hours a price is full, on the provider's clock (`utcOffset` hours from
 * UTC), as [from, to) pairs of hours. Weekends are off-peak all day, unless
 * the peak hours hold `everyDay`.
 */
export interface OffPeak {
  factor: number;
  utcOffset: number;
  peakHours: readonly (readonly [number, number])[];
  everyDay?: true;
}

/** What a model can do and what it costs. */
export interface ModelFacts {
  input: ModelInput[];
  /** It can call Tools. */
  tools: boolean;
  structuredOutput: StructuredOutput;
  /** It can reason before it answers. */
  reasoning: boolean;
  /** It takes a temperature (false: send none, it rejects one or should run at its default). */
  temperature: boolean;
  /** It accepts a forced Tool choice ("required" or a named Tool). */
  forcedToolChoice: boolean;
  /** Its context window, in tokens, input and output together. */
  context: number;
  /** The most a request may send, when the provider limits input below the window. */
  maxInput: number;
  /** The most it writes in one reply, in tokens. */
  maxOutput: number;
  price: Price;
  /** How it cites, when that was measured or the server decides it (a small local model). */
  citing: Citing;
}

/** A model in the catalog: its facts, any of which its sources may not give. */
export interface CatalogModel extends Partial<ModelFacts> {
  /** The id its provider's API takes. */
  id: string;
  name: string;
  status?: "preview" | "deprecated";
}

/** The generated model facts of one provider (models.json). */
export interface GeneratedProviderModels {
  models: CatalogModel[];
  /** Ids a source says answer by another API than chat: completions, or in audio. */
  cannotAnswer: Record<string, "completion" | "audio">;
  /** Ids the provider lists that aren't chat models (embeddings, speech, images…): never offered. */
  nonChat: string[];
}

/** Hand-written corrections to the generated facts (see ./overrides). */
export interface CatalogOverrides {
  /** Facts for every model of a provider, below a model's own overrides. */
  providers: Record<string, Partial<ModelFacts>>;
  /** Facts for one model, by provider and model id. */
  models: Record<string, Record<string, Partial<ModelFacts>>>;
  /** Ids that need another API than chat, by provider, where no source says so. */
  cannotAnswer: Record<string, Record<string, "deep-research" | "computer-use">>;
}

/** A model recommended for Ollama on this computer, with what the server doesn't report reliably. */
export interface LocalModel {
  /** Its Ollama tag, e.g. "qwen3.5:4b". */
  tag: string;
  /** Total parameters, in billions; for a mixture of experts, `activeParameters` are those used per token. */
  parameters: number;
  activeParameters?: number;
  /** The download, in GB: the GGUF build (llama.cpp), and the MLX build on Apple silicon where there is one. */
  download: { gguf: number; mlx?: number };
  /** The context it was trained for, in tokens. */
  context: number;
  capabilities: readonly ("tools" | "thinking" | "vision" | "audio")[];
  languages: readonly ("en" | "zh")[];
  licence: string;
  /** When IncarnaMind's evaluation measured it, e.g. "#67, 2026-10-10"; absent: not tested with IncarnaMind. */
  evaluated?: string;
}
