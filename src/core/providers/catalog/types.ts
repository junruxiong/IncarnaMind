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
export type Adapter = "openai" | "anthropic" | "google" | "openai-compatible" | "ollama";

/** Where a provider's API is, for one region. */
export interface Endpoint {
  /** Stable, stored on the provider row: "global", "cn", "intl". */
  id: string;
  label: Localized;
  /** The API's base URL, as the AI SDK provider takes it (e.g. "https://api.openai.com/v1"). */
  baseUrl: string;
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
  /** Ask the provider not to keep requests (OpenAI's Responses API keeps them otherwise). */
  store?: false;
}

export interface CatalogProvider {
  /** Stable, stored on the provider row (`chat_providers.catalog_id`). */
  id: string;
  name: Localized;
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
  docsUrl?: string;
  request?: RequestQuirks;
  /** Model ids for the roles it offers. A provider whose models the User names has none. */
  roles: Partial<Record<Role, string>>;
  dataUse: DataUse;
  /** Short notes shown under the key field. */
  notes?: readonly Localized[];
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
  /** Ids the provider lists that aren't chat models (embeddings, speech, images…): never offered. */
  nonChat: string[];
}

/** Hand-written corrections to the generated facts (see ./overrides). */
export interface CatalogOverrides {
  /** Facts for every model of a provider, below a model's own overrides. */
  providers: Record<string, Partial<ModelFacts>>;
  /** Facts for one model, by provider and model id. */
  models: Record<string, Record<string, Partial<ModelFacts>>>;
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
