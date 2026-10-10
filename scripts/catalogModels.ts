/**
 * Writes src/core/providers/catalog/models.json, the catalog's model facts,
 * from models.dev's api.json (MIT), cross-checked against LiteLLM's
 * model_prices_and_context_window.json (MIT): `node scripts/catalogModels.ts`.
 * It runs on a maintainer's computer, never in the app, and the app reads
 * only the file it writes. With `--models-dev <file>` and `--litellm <file>`
 * it reads copies instead of fetching them.
 *
 * For each catalog provider it keeps models.dev's chat models (text in, text
 * out, a context window, and not something else by LiteLLM's mode) with
 * their facts, and lists the ids either source knows as something else
 * (embeddings, speech, images, video, realtime), which the app never offers.
 * Where LiteLLM disagrees on Tools, images, JSON schema or limits, it prints
 * the difference and changes nothing: ./src/core/providers/catalog/overrides.ts
 * is where a person corrects a fact.
 *
 * Both sources' licences ask for their notice to go with the data: it is
 * written to resources/notices/model-catalog.txt, which ships in the app's
 * resources folder (electron-builder.yml).
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type {
  CatalogModel,
  GeneratedProviderModels,
  ModelInput,
} from "../src/core/providers/catalog/types";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT = join(root, "src/core/providers/catalog/models.json");
const NOTICE_FILE = join(root, "resources/notices/model-catalog.txt");

const MODELS_DEV_URL = "https://models.dev/api.json";
const LITELLM_URL =
  "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

/** Each catalog provider's id at models.dev and at LiteLLM (`litellm_provider`). */
const SOURCES: Record<string, { modelsDev: string; litellm: string }> = {
  anthropic: { modelsDev: "anthropic", litellm: "anthropic" },
  openai: { modelsDev: "openai", litellm: "openai" },
  google: { modelsDev: "google", litellm: "gemini" },
};

const MIT = (copyright: string) =>
  [
    "MIT License",
    "",
    copyright,
    "",
    'Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:',
    "",
    "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.",
    "",
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.',
  ].join("\n");

/** The notice that goes with the generated facts. */
const NOTICE = [
  "IncarnaMind's model catalog (src/core/providers/catalog/models.json) is generated from",
  "models.dev and checked against LiteLLM's model list, under these licences. The rest of",
  "IncarnaMind is under its own licence (Apache-2.0).",
  "",
  "models.dev: https://github.com/anomalyco/models.dev",
  "",
  MIT("Copyright (c) 2025 models.dev"),
  "",
  "LiteLLM (model_prices_and_context_window.json): https://github.com/BerriAI/litellm",
  "",
  MIT("Copyright (c) 2023 Berri AI"),
  "",
].join("\n");

interface ModelsDevModel {
  name?: string;
  reasoning?: boolean;
  tool_call?: boolean;
  structured_output?: boolean;
  temperature?: boolean;
  modalities?: { input?: string[]; output?: string[] };
  limit?: { context?: number; input?: number; output?: number };
  cost?: { input?: number; output?: number; cache_read?: number };
  status?: string;
}

interface LiteLlmEntry {
  litellm_provider?: string;
  mode?: string;
  max_input_tokens?: number;
  max_output_tokens?: number;
  supports_function_calling?: boolean;
  supports_vision?: boolean;
  supports_response_schema?: boolean;
}

const INPUTS: readonly ModelInput[] = ["text", "image", "pdf", "audio", "video"];
const CHAT_MODES = new Set(["chat", "responses"]);

/** LiteLLM's entries of one provider, by model id (its key without a provider or size prefix). */
function litellmEntries(all: Record<string, LiteLlmEntry>, provider: string) {
  const entries = new Map<string, LiteLlmEntry>();
  for (const [key, entry] of Object.entries(all)) {
    if (entry.litellm_provider !== provider) continue;
    const id = key.split("/").at(-1) ?? key;
    // A key without a prefix wins over a prefixed one for the same id.
    if (!entries.has(id) || !key.includes("/")) entries.set(id, entry);
  }
  return entries;
}

/** Whether models.dev's entry is a chat model: text in, text only out, a context window. */
function readsAndWritesText(model: ModelsDevModel): boolean {
  const input = model.modalities?.input ?? [];
  const output = model.modalities?.output ?? [];
  return (
    input.includes("text") &&
    output.length === 1 &&
    output[0] === "text" &&
    (model.limit?.context ?? 0) > 0
  );
}

function factsOf(id: string, model: ModelsDevModel): CatalogModel {
  const { context = 0, input: maxInput, output: maxOutput } = model.limit ?? {};
  const cost = model.cost;
  const status =
    model.status === "deprecated"
      ? "deprecated"
      : model.status === "alpha" || model.status === "beta"
        ? "preview"
        : undefined;
  return {
    id,
    name: model.name ?? id,
    input: INPUTS.filter((each) => model.modalities?.input?.includes(each)),
    tools: model.tool_call === true,
    ...(model.structured_output !== undefined && {
      structuredOutput: model.structured_output ? ("json_schema" as const) : ("none" as const),
    }),
    reasoning: model.reasoning === true,
    ...(model.temperature !== undefined && { temperature: model.temperature }),
    context,
    // models.dev gives an input limit where the provider has one below the window.
    ...(maxInput !== undefined && maxInput > 0 && maxInput < context && { maxInput }),
    ...(maxOutput !== undefined && maxOutput > 0 && { maxOutput }),
    ...(cost?.input !== undefined &&
      cost.output !== undefined && {
        price: {
          currency: "USD" as const,
          input: cost.input,
          output: cost.output,
          ...(cost.cache_read !== undefined && { cacheRead: cost.cache_read }),
        },
      }),
    ...(status && { status }),
  };
}

/** Where LiteLLM says otherwise than the facts, as lines to print. */
function disagreements(provider: string, model: CatalogModel, entry: LiteLlmEntry): string[] {
  const lines: string[] = [];
  const differ = (what: string, ours: unknown, theirs: unknown) => {
    if (theirs !== undefined && ours !== theirs) {
      lines.push(`${provider}/${model.id}: ${what} is ${ours} here, ${theirs} in LiteLLM`);
    }
  };
  differ("Tools", model.tools, entry.supports_function_calling);
  differ("reading images", model.input?.includes("image"), entry.supports_vision);
  if (model.structuredOutput !== undefined) {
    differ("JSON schema", model.structuredOutput === "json_schema", entry.supports_response_schema);
  }
  differ("the input limit", model.maxInput ?? model.context, entry.max_input_tokens);
  differ("the output limit", model.maxOutput, entry.max_output_tokens);
  return lines;
}

/** The generated facts of every catalog provider, and the disagreements found. */
export function generateModels(
  modelsDev: Record<string, { models?: Record<string, ModelsDevModel> }>,
  litellm: Record<string, LiteLlmEntry>,
): { providers: Record<string, GeneratedProviderModels>; disagreements: string[] } {
  const providers: Record<string, GeneratedProviderModels> = {};
  const found: string[] = [];
  for (const [id, source] of Object.entries(SOURCES)) {
    const entries = litellmEntries(litellm, source.litellm);
    const models: CatalogModel[] = [];
    const nonChat = new Set<string>();
    for (const [modelId, model] of Object.entries(modelsDev[source.modelsDev]?.models ?? {})) {
      const mode = entries.get(modelId)?.mode;
      if (!readsAndWritesText(model) || (mode !== undefined && !CHAT_MODES.has(mode))) {
        nonChat.add(modelId);
        continue;
      }
      const facts = factsOf(modelId, model);
      models.push(facts);
      const entry = entries.get(modelId);
      if (entry) found.push(...disagreements(id, facts, entry));
    }
    const chat = new Set(models.map((model) => model.id));
    for (const [modelId, entry] of entries) {
      if (entry.mode && !CHAT_MODES.has(entry.mode) && !chat.has(modelId)) nonChat.add(modelId);
    }
    providers[id] = {
      models: models.sort((a, b) => a.id.localeCompare(b.id)),
      nonChat: [...nonChat].sort((a, b) => a.localeCompare(b)),
    };
  }
  return { providers, disagreements: found };
}

async function load(flag: string, url: string): Promise<unknown> {
  const at = process.argv.indexOf(flag);
  const file = at === -1 ? undefined : process.argv[at + 1];
  if (file) return JSON.parse(readFileSync(file, "utf8"));
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const modelsDev = (await load("--models-dev", MODELS_DEV_URL)) as Parameters<
    typeof generateModels
  >[0];
  const litellm = (await load("--litellm", LITELLM_URL)) as Record<string, LiteLlmEntry>;
  const { providers, disagreements: found } = generateModels(modelsDev, litellm);
  writeFileSync(OUTPUT, `${JSON.stringify({ providers }, null, 2)}\n`);
  execFileSync("npx", ["biome", "format", "--write", OUTPUT], { cwd: root, stdio: "inherit" });
  mkdirSync(dirname(NOTICE_FILE), { recursive: true });
  writeFileSync(NOTICE_FILE, NOTICE);
  for (const [id, { models, nonChat }] of Object.entries(providers)) {
    console.log(`${id}: ${models.length} chat models, ${nonChat.length} others`);
  }
  if (found.length > 0) {
    console.log("\nLiteLLM disagrees (nothing changed; correct a fact in overrides.ts):");
    for (const line of found) console.log(`  ${line}`);
  }
}
