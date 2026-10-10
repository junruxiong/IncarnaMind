/**
 * The classifiers Organize can use, built exactly as the core builds them for
 * each Library setting (src/core/core.ts, `createLibrary`'s `prepare`):
 *
 * - "auto": Auto · local models, one controller for the run, as the app keeps
 *   one per Ollama address. Text goes to Tev1 4B (or 0.8B on slow or small
 *   machines), PDFs with sparse text to Clef-Flash with page images.
 * - "tev-0.8b": the local decision model Tev1 0.8B, chosen by name.
 * - "clef-flash": Clef-Flash chosen by name with PDF page images on.
 * - "chat": the connected chat model, from INCARNAMIND_EVAL_CHAT_* (see eval/lib/config.ts),
 *   with the page images of PDFs without text when the model reads images.
 *
 * Local routes share this computer's Ollama with other work, so each waits
 * until no other model is loaded and unloads its own models afterwards.
 */

import { automaticGroupClassifier } from "../../../src/core/library/automatic";
import {
  chatGroupClassifier,
  decisionGroupClassifier,
  type GroupClassifier,
} from "../../../src/core/library/classifier";
import { readsImages, resolveCapabilities } from "../../../src/core/providers/capabilities";
import { catalogFacts, modelsProviderOfKind } from "../../../src/core/providers/catalog";
import { serviceForUrl } from "../../../src/core/providers/kinds";
import { createAiSdkChatModel } from "../../../src/core/providers/models";
import { DEFAULT_OLLAMA_SETTINGS } from "../../../src/core/providers/ollamaModels";
import type { ChatSettings } from "../../lib/config";

export const ROUTES = ["auto", "tev-0.8b", "clef-flash", "chat"] as const;
export type RouteName = (typeof ROUTES)[number];

export interface Route {
  name: RouteName;
  /** For the report. */
  label: string;
  /** The Ollama models it may load; empty for a chat model elsewhere. */
  models: string[];
  /** Runs on this computer: no data leaves it. */
  local: boolean;
  classifier: GroupClassifier;
}

export function buildRoute(
  name: RouteName,
  options: { ollama: string; chat: ChatSettings | null },
): Route {
  const { ollama } = options;
  switch (name) {
    case "auto":
      return {
        name,
        label: "Auto · local models (Tev1 4B, Clef-Flash for sparse PDFs)",
        models: ["tev1:4b", "tev1:0.8b", "clef-flash"],
        local: true,
        classifier: automaticGroupClassifier(ollama),
      };
    case "tev-0.8b":
      return {
        name,
        label: "Tev1 0.8B, chosen by name",
        models: ["tev1:0.8b"],
        local: true,
        classifier: decisionGroupClassifier({
          baseUrl: ollama,
          apiKey: "ollama",
          model: "tev1:0.8b",
          local: true,
        }),
      };
    case "clef-flash":
      return {
        name,
        label: "Clef-Flash, chosen by name, with PDF page images",
        models: ["clef-flash"],
        local: true,
        classifier: decisionGroupClassifier({
          baseUrl: ollama,
          apiKey: "ollama",
          model: "clef-flash",
          local: true,
          usePageImages: true,
        }),
      };
    case "chat": {
      const chat = options.chat;
      if (!chat)
        throw new Error(
          "The chat route needs INCARNAMIND_EVAL_CHAT_KIND and INCARNAMIND_EVAL_CHAT_MODEL (and a key for a cloud provider).",
        );
      const baseUrl = chat.kind === "ollama" ? (chat.baseUrl ?? ollama) : chat.baseUrl;
      const local =
        chat.kind === "ollama" ||
        (chat.kind === "openai-compatible" && baseUrl !== null && !serviceForUrl(baseUrl));
      const model = createAiSdkChatModel({
        kind: chat.kind,
        baseUrl,
        apiKey: chat.apiKey,
        modelId: chat.modelId,
        // Local chat models keep within an 8K window, like the app's default.
        ollama: DEFAULT_OLLAMA_SETTINGS,
      });
      // As the app knows it before a request: from the catalog.
      const images = readsImages(
        resolveCapabilities({
          catalog: catalogFacts(modelsProviderOfKind(chat.kind), chat.modelId),
        }),
      );
      return {
        name,
        label: `Chat model ${chat.kind}/${chat.modelId}${images ? ", with page images of scans" : ""}`,
        models: chat.kind === "ollama" ? [chat.modelId] : [],
        local,
        classifier: {
          ...chatGroupClassifier(model, local, images),
          model: { id: chat.modelId, images, reason: "selected" },
        },
      };
    }
  }
}

/** The models Ollama has loaded now. */
export async function loadedModels(ollama: string): Promise<string[]> {
  const response = await fetch(`${ollama}/api/ps`, { signal: AbortSignal.timeout(5_000) });
  if (!response.ok) throw new Error(`Ollama's /api/ps answered ${response.status}.`);
  const body = (await response.json()) as { models?: { name?: string; model?: string }[] };
  return (body.models ?? []).map((item) => item.name ?? item.model ?? "?");
}

const same = (a: string, b: string) => a === b || a === `${b}:latest` || b === `${a}:latest`;

/**
 * Waits until Ollama has no model loaded except `ours`: another agent's model
 * means a second model in memory, so the run waits for it to finish.
 */
export async function waitForOllama(
  ollama: string,
  ours: readonly string[],
  log: (line: string) => void,
  maxWaitMs = 60 * 60_000,
): Promise<void> {
  const started = Date.now();
  let told = false;
  for (;;) {
    const others = (await loadedModels(ollama)).filter(
      (model) => !ours.some((own) => same(model, own)),
    );
    if (others.length === 0) return;
    if (Date.now() - started > maxWaitMs)
      throw new Error(`Ollama still has ${others.join(", ")} loaded. Try again later.`);
    if (!told) log(`Waiting: Ollama has ${others.join(", ")} loaded by other work.`);
    told = true;
    await new Promise((resolve) => setTimeout(resolve, 20_000));
  }
}

/** Unloads models, as `ollama stop` does. Never throws. */
export async function unload(ollama: string, models: readonly string[]): Promise<void> {
  const loaded = await loadedModels(ollama).catch(() => [] as string[]);
  for (const model of models) {
    if (!loaded.some((each) => same(each, model))) continue;
    await fetch(`${ollama}/api/generate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, keep_alive: 0 }),
      signal: AbortSignal.timeout(30_000),
    }).catch(() => undefined);
  }
}

/** Installed Ollama models and their digests, for the report and the cache key. */
export async function installedModels(ollama: string): Promise<{ name: string; digest: string }[]> {
  const response = await fetch(`${ollama}/api/tags`, { signal: AbortSignal.timeout(5_000) });
  if (!response.ok) return [];
  const body = (await response.json()) as { models?: { name: string; digest: string }[] };
  return body.models ?? [];
}
