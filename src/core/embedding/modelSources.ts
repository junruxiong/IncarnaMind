/**
 * Where the built-in models download from (#1578), so they reach the User wherever they are,
 * mainland China included, with no setting. Each file is tried at Hugging Face, IncarnaMind's own
 * copy on ModelScope, hf-mirror.com and IncarnaMind's own copy on Cloudflare R2, in turn. A probe
 * of Hugging Face and ModelScope decides which leads. Whatever a source sends is checked against
 * the size and SHA-256 the app pins (./download), never against the source's own word, so a
 * source that serves other bytes falls through to the next.
 *
 * Network traffic that carries no User content: the probe is a HEAD request to each host's
 * front page.
 */
import type { ModelFile } from "../adapters";
import type { ModelSourceChoice, ModelSourceId } from "../api";
import type { Language } from "../language";

/**
 * IncarnaMind's own copies of the models, each the base URL of a model's folder ("{folder}" is
 * its folder under models/), holding the files at the paths the app pins. Off until they exist.
 */
export interface ModelCopies {
  modelscope?: string;
  r2?: string;
}

export const MODEL_COPIES: ModelCopies = {};

/** The probe gives each host this long to answer. */
export const PROBE_MS = 3000;
/** The probe's answer holds this long, unless the proxy changes. */
export const PROBE_KEPT_MS = 10 * 60_000;
/** A file that gets no bytes for this long moves to the next source. */
export const STALL_MS = 25_000;

const HUGGING_FACE = "huggingface.co";
const HF_MIRROR = "hf-mirror.com";
const PROBED = {
  huggingface: "https://huggingface.co/",
  modelscope: "https://www.modelscope.cn/",
} as const;

/** Outside mainland China Hugging Face leads; inside, ModelScope and the Hugging Face mirror do. */
const ORDER: readonly ModelSourceId[] = ["huggingface", "modelscope", "hf-mirror", "r2"];
const CHINA_ORDER: readonly ModelSourceId[] = ["modelscope", "hf-mirror", "huggingface", "r2"];

export interface ModelSourcesOptions {
  fetch: typeof fetch;
  now(): number;
  /** The interface's language: Chinese puts mainland China's order first when the probe can't tell. */
  language(): Language;
  /** The User's pick on a failed download's line ("auto" unless they made one). */
  choice(): ModelSourceChoice;
  /** The system proxy for Hugging Face, as the host resolves it: a change probes again. */
  proxy?(): Promise<string>;
  copies?: ModelCopies;
  /** Overrides how long a file may get no bytes (tests). */
  stallMs?: number;
}

export interface SourcedUrl {
  source: ModelSourceId;
  url: URL;
}

export function createModelSources(options: ModelSourcesOptions) {
  const copies = options.copies ?? MODEL_COPIES;
  let probed: { at: number; proxy: string; china: Promise<boolean> } | null = null;
  /** Sources that failed, to the back of the order until this time. */
  const failedUntil = new Map<ModelSourceId, number>();

  /** Whichever host answers first: ModelScope first is mainland China. */
  async function probe(): Promise<boolean> {
    const ask = async (source: keyof typeof PROBED) => {
      const response = await options.fetch(PROBED[source], {
        method: "HEAD",
        signal: AbortSignal.timeout(PROBE_MS),
      });
      void response.body?.cancel();
      return source;
    };
    try {
      return (await Promise.any([ask("huggingface"), ask("modelscope")])) === "modelscope";
    } catch {
      return options.language() === "zh-CN";
    }
  }

  async function inChina(): Promise<boolean> {
    const proxy = (await options.proxy?.().catch(() => "")) ?? "";
    const now = options.now();
    if (!probed || now - probed.at >= PROBE_KEPT_MS || probed.proxy !== proxy) {
      probed = { at: now, proxy, china: probe() };
      failedUntil.clear();
    }
    return probed.china;
  }

  const copyOf = (base: string | undefined, folder: string, file: ModelFile) =>
    base ? new URL(file.path, base.replaceAll("{folder}", encodeURIComponent(folder))) : null;

  function urlOf(source: ModelSourceId, folder: string, file: ModelFile, baseUrl: string) {
    const pinned = new URL(file.url ?? file.path, baseUrl);
    switch (source) {
      case "huggingface":
        return pinned;
      case "hf-mirror": {
        if (pinned.host !== HUGGING_FACE) return null;
        const mirrored = new URL(pinned);
        mirrored.host = HF_MIRROR;
        return mirrored;
      }
      case "modelscope":
        return copyOf(copies.modelscope, folder, file);
      case "r2":
        return copyOf(copies.r2, folder, file);
    }
  }

  return {
    /**
     * Where to download a model's file from, in the order to try: the User's pick first, then
     * the probe's order, sources that failed lately last. `baseUrl` and the file's `url` are
     * the pinned Hugging Face location; `folder` is the model's folder under models/.
     */
    async urlsFor(folder: string, file: ModelFile, baseUrl: string): Promise<SourcedUrl[]> {
      const order = (await inChina()) ? CHINA_ORDER : ORDER;
      const picked = options.choice();
      const now = options.now();
      const late = (source: ModelSourceId) => (failedUntil.get(source) ?? 0) > now;
      const ranked = [
        ...order.filter((source) => source === picked),
        ...order.filter((source) => source !== picked && !late(source)),
        ...order.filter((source) => source !== picked && late(source)),
      ];
      return ranked.flatMap((source) => {
        const url = urlOf(source, folder, file, baseUrl);
        return url ? [{ source, url }] : [];
      });
    },
    /** A source failed a file: it goes last for the other files, until the probe is run again. */
    failed(source: ModelSourceId): void {
      failedUntil.set(source, options.now() + PROBE_KEPT_MS);
    },
    /** The sources the User may pick on a failed download's line. */
    choices(): ModelSourceChoice[] {
      return copies.modelscope ? ["auto", "huggingface", "modelscope"] : ["auto", "huggingface"];
    },
    stallMs: options.stallMs ?? STALL_MS,
    fetch: options.fetch,
  };
}

export type ModelSources = ReturnType<typeof createModelSources>;
