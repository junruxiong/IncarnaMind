/**
 * The grouping check's own settings, from environment variables only
 * (eval/grouping/README.md). The chat model, the model cache, the results
 * folder and keeping the data folder are the evaluation's
 * (eval/lib/config.ts). Keys are read from INCARNAMIND_EVAL_* variables only.
 */
import { existsSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { JEV_DEFAULT_MODEL } from "../../../src/core";
import { JEV_HOSTED_URL, JEV_PATH } from "../../../src/core/providers/jev";
import { normalizeBaseUrl, serviceForUrl } from "../../../src/core/providers/kinds";

/** A classifier answering `POST {base}/v1/systemone`: TypeSafe's Jev, or Clef-Flash in Ollama. */
export interface SystemOneSettings {
  name: string;
  baseUrl: string;
  apiKey: string;
  model: string;
  /** On this computer: nothing leaves it. */
  local: boolean;
}

export interface FounderSettings {
  /** The User's own library, read only: its files are added to a temporary data folder. */
  folder: string;
  /** At most this many of its files, drawn at random with `seed`; null for all. */
  limit: number | null;
  /** Draws the files (with a limit) and the 30-Document sample. */
  seed: number;
}

export interface GroupingConfig {
  clef: SystemOneSettings | null;
  jev: SystemOneSettings | null;
  founder: FounderSettings | null;
  /** Measure k-means at 5,000 Documents and the means at 100,000 Passages. */
  timing: boolean;
}

type Env = Readonly<Record<string, string | undefined>>;

const PREFIX = "INCARNAMIND_EVAL_";

/** Ollama answers Clef-Flash's requests without a key; the request still carries one. */
const NO_KEY = "ollama";

export const CLEF_DEFAULT_MODEL = "clef-flash";

/** Draws the founder's sample; any fixed number keeps it repeatable. */
export const DEFAULT_SAMPLE_SEED = 51;

function value(env: Env, name: string): string | null {
  const raw = env[`${PREFIX}${name}`]?.trim();
  return raw ? raw : null;
}

function positiveInteger(env: Env, name: string): number | null {
  const raw = value(env, name);
  if (raw === null) return null;
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${PREFIX}${name} must be a whole number of at least 1, not "${raw}".`);
  }
  return parsed;
}

/** A base URL, without a trailing `/v1/systemone` if the whole endpoint was given. */
function systemOneBase(raw: string): string {
  const url = normalizeBaseUrl(raw);
  return url.endsWith(JEV_PATH) ? url.slice(0, -JEV_PATH.length) : url;
}

function clefSettings(env: Env): SystemOneSettings | null {
  const url = value(env, "CLEF_URL");
  const model = value(env, "CLEF_MODEL");
  if (url === null) {
    if (model) throw new Error(`Set ${PREFIX}CLEF_URL to Ollama's address too.`);
    return null;
  }
  const baseUrl = systemOneBase(url);
  return {
    name: `Clef-Flash (${model ?? CLEF_DEFAULT_MODEL}, Ollama at ${baseUrl})`,
    baseUrl,
    apiKey: NO_KEY,
    model: model ?? CLEF_DEFAULT_MODEL,
    local: serviceForUrl(baseUrl) === null,
  };
}

function jevSettings(env: Env): SystemOneSettings | null {
  const apiKey = value(env, "JEV_KEY");
  const url = value(env, "JEV_URL");
  const model = value(env, "JEV_MODEL");
  if (apiKey === null) {
    if (url || model)
      throw new Error(`Set ${PREFIX}JEV_KEY too, or none of the ${PREFIX}JEV_* variables.`);
    return null;
  }
  const baseUrl = url ? systemOneBase(url) : JEV_HOSTED_URL;
  return {
    name: `Jev (${model ?? JEV_DEFAULT_MODEL}${url ? `, ${baseUrl}` : ""})`,
    baseUrl,
    apiKey,
    model: model ?? JEV_DEFAULT_MODEL,
    local: serviceForUrl(baseUrl) === null,
  };
}

function founderSettings(env: Env): FounderSettings | null {
  const folder = value(env, "GROUPING_FOLDER");
  const limit = positiveInteger(env, "GROUPING_LIMIT");
  const seed = positiveInteger(env, "GROUPING_SEED") ?? DEFAULT_SAMPLE_SEED;
  if (folder === null) {
    if (limit !== null) throw new Error(`${PREFIX}GROUPING_LIMIT needs ${PREFIX}GROUPING_FOLDER.`);
    return null;
  }
  const absolute = resolve(folder);
  if (!existsSync(absolute) || !statSync(absolute).isDirectory()) {
    throw new Error(`${PREFIX}GROUPING_FOLDER isn't a folder: ${absolute}`);
  }
  return { folder: absolute, limit, seed };
}

export function readGroupingConfig(env: Env = process.env): GroupingConfig {
  const timing = value(env, "GROUPING_TIMING");
  if (timing !== null && timing !== "0" && timing !== "1") {
    throw new Error(`${PREFIX}GROUPING_TIMING must be 0 or 1, not "${timing}".`);
  }
  return {
    clef: clefSettings(env),
    jev: jevSettings(env),
    founder: founderSettings(env),
    timing: timing !== "0",
  };
}
