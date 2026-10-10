/**
 * Queries a chat model writes for a Question before it is searched, for the
 * reranked modes keyword + rewrites and keyword + sub-questions (see
 * ./searches): two other phrasings in the Documents' own words, or the
 * Question broken down into one-hop questions, as the old command-line app
 * refined a Question before searching. One short call per Question, only when
 * a chat model is given (INCARNAMIND_EVAL_CHAT_*).
 *
 * The answers are kept in eval/results/query-rewrites.json, by model and
 * prompt, so a later run searches the same queries and its figures compare;
 * each keeps the time the call took and the tokens it used when it was made.
 */
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { generateText } from "ai";
import { defaultOllamaUrl } from "../../src/core/providers/kinds";
import { createAiSdkChatModel } from "../../src/core/providers/models";
import type { ChatSettings } from "./config";
import type { EvalQuestion } from "./evaluationSet";
import type { Log } from "./log";

/** The reranked searches whose queries a chat model writes. */
export type QueryModelSearch = "rewrites" | "sub-questions";

export const QUERY_MODEL_SEARCHES: readonly QueryModelSearch[] = ["rewrites", "sub-questions"];

/** At most this many queries are kept of a model's answer, besides the Question. */
export const MOST_QUERIES: Record<QueryModelSearch, number> = { rewrites: 2, "sub-questions": 3 };

export const CACHE_FILE = "query-rewrites.json";

/** A Document as the prompt names it. */
export interface PromptDocument {
  name: string;
  language: string | null;
}

/** The prompt for a Question: the Documents searched, what to write, and the Question. */
export function queryPrompt(
  search: QueryModelSearch,
  question: string,
  documents: readonly PromptDocument[],
): string {
  const listed = documents
    .map(({ name, language }) => `- ${name}${language ? ` (${language})` : ""}`)
    .join("\n");
  const task =
    search === "rewrites"
      ? "Write 2 other ways to ask the question below, in the words and the language the documents themselves would most likely use where they answer it: their terms, not the question's. Write each on a line of its own, and nothing else."
      : "If the question below asks more than one thing, compares things or needs several steps, break it down into at most 3 standalone questions that each ask one thing. Otherwise write the question as it is. Write each on a line of its own, and nothing else.";
  return `A question will be searched for by keyword in these documents:\n${listed}\n\n${task}\n\nQuestion: ${question}`;
}

const NUMBERING = /^\s*(?:\d+\s*[.)、:：]|[-*•])\s*/u;
const QUOTES = /^["'“”‘’「」]+|["'“”‘’「」]+$/gu;
const comparable = (text: string) => text.toLowerCase().replace(/[\s\p{P}]+/gu, "");

/**
 * The queries in a model's answer: one per line, numbering, bullets and
 * quotes taken off, the Question itself and repeats left out, at most `most`.
 */
export function parseQueries(text: string, question: string, most: number): string[] {
  const seen = new Set([comparable(question)]);
  const queries: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    const query = line.replace(NUMBERING, "").replace(QUOTES, "").trim();
    const key = comparable(query);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    queries.push(query);
    if (queries.length === most) break;
  }
  return queries;
}

/** One answer of the model, as kept. */
export interface QueryModelCall {
  queries: string[];
  /** How long the call took when it was made. */
  ms: number;
  inputTokens: number | null;
  outputTokens: number | null;
}

interface CacheEntry extends QueryModelCall {
  model: string;
  search: QueryModelSearch;
  questionId: string;
  question: string;
  madeAt: string;
}

interface Cache {
  version: 1;
  entries: Record<string, CacheEntry>;
}

/** What one reranked search's model calls cost, for the report. */
export interface QueryModelCost {
  search: QueryModelSearch;
  model: string;
  /** Questions it wrote queries for. */
  questions: number;
  /** Answers made in this run; the others came from the cache. */
  calls: number;
  cached: number;
  /** Per Question, as measured when each call was made. */
  meanMs: number;
  p95Ms: number;
  meanInputTokens: number | null;
  meanOutputTokens: number | null;
  /** How many queries it wrote per Question besides the Question, on average. */
  meanQueries: number;
}

/** Asks the model a prompt: its answer, and the tokens it used when the provider says. */
export type Generate = (
  prompt: string,
) => Promise<{ text: string; inputTokens: number | null; outputTokens: number | null }>;

/** The chat model given, through the core's own AI SDK providers. */
export function chatGenerate(chat: ChatSettings): Generate {
  const model = createAiSdkChatModel({
    kind: chat.kind,
    modelId: chat.modelId,
    apiKey: chat.apiKey,
    baseUrl: chat.kind === "ollama" ? (chat.baseUrl ?? defaultOllamaUrl()) : chat.baseUrl,
  });
  return async (prompt) => {
    // No cap on the answer's tokens: a reasoning model counts its reasoning in them.
    const result = await generateText({
      model,
      prompt,
      temperature: 0,
      maxRetries: 2,
      abortSignal: AbortSignal.timeout(120_000),
    });
    return {
      text: result.text,
      inputTokens: result.usage.inputTokens ?? null,
      outputTokens: result.usage.outputTokens ?? null,
    };
  };
}

const cacheKey = (model: string, prompt: string) =>
  createHash("sha256")
    .update(JSON.stringify([model, prompt]))
    .digest("hex");

async function readCache(path: string): Promise<Cache> {
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as Cache;
    if (parsed.version === 1 && parsed.entries) return parsed;
  } catch {
    // None yet, or unreadable: start afresh.
  }
  return { version: 1, entries: {} };
}

const quantile = (sorted: readonly number[], share: number) =>
  sorted[Math.min(sorted.length - 1, Math.ceil(share * sorted.length) - 1)] ?? 0;

const meanOf = (values: readonly (number | null)[]) => {
  const known = values.filter((value): value is number => value !== null);
  return known.length ? known.reduce((sum, value) => sum + value, 0) / known.length : null;
};

/**
 * Each Question's queries for a reranked search: from the cache when this
 * model was asked this prompt before, else asked now and kept.
 */
export async function prepareQueries(options: {
  search: QueryModelSearch;
  model: string;
  generate: Generate;
  questions: readonly EvalQuestion[];
  documents: readonly PromptDocument[];
  resultsDir: string;
  log: Log;
}): Promise<{ queries: Map<string, string[]>; cost: QueryModelCost }> {
  const { search, model, generate, questions, documents, resultsDir, log } = options;
  const path = join(resultsDir, CACHE_FILE);
  const cache = await readCache(path);
  const queries = new Map<string, string[]>();
  const used: CacheEntry[] = [];
  let calls = 0;
  for (const question of questions) {
    const prompt = queryPrompt(search, question.question, documents);
    const key = cacheKey(model, prompt);
    let entry = cache.entries[key];
    if (!entry) {
      const started = performance.now();
      const answer = await generate(prompt);
      entry = {
        model,
        search,
        questionId: question.id,
        question: question.question,
        madeAt: new Date().toISOString(),
        queries: parseQueries(answer.text, question.question, MOST_QUERIES[search]),
        ms: performance.now() - started,
        inputTokens: answer.inputTokens,
        outputTokens: answer.outputTokens,
      };
      cache.entries[key] = entry;
      calls++;
      // Kept as it goes, so a run stopped halfway keeps what it paid for.
      await mkdir(resultsDir, { recursive: true });
      await writeFile(path, `${JSON.stringify(cache, null, 2)}\n`);
    }
    used.push(entry);
    queries.set(question.id, entry.queries);
  }
  log(
    `${search}: ${model} wrote queries for ${questions.length} Questions (${calls} asked now, ${questions.length - calls} from ${CACHE_FILE})`,
  );
  const ms = used.map((entry) => entry.ms).sort((a, b) => a - b);
  return {
    queries,
    cost: {
      search,
      model,
      questions: used.length,
      calls,
      cached: used.length - calls,
      meanMs: meanOf(ms) ?? 0,
      p95Ms: quantile(ms, 0.95),
      meanInputTokens: meanOf(used.map((entry) => entry.inputTokens)),
      meanOutputTokens: meanOf(used.map((entry) => entry.outputTokens)),
      meanQueries: meanOf(used.map((entry) => entry.queries.length)) ?? 0,
    },
  };
}
