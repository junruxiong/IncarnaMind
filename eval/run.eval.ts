/**
 * `npm run eval`: the retrieval and Citation evaluation (eval/README.md, #31).
 *
 * It drives the core's public interface in Node, as the desktop app's UI
 * would: a new temporary data folder, the evaluation set's Documents added
 * and processed with the real built-in embedding model (on a worker thread),
 * then searches for each Question, reranked by the real built-in reranking
 * model as the search Tool does by default (the gate), and, when a chat model
 * is given, Answers and their Citations. It runs under Vitest only for its TypeScript and
 * worker-thread handling (eval/vitest.config.ts); `npm test` never runs it.
 */
import { execFileSync } from "node:child_process";
import { arch, cpus, platform } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";
import {
  BUILT_IN_EMBEDDING_MODEL,
  BUILT_IN_RERANKING_MODEL,
  type RerankingModelDefinition,
} from "../src/core";
import { type CitationRun, runCitations } from "./lib/citations";
import { readConfig } from "./lib/config";
import { cloudEmbeddingProvider, createWorkerEmbedder } from "./lib/embedder";
import { loadEvaluationSet } from "./lib/evaluationSet";
import { type Library, openLibrary } from "./lib/library";
import { createLog, type Log } from "./lib/log";
import { type EvalReport, terminalSummary, writeReports } from "./lib/report";
import { createWorkerCrossEncoder, openReranker, type RerankerInfo } from "./lib/rerank";
import {
  candidateCounts,
  GATING_MODE,
  HYBRID,
  RERANK_PER_LIST,
  type RetrievalMode,
  type RetrievalRun,
  retrievalFailures,
  runReranked,
  runRetrieval,
  summarise,
  TOP_K,
} from "./lib/retrieval";

const root = fileURLToPath(new URL("..", import.meta.url));

function git(...args: string[]): string {
  return execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
}

function commit(): string {
  try {
    const changed = git("status", "--porcelain") !== "";
    return `${git("rev-parse", "--short", "HEAD")}${changed ? " (with uncommitted changes)" : ""}`;
  } catch {
    return "unknown";
  }
}

async function retrieve(
  library: Library,
  embedding: string,
  gating: boolean,
  questions: Parameters<typeof runRetrieval>[1],
  log: Log,
  rerank: { candidates: readonly RerankingModelDefinition[]; cacheDir: string } | null = null,
): Promise<RetrievalRun> {
  const ids = new Map([...library.documents].map(([key, document]) => [key, document.id]));
  const results = await runRetrieval(library.core, questions, ids);
  const line = (mode: RetrievalMode, label: string) => {
    const summary = summarise(results, [mode])[mode];
    if (!summary) return;
    const translated = summary.crossLingualTranslated;
    log(
      `${embedding}, ${label}: English ${summary.en.hits}/${summary.en.total}, Chinese ${summary.zh.hits}/${summary.zh.total}, cross-lingual ${summary.crossLingual.hits}/${summary.crossLingual.total}${translated ? ` (${translated.hits}/${translated.total} with a translated second query)` : ""}`,
    );
  };
  line(HYBRID, HYBRID);

  // Reranked modes, one model at a time: each is downloaded once into the model cache.
  const rerankers: RerankerInfo[] = [];
  for (const candidate of rerank?.candidates ?? []) {
    log(
      `Reranking keyword search's top ${RERANK_PER_LIST} and vector search's top ${RERANK_PER_LIST} with ${candidate.name}`,
    );
    const reranker = await openReranker(candidate, rerank?.cacheDir ?? "", log);
    try {
      await runReranked(library.core, questions, ids, reranker, results);
      const info = reranker.info();
      rerankers.push(info);
      line(
        info.mode as RetrievalMode,
        info.mode === GATING_MODE
          ? `${HYBRID} + ${candidate.name} (gating)`
          : `${HYBRID} + ${candidate.name}`,
      );
      log(
        `${candidate.name}: ${info.latency.mean.toFixed(0)} ms a query on average (95th percentile ${info.latency.p95.toFixed(0)} ms), loaded in ${info.loadSeconds.toFixed(1)} s`,
      );
    } finally {
      reranker.close();
    }
  }

  const candidates = candidateCounts(results);
  if (candidates) {
    log(
      `Reranked searches had ${candidates.mean.toFixed(1)} candidates on average (${candidates.min} to ${candidates.max}, over ${candidates.searches} searches)`,
    );
  }

  return {
    embedding,
    gating,
    passageCount: library.passageCount,
    processingSeconds: library.processingSeconds,
    questions: results,
    summary: summarise(results),
    ...(rerankers.length > 0 && { rerankers }),
    ...(candidates && { rerankCandidates: candidates }),
  };
}

test("retrieval and Citation evaluation", async () => {
  const started = new Date();
  const log = createLog(started.getTime());
  const config = readConfig(root);
  const set = loadEvaluationSet(root);
  log(`${set.questions.length} Questions over ${set.documents.length} Documents`);
  log(`Embedding model cache: ${config.cacheDir}`);

  // The core reranks Answers' searches with the built-in reranking model, as the app does by default.
  const builtIn = await openLibrary({
    name: "built-in",
    embedder: createWorkerEmbedder(),
    modelCache: join(config.cacheDir, "models"),
    reranker: createWorkerCrossEncoder(),
    documents: set.documents,
    keep: config.keepData,
    log,
  });
  let report: EvalReport;
  try {
    // The built-in reranking model gates; the other candidates given are compared with it.
    const rerankWith = [
      BUILT_IN_RERANKING_MODEL,
      ...config.rerank.filter((candidate) => candidate.id !== BUILT_IN_RERANKING_MODEL.id),
    ];
    const runs = [
      await retrieve(
        builtIn,
        `${BUILT_IN_EMBEDDING_MODEL.name} (built-in)`,
        true,
        set.questions,
        log,
        { candidates: rerankWith, cacheDir: config.cacheDir },
      ),
    ];

    const cloud = config.cloudEmbedding;
    if (cloud) {
      const library = await openLibrary({
        name: "cloud",
        embedder: createWorkerEmbedder(),
        embeddingProvider: cloudEmbeddingProvider(cloud),
        documents: set.documents,
        keep: config.keepData,
        log,
      });
      try {
        runs.push(
          await retrieve(library, `${cloud.kind}/${cloud.modelId}`, false, set.questions, log),
        );
      } finally {
        await library.close();
      }
    }

    const citations: CitationRun | { skipped: string } = config.chat
      ? await runCitations(builtIn, set.questions, config.chat, config, log)
      : {
          skipped:
            "no chat model was given. Set INCARNAMIND_EVAL_CHAT_KIND, INCARNAMIND_EVAL_CHAT_MODEL and INCARNAMIND_EVAL_CHAT_KEY (eval/README.md).",
        };

    const failures = [
      ...retrievalFailures(runs[0]?.summary[GATING_MODE]),
      ...("skipped" in citations ? [] : citations.failures),
    ];
    const cpu = cpus();
    const gating = set.questions.filter((question) => !question.crossLingual);
    report = {
      result: failures.length === 0 ? "pass" : "fail",
      failures,
      run: {
        startedAt: started.toISOString(),
        seconds: (Date.now() - started.getTime()) / 1000,
        commit: commit(),
        node: process.version,
        platform: `${platform()} ${arch()}`,
        cpu: `${cpu[0]?.model.trim() ?? "unknown CPU"} (${cpu.length} cores)`,
      },
      evaluationSet: {
        source: set.source,
        hitRule: set.hitRule,
        questions: {
          gating: {
            en: gating.filter((question) => question.language === "en").length,
            zh: gating.filter((question) => question.language === "zh").length,
          },
          crossLingual: set.questions.length - gating.length,
        },
      },
      documents: set.documents.map(({ key }) => {
        const document = builtIn.documents.get(key);
        return { key, name: document?.name ?? key, pageCount: document?.pageCount ?? null };
      }),
      retrieval: { topK: TOP_K, gatingMode: GATING_MODE, runs },
      citations,
    };
  } finally {
    await builtIn.close();
  }

  const dir = await writeReports(report, config.resultsDir, root);
  console.log(terminalSummary(report, dir, root));
  expect(report.failures, "The evaluation missed its targets (see the report).").toEqual([]);
});
