/**
 * `npm run eval`: the retrieval and Citation evaluation (eval/README.md, #31).
 *
 * It drives the core's public interface in Node, as the desktop app's UI
 * would: a new temporary data folder, embeddings turned on (they are off by
 * default) so the evaluation set's Documents are also processed with the real
 * built-in embedding model (on a worker thread), then searches for each
 * Question: keyword search's top 20 reranked by the real built-in reranking
 * model, as the search Tool does by default (the gate), hybrid search's
 * candidates reranked as with embeddings on, and the plain search modes,
 * reported only. When a chat model is given, embeddings are turned off again,
 * as Users have them, and Answers and their Citations are scored. Then the
 * same for the every-format set (#70), on a library of its own, reported per
 * format and never gating.
 * It runs under Vitest only for its TypeScript and worker-thread handling
 * (eval/vitest.config.ts); `npm test` never runs it.
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
import { type CitationRun, runCitations, summariseGroup } from "./lib/citations";
import { type EvalConfig, readConfig } from "./lib/config";
import { cloudEmbeddingProvider, createWorkerEmbedder } from "./lib/embedder";
import { FORMATS_SET, isGating, loadEvaluationSet } from "./lib/evaluationSet";
import { type FormatsReport, formatOf, summariseFormats } from "./lib/formats";
import { type Library, openLibrary } from "./lib/library";
import { createLog, type Log } from "./lib/log";
import { type EvalReport, terminalSummary, writeReports } from "./lib/report";
import { createWorkerCrossEncoder, openReranker, type RerankerInfo } from "./lib/rerank";
import {
  candidateCounts,
  GATING_MODE,
  HYBRID,
  KEYWORD_RERANK_DEPTH,
  RERANK_PER_LIST,
  RERANKED_SEARCHES,
  type RetrievalMode,
  type RetrievalRun,
  rerankedLabel,
  rerankMode,
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
    const paraphrase = summary.paraphrase.all;
    log(
      `${embedding}, ${label}: English ${summary.en.hits}/${summary.en.total}, Chinese ${summary.zh.hits}/${summary.zh.total}, cross-lingual ${summary.crossLingual.hits}/${summary.crossLingual.total}${translated ? ` (${translated.hits}/${translated.total} with a translated second query)` : ""}${paraphrase.total > 0 ? `, paraphrase ${paraphrase.hits}/${paraphrase.total}` : ""}`,
    );
  };
  line("keyword", "keyword");
  line(HYBRID, HYBRID);

  // Reranked modes, one model at a time: each is downloaded once into the model cache. Each
  // model reranks keyword search's top 20 (keyword + rerank, the gate's mode), then hybrid
  // search's candidates, opened afresh for each so the timings are the mode's own.
  const rerankers: RerankerInfo[] = [];
  for (const candidate of rerank?.candidates ?? []) {
    for (const search of RERANKED_SEARCHES) {
      const mode = rerankMode(candidate, search);
      const label = rerankedLabel(search, candidate.name);
      log(
        search === "hybrid"
          ? `Reranking keyword search's top ${RERANK_PER_LIST} and vector search's top ${RERANK_PER_LIST} with ${candidate.name}`
          : `Reranking keyword search's top ${KEYWORD_RERANK_DEPTH} with ${candidate.name}`,
      );
      const reranker = await openReranker(candidate, rerank?.cacheDir ?? "", log, mode);
      try {
        await runReranked(library.core, questions, ids, reranker, results);
        const info = reranker.info();
        rerankers.push(info);
        line(mode, mode === GATING_MODE ? `${label} (gating)` : label);
        log(
          `${label}: ${info.latency.mean.toFixed(0)} ms a query on average (95th percentile ${info.latency.p95.toFixed(0)} ms), loaded in ${info.loadSeconds.toFixed(1)} s`,
        );
      } finally {
        reranker.close();
      }
    }
  }

  const candidates = candidateCounts(results);
  const keywordCandidates = candidateCounts(results, "keyword");
  for (const counts of [candidates, keywordCandidates]) {
    if (!counts) continue;
    log(
      `${counts.search === "keyword" ? "Keyword + rerank searches" : "Reranked hybrid searches"} had ${counts.mean.toFixed(1)} candidates on average (${counts.min} to ${counts.max}, over ${counts.searches} searches)`,
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
    ...(keywordCandidates && { keywordRerankCandidates: keywordCandidates }),
  };
}

/**
 * Turns embeddings off in a library before its Answers are asked, so they
 * search as Users' do by default: keyword search, reranked. The vectors made
 * for the hybrid and vector modes stay, unused.
 */
async function searchAsByDefault(library: Library, log: Log): Promise<void> {
  await library.core.saveEmbeddingProvider({ kind: "off" });
  log("Embeddings off for the Answers, as by default: keyword search, reranked");
}

/**
 * The every-format set (#70): its own library of Word, PowerPoint, Excel,
 * CSV, Markdown, text and PDF Documents, searched as the gating set is and,
 * with a chat model, each Question asked once. Reported per format; it never
 * gates, so its Citation targets aren't failures either.
 */
async function runFormats(config: EvalConfig, log: Log): Promise<FormatsReport> {
  const set = loadEvaluationSet(root, FORMATS_SET);
  log(`Every format: ${set.questions.length} Questions over ${set.documents.length} Documents`);
  const library = await openLibrary({
    name: "formats",
    embedder: createWorkerEmbedder(),
    modelCache: join(config.cacheDir, "models"),
    reranker: createWorkerCrossEncoder(),
    documents: set.documents,
    keep: config.keepData,
    log,
    // A scanned page has no text: the Questions about it are known gaps.
    allowUnprocessed: true,
  });
  try {
    const unprocessed = set.documents.filter(
      ({ key }) =>
        library.documents.get(key)?.status !== "ready" &&
        !set.questions.some((question) => question.expected.document === key && question.knownGap),
    );
    if (unprocessed.length > 0) {
      throw new Error(
        `Documents didn't process: ${unprocessed.map(({ path }) => path).join(", ")}`,
      );
    }
    const retrieval = await retrieve(
      library,
      `${BUILT_IN_EMBEDDING_MODEL.name} (built-in)`,
      false,
      set.questions,
      log,
      { candidates: [BUILT_IN_RERANKING_MODEL], cacheDir: config.cacheDir },
    );
    if (config.chat) await searchAsByDefault(library, log);
    const citations: CitationRun | { skipped: string } = config.chat
      ? {
          ...(await runCitations(
            library,
            set.questions,
            config.chat,
            { ...config, maxRounds: 1 },
            log,
          )),
          failures: [],
        }
      : { skipped: "no chat model was given." };
    return {
      source: set.source,
      documents: set.documents.map(({ key, path }) => {
        const document = library.documents.get(key);
        return {
          key,
          name: document?.name ?? key,
          format: formatOf(path),
          status: document?.status ?? "failed",
        };
      }),
      retrieval,
      formats: summariseFormats(set, retrieval, citations),
      crossLingualCitations:
        "skipped" in citations
          ? null
          : summariseGroup(citations.answers.filter((answer) => answer.crossLingual)),
      citations,
    };
  } finally {
    await library.close();
  }
}

test("retrieval and Citation evaluation", async () => {
  const started = new Date();
  const log = createLog(started.getTime());
  const config = readConfig(root);
  const set = loadEvaluationSet(root);
  log(`${set.questions.length} Questions over ${set.documents.length} Documents`);
  log(`Embedding model cache: ${config.cacheDir}`);

  // The core reranks Answers' searches with the built-in reranking model, as the app does by
  // default. The library turns embeddings on, for the hybrid and vector modes reported.
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

    if (config.chat) await searchAsByDefault(builtIn, log);
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
    const gating = set.questions.filter(isGating);
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
          crossLingual: set.questions.filter((question) => question.crossLingual).length,
          paraphrase: set.questions.filter((question) => question.paraphrase).length,
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
  report.formats = await runFormats(config, log);
  report.run.seconds = (Date.now() - started.getTime()) / 1000;

  const dir = await writeReports(report, config.resultsDir, root);
  console.log(terminalSummary(report, dir, root));
  expect(report.failures, "The evaluation missed its targets (see the report).").toEqual([]);
});
