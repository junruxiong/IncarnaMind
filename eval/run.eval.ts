/**
 * `npm run eval`: the retrieval and Citation evaluation (eval/README.md, #31).
 *
 * It drives the core's public interface in Node, as the desktop app's UI
 * would: a new temporary data folder, the evaluation set's Documents added
 * and processed with the real built-in embedding model (on a worker thread),
 * then searches for each Question, and, when a chat model is given, Answers
 * and their Citations. It runs under Vitest only for its TypeScript and
 * worker-thread handling (eval/vitest.config.ts); `npm test` never runs it.
 */
import { execFileSync } from "node:child_process";
import { arch, cpus, platform } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";
import { BUILT_IN_EMBEDDING_MODEL } from "../src/core";
import { type CitationRun, runCitations } from "./lib/citations";
import { readConfig } from "./lib/config";
import { cloudEmbeddingProvider, createWorkerEmbedder } from "./lib/embedder";
import { loadEvaluationSet } from "./lib/evaluationSet";
import { type Library, openLibrary } from "./lib/library";
import { createLog, type Log } from "./lib/log";
import { type EvalReport, terminalSummary, writeReports } from "./lib/report";
import {
  GATING_MODE,
  type RetrievalRun,
  retrievalFailures,
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
): Promise<RetrievalRun> {
  const ids = new Map([...library.documents].map(([key, document]) => [key, document.id]));
  const results = await runRetrieval(library.core, questions, ids);
  const summary = summarise(results);
  const gate = summary[GATING_MODE];
  if (gate) {
    log(
      `${embedding}, ${GATING_MODE}: English ${gate.en.hits}/${gate.en.total}, Chinese ${gate.zh.hits}/${gate.zh.total}, cross-lingual ${gate.crossLingual.hits}/${gate.crossLingual.total}`,
    );
  }
  return {
    embedding,
    gating,
    passageCount: library.passageCount,
    processingSeconds: library.processingSeconds,
    questions: results,
    summary,
  };
}

test("retrieval and Citation evaluation", async () => {
  const started = new Date();
  const log = createLog(started.getTime());
  const config = readConfig(root);
  const set = loadEvaluationSet(root);
  log(`${set.questions.length} Questions over ${set.documents.length} Documents`);
  log(`Embedding model cache: ${config.cacheDir}`);

  const builtIn = await openLibrary({
    name: "built-in",
    embedder: createWorkerEmbedder(),
    modelCache: join(config.cacheDir, "models"),
    documents: set.documents,
    keep: config.keepData,
    log,
  });
  let report: EvalReport;
  try {
    const runs = [
      await retrieve(
        builtIn,
        `${BUILT_IN_EMBEDDING_MODEL.name} (built-in)`,
        true,
        set.questions,
        log,
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
