/**
 * `npm run eval:hard`: the hard tier (eval/hard/README.md), reported and never
 * gating. It fetches a library of public Documents across domains into the
 * evaluation's cache (checked against their SHA-256), adds them to a new
 * temporary data folder with embeddings on, and asks the hard tier's
 * Questions in every retrieval mode, per domain and difficulty; with a chat
 * model, it asks them once each and scores the Citations per difficulty.
 * It runs under Vitest only for its TypeScript and worker threads
 * (eval/hard/vitest.config.ts); neither `npm test` nor `npm run eval` runs it.
 */
import { execFileSync } from "node:child_process";
import { arch, cpus, platform, totalmem } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";
import {
  BUILT_IN_EMBEDDING_MODEL,
  BUILT_IN_RERANKING_MODEL,
  type Core,
  DATABASE_FILE,
  type DocumentStatus,
  type Embedder,
} from "../../src/core";
import { openDatabase } from "../../src/core/storage";
import type { UnitKind, UnitLabel } from "../../src/shared/units";
import { readConfig } from "../lib/config";
import { createWorkerEmbedder } from "../lib/embedder";
import { openLibrary } from "../lib/library";
import { createLog } from "../lib/log";
import { createWorkerCrossEncoder, openReranker, type RerankerInfo } from "../lib/rerank";
import { KEYWORD_RERANK_DEPTH, rerankedLabel, rerankMode, TOP_K } from "../lib/retrieval";
import { type HardCitations, runHardCitations } from "./lib/citations";
import { fetchLibrary } from "./lib/fetch";
import { timeKeywordSearch } from "./lib/keywordTiming";
import { downloadBytes, loadManifest, MANIFEST } from "./lib/manifest";
import {
  type HardQuestion,
  loadHardQuestions,
  type StoredText,
  textProblems,
} from "./lib/questions";
import { type HardReport, hardSummary, modeLabels, writeHardReports } from "./lib/report";
import { HARD_RERANKED_SEARCHES, rerankAll, searchAll, summariseHard } from "./lib/scoring";

const root = fileURLToPath(new URL("../..", import.meta.url));

function commit(): string {
  const git = (...args: string[]) =>
    execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
  try {
    const changed = git("status", "--porcelain") !== "";
    return `${git("rev-parse", "--short", "HEAD")}${changed ? " (with uncommitted changes)" : ""}`;
  } catch {
    return "unknown";
  }
}

/** The built-in model, timing how long it spends embedding until `stop` is called. */
function timedEmbedder(embedder: Embedder) {
  let busyMs = 0;
  let counting = true;
  return {
    embedder: {
      load: (files) => embedder.load(files),
      async embed(text) {
        const started = performance.now();
        try {
          return await embedder.embed(text);
        } finally {
          if (counting) busyMs += performance.now() - started;
        }
      },
      close: () => embedder.close(),
    } satisfies Embedder,
    stop: () => {
      counting = false;
    },
    seconds: () => busyMs / 1000,
  };
}

/** Statuses after which a Document's Passages are in the keyword index. */
const INDEXED: ReadonlySet<DocumentStatus> = new Set([
  "waiting-for-model",
  "embedding",
  "ready",
  "failed",
  "no-text",
]);

/** When each Document was first seen, keyword-indexed and ready, from the core's events. */
function watchIndexing(core: Core) {
  const seen = new Map<string, { first: number; indexed?: number; ready?: number }>();
  core.on("document.status", (document) => {
    const now = Date.now();
    const times = seen.get(document.id) ?? { first: now };
    if (INDEXED.has(document.status)) times.indexed ??= now;
    if (
      document.status === "ready" ||
      document.status === "no-text" ||
      document.status === "failed"
    ) {
      times.ready ??= now;
    }
    seen.set(document.id, times);
  });
  return () => {
    const all = [...seen.values()];
    const start = Math.min(...all.map((times) => times.first));
    const latest = (pick: (times: (typeof all)[number]) => number | undefined) =>
      (Math.max(...all.map((times) => pick(times) ?? times.first)) - start) / 1000;
    return {
      keywordSeconds: latest((times) => times.indexed),
      readySeconds: latest((times) => times.ready),
    };
  };
}

test("the hard tier", async () => {
  const started = new Date();
  const log = createLog(started.getTime());
  const config = readConfig(root);
  const manifest = loadManifest(root);
  const set = loadHardQuestions(root, manifest);
  log(
    `The hard tier: ${set.questions.length} Questions over ${manifest.documents.length} Documents`,
  );

  let peak = 0;
  let peakIndexing = 0;
  const sampler = setInterval(() => {
    peak = Math.max(peak, process.memoryUsage().rss);
  }, 500);
  sampler.unref();

  try {
    const fetched = await fetchLibrary(manifest, { root, cacheDir: config.cacheDir, log });
    const available = fetched.documents.filter((document) => document.status === "ready");
    log(
      `${available.length} of ${manifest.documents.length} Documents in the cache (${(fetched.fetchedBytes / 1e6).toFixed(0)} MB fetched in ${fetched.seconds.toFixed(0)} s)`,
    );

    const timing = timedEmbedder(createWorkerEmbedder());
    let indexingTimes: () => { keywordSeconds: number; readySeconds: number } = () => ({
      keywordSeconds: 0,
      readySeconds: 0,
    });
    const library = await openLibrary({
      name: "hard",
      embedder: timing.embedder,
      modelCache: join(config.cacheDir, "models"),
      reranker: createWorkerCrossEncoder(),
      documents: available.map((document) => ({
        key: document.key,
        path: document.path as string,
      })),
      keep: config.keepData,
      log,
      allowUnprocessed: true,
      watch: (core) => {
        indexingTimes = watchIndexing(core);
      },
    });
    timing.stop();
    peakIndexing = Math.max(peak, process.memoryUsage().rss);
    const db = openDatabase(join(library.dataDir, DATABASE_FILE));
    try {
      const count = (sql: string) => db.get<{ count: number }>(sql)?.count ?? 0;
      const embeddedPassages = count(
        "SELECT COUNT(*) AS count FROM passages WHERE deleted_at IS NULL AND embedding IS NOT NULL",
      );
      if (embeddedPassages === 0) {
        throw new Error(
          "No Passage was embedded: the hard tier compares the dense modes, so embeddings must be on for this run (eval/hard/README.md).",
        );
      }
      const readyKeys = new Set(
        [...library.documents]
          .filter(([, document]) => document.status === "ready")
          .map(([key]) => key),
      );
      const ids = new Map([...library.documents].map(([key, document]) => [key, document.id]));

      // What the app stored for each Document, read back to check each Question against.
      const storedCache = new Map<string, StoredText>();
      const stored = (key: string): StoredText | undefined => {
        if (!readyKeys.has(key)) return undefined;
        let text = storedCache.get(key);
        if (!text) {
          const id = ids.get(key) ?? "";
          text = {
            units: db
              .all<{ page: number | null; kind: UnitKind; label: string | null; text: string }>(
                "SELECT page, kind, label, text FROM document_pages WHERE document_id = ? AND deleted_at IS NULL ORDER BY page",
                [id],
              )
              .map((unit) => ({
                ...unit,
                label: unit.label ? (JSON.parse(unit.label) as UnitLabel) : null,
              })),
            passages: db.all<{ pageFrom: number | null; pageTo: number | null; text: string }>(
              "SELECT page_from AS pageFrom, page_to AS pageTo, text FROM passages WHERE document_id = ? AND deleted_at IS NULL",
              [id],
            ),
          };
          storedCache.set(key, text);
        }
        return text;
      };
      const leftOutQuestions: { id: string; reasons: string[] }[] = [];
      const scored: HardQuestion[] = [];
      for (const question of set.questions) {
        const reasons = textProblems(question, stored, manifest);
        if (reasons.length > 0) leftOutQuestions.push({ id: question.id, reasons });
        else scored.push(question);
      }
      log(`${scored.length} Questions to score; ${leftOutQuestions.length} left out`);

      const results = await searchAll(library.core, scored, ids);
      const rerankers: RerankerInfo[] = [];
      for (const search of HARD_RERANKED_SEARCHES) {
        const mode = rerankMode(BUILT_IN_RERANKING_MODEL, search);
        log(`Reranking with ${rerankedLabel(search, BUILT_IN_RERANKING_MODEL.name)}`);
        const reranker = await openReranker(BUILT_IN_RERANKING_MODEL, config.cacheDir, log, mode);
        try {
          await rerankAll(library.core, scored, ids, reranker, results);
          rerankers.push(reranker.info());
        } finally {
          reranker.close();
        }
      }
      // Keyword search's own time at this size, measured on the run's data folder (no app change).
      const keywordTiming = timeKeywordSearch(
        db,
        set.questions.flatMap((question) => [
          question.question,
          ...(question.translatedQuery ? [question.translatedQuery] : []),
        ]),
        KEYWORD_RERANK_DEPTH,
      );
      log(
        `Keyword search at ${keywordTiming.passages} Passages: median ${keywordTiming.bm25.median.toFixed(1)} ms, 95th percentile ${keywordTiming.bm25.p95.toFixed(1)} ms a query (ORDER BY rank: ${keywordTiming.rank.median.toFixed(1)} and ${keywordTiming.rank.p95.toFixed(1)} ms)`,
      );

      if (config.chat) {
        // Answers search as Users' do by default, with embeddings off: keyword search, reranked.
        await library.core.saveEmbeddingProvider({ kind: "off" });
        log("Embeddings off for the Answers, as by default: keyword search, reranked");
      }
      const citations: HardCitations | { skipped: string } = config.chat
        ? await runHardCitations(library, scored, config.chat, config, manifest, log)
        : {
            skipped:
              "no chat model was given. Set INCARNAMIND_EVAL_CHAT_KIND, INCARNAMIND_EVAL_CHAT_MODEL and INCARNAMIND_EVAL_CHAT_KEY (eval/README.md).",
          };

      const byKey = new Map(manifest.documents.map((document) => [document.key, document]));
      const added = [...readyKeys]
        .map((key) => byKey.get(key))
        .filter((each) => each !== undefined);
      const composition = new Map<string, HardReport["library"]["composition"][number]>();
      for (const document of added) {
        const key = `${document.domain} ${document.language} ${document.format}`;
        const row = composition.get(key) ?? {
          domain: document.domain,
          language: document.language,
          format: document.format,
          documents: 0,
        };
        row.documents++;
        composition.set(key, row);
      }
      const counts: HardReport["questions"]["counts"] = {};
      for (const question of set.questions) {
        const byLanguage = counts[question.difficulty] ?? {};
        byLanguage[question.language] = (byLanguage[question.language] ?? 0) + 1;
        counts[question.difficulty] = byLanguage;
      }
      const cpu = cpus();
      const report: HardReport = {
        run: {
          startedAt: started.toISOString(),
          seconds: (Date.now() - started.getTime()) / 1000,
          commit: commit(),
          node: process.version,
          platform: `${platform()} ${arch()}`,
          cpu: `${cpu[0]?.model.trim() ?? "unknown CPU"} (${cpu.length} cores)`,
          memoryBytes: totalmem(),
        },
        library: {
          manifest: MANIFEST,
          documents: manifest.documents.length,
          added: added.length,
          leftOut: [
            ...fetched.documents
              .filter((document) => document.status !== "ready")
              .map((document) => ({
                key: document.key,
                status: document.status,
                reason: document.reason ?? "",
              })),
            ...[...library.documents]
              .filter(([, document]) => document.status !== "ready")
              .map(([key, document]) => ({
                key,
                status: document.status === "no-text" ? ("no-text" as const) : ("failed" as const),
                reason: document.failure?.message ?? document.status,
              })),
          ],
          downloadBytes: downloadBytes(manifest),
          fetchedBytes: fetched.fetchedBytes,
          fetchSeconds: fetched.seconds,
          composition: [...composition.values()],
          sources: Object.entries(manifest.sources).map(([id, source]) => ({
            id,
            name: source.name,
            licence: source.licence,
            terms: source.terms,
            documents: added.filter((document) => document.source === id).length,
          })),
        },
        indexing: {
          passages: library.passageCount,
          embeddedPassages,
          ...indexingTimes(),
          embeddingSeconds: timing.seconds(),
          peakRssBytes: peakIndexing,
          peakRssBytesRun: Math.max(peak, process.memoryUsage().rss),
        },
        questions: {
          source: set.source,
          total: set.questions.length,
          counts,
          leftOut: leftOutQuestions,
        },
        retrieval: {
          topK: TOP_K,
          modes: modeLabels(BUILT_IN_RERANKING_MODEL.name),
          results,
          summary: summariseHard(results),
          rerankers,
          keywordTiming,
        },
        citations,
      };
      log(`Embedding model: ${BUILT_IN_EMBEDDING_MODEL.name}`);
      const dir = await writeHardReports(report, config.resultsDir);
      console.log(hardSummary(report, dir, root));
      expect(results.length, "No Question could be scored (see the report).").toBeGreaterThan(0);
    } finally {
      db.close();
      await library.close();
    }
  } finally {
    clearInterval(sampler);
  }
});
