/**
 * `npm run eval:grouping`: the check before building the grouping
 * (eval/grouping/README.md, #51; "Check before building the grouping" in
 * docs/designs/library-structure-view.md).
 *
 * On the fixture set: the Documents are added to a temporary data folder and
 * processed with the built-in model, as the retrieval evaluation does; each
 * of the three Document vectors (R3) groups them with the core's grouping
 * functions; the variant is chosen on the choosing set and the pass bars are
 * scored on the held-out set (O8a); then the incremental cases, the
 * classifier fallback when one is given (R0), and the timings (R1, R2, O5).
 *
 * With INCARNAMIND_EVAL_GROUPING_FOLDER, it groups the User's own library
 * instead and writes the founder's 30-Document sample for them to judge.
 *
 * It runs under Vitest only for its TypeScript and worker-thread handling
 * (eval/grouping/vitest.config.ts); neither `npm test` nor `npm run eval`
 * runs it.
 */
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";
import type { Embedder } from "../../src/core";
import { ollamaBaseUrl, serviceFor } from "../../src/core/providers/kinds";
import { createAiSdkChatModel } from "../../src/core/providers/models";
import { DEFAULT_SEED, topicCount } from "../../src/core/topics/grouping";
import { type EvalConfig, readConfig } from "../lib/config";
import { createWorkerEmbedder } from "../lib/embedder";
import { openLibrary } from "../lib/library";
import { createLog, type Log } from "../lib/log";
import {
  chooseVariant,
  evaluateVariant,
  fixturesReport,
  founderSample,
  rankClassifiers,
  scoreClassifier,
} from "./lib/check";
import {
  chatTopicClassifier,
  chatTopicProposer,
  runClassifier,
  systemOneTopicClassifier,
  type TopicClassifier,
} from "./lib/classifiers";
import { type FounderSettings, type GroupingConfig, readGroupingConfig } from "./lib/config";
import { founderSheet, libraryFiles } from "./lib/founder";
import { runIncremental } from "./lib/incremental";
import {
  type Classifiers,
  type FixturesReport,
  type FounderReport,
  fixturesMarkdown,
  fixturesSummary,
  founderMarkdown,
  runInfo,
  writeGroupingReports,
} from "./lib/report";
import { type GroupingSet, loadGroupingSet } from "./lib/set";
import { timeKMeans, timeMeans } from "./lib/timing";
import {
  builtInTextEmbedder,
  type Corpus,
  nameFree,
  namesIncluded,
  namesRemoved,
  readCorpus,
  type VariantVectors,
} from "./lib/variants";

const root = fileURLToPath(new URL("../..", import.meta.url));

/** All three Document vectors for the corpus, with the built-in model. */
async function buildVariants(
  corpus: Corpus,
  embedder: Embedder,
  dataDir: string,
  log: Log,
): Promise<VariantVectors[]> {
  const embed = await builtInTextEmbedder(embedder, dataDir);
  const included = namesIncluded(corpus);
  log(`Names included: ${included.vectors.size} Document means from the vector index`);
  const removed = await namesRemoved(corpus, included, embed);
  log(`Names removed: ${removed.embeddings} names embedded in ${removed.seconds.toFixed(1)} s`);
  const free = await nameFree(corpus, embed, log);
  log(`Name-free: ${free.embeddings} Passages embedded again in ${free.seconds.toFixed(0)} s`);
  return [included, removed, free];
}

/** R0's fallback, when a chat model is given to propose the Topic list. */
async function classifierFallback(
  config: EvalConfig,
  grouping: GroupingConfig,
  set: GroupingSet,
  corpus: Corpus,
  minTopics: number,
  log: Log,
): Promise<Classifiers> {
  const chat = config.chat;
  if (!chat) {
    const given = [grouping.clef && "Clef-Flash", grouping.jev && "Jev"].filter(Boolean);
    return {
      skipped: given.length
        ? `${given.join(" and ")} given, but no chat model to propose the Topic list: set INCARNAMIND_EVAL_CHAT_KIND and INCARNAMIND_EVAL_CHAT_MODEL too (eval/grouping/README.md).`
        : "no classifier was given. A chat model (INCARNAMIND_EVAL_CHAT_*) proposes the Topic list and classifies; Clef-Flash in Ollama (INCARNAMIND_EVAL_CLEF_URL) and Jev (INCARNAMIND_EVAL_JEV_KEY) are measured beside it (eval/grouping/README.md).",
    };
  }
  const baseUrl = chat.kind === "ollama" ? ollamaBaseUrl(chat.baseUrl ?? undefined) : chat.baseUrl;
  const model = createAiSdkChatModel({
    kind: chat.kind,
    baseUrl,
    apiKey: chat.apiKey,
    modelId: chat.modelId,
  });
  const chatName = `${chat.kind}/${chat.modelId}`;
  const count = Math.max(topicCount(corpus.documents.length), minTopics);
  const started = performance.now();
  let topics: string[];
  try {
    topics = await chatTopicProposer(model, chatName).propose(
      corpus.documents.map((document) => document.name),
      count,
    );
  } catch (error) {
    return {
      skipped: `${chatName} couldn't propose the Topic list: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
  const proposeSeconds = (performance.now() - started) / 1000;
  log(`${chatName} proposed ${topics.length} Topics: ${topics.join("; ")}`);
  const classifiers: TopicClassifier[] = [
    ...(grouping.clef ? [systemOneTopicClassifier(grouping.clef)] : []),
    ...(grouping.jev ? [systemOneTopicClassifier(grouping.jev)] : []),
    chatTopicClassifier(model, chatName, serviceFor(chat.kind, baseUrl) === null),
  ];
  const keys = corpus.documents.map((document) => document.key);
  const runs = [];
  for (const classifier of classifiers) {
    const run = await runClassifier(classifier, topics, corpus.documents, log);
    log(`${run.name}: ${run.seconds.toFixed(0)} s, ${run.failed} failed`);
    runs.push(scoreClassifier(set, keys, run));
  }
  return { proposer: chatName, topics, proposeSeconds, runs, ranking: rankClassifiers(runs) };
}

async function fixturesRun(
  config: EvalConfig,
  grouping: GroupingConfig,
  started: Date,
  log: Log,
): Promise<FixturesReport> {
  const set = loadGroupingSet(root);
  const minTopics = Object.keys(set.subjects).length;
  log(`${set.documents.length} Documents on ${minTopics} subjects (${set.source})`);
  const embedder = createWorkerEmbedder();
  const library = await openLibrary({
    name: "grouping",
    embedder,
    modelCache: join(config.cacheDir, "models"),
    documents: set.documents,
    keep: config.keepData,
    log,
  });
  let corpus: Corpus | undefined;
  let parts: Omit<Parameters<typeof fixturesReport>[0], "timing" | "run">;
  try {
    corpus = readCorpus(library.dataDir, library.documents);
    const built = await buildVariants(corpus, embedder, library.dataDir, log);
    const names = new Map(corpus.documents.map((document) => [document.key, document.name]));
    const variants = built.map((each) => evaluateVariant(set, each, names, minTopics));
    // The incremental cases use the variant the report chooses.
    const chosen = chooseVariant(variants);
    log(`Chosen on the choosing set: ${chosen.id}, by ${chosen.reason}`);
    const incremental = runIncremental(
      set,
      (built.find((each) => each.id === chosen.id) as VariantVectors).vectors,
      { seed: DEFAULT_SEED },
    );
    const classifiers = await classifierFallback(config, grouping, set, corpus, minTopics, log);
    parts = {
      set,
      names,
      passages: new Map(
        corpus.documents.map((document) => [document.key, document.passages.length]),
      ),
      processing: { passages: library.passageCount, seconds: library.processingSeconds },
      variants,
      incremental,
      classifiers,
    };
  } finally {
    corpus?.close();
    await library.close();
  }

  // After the library is closed, so the embedding worker doesn't compete for the CPU.
  let timing: FixturesReport["timing"] = { skipped: "INCARNAMIND_EVAL_GROUPING_TIMING=0" };
  if (grouping.timing) {
    log("Timing k-means at 5,000 Documents");
    const kMeans = timeKMeans();
    log(`k-means: ${kMeans.seconds.toFixed(2)} s`);
    log("Timing the means at 100,000 Passages (writing a generated database first)");
    const means = await timeMeans();
    log(`Means: cold ${means.coldMs.map((value) => value.toFixed(0)).join(", ")} ms`);
    timing = { kMeans, means };
  }
  return fixturesReport({ ...parts, timing, run: runInfo(root, started) });
}

async function founderRun(
  config: EvalConfig,
  founder: FounderSettings,
  started: Date,
  log: Log,
): Promise<{ report: FounderReport; sheet: string }> {
  const files = libraryFiles(founder.folder, founder.limit, founder.seed);
  if (files.length === 0) throw new Error(`No files IncarnaMind can index in ${founder.folder}.`);
  log(`${files.length} files from ${founder.folder} (read only)`);
  const embedder = createWorkerEmbedder();
  const library = await openLibrary({
    name: "founder",
    embedder,
    modelCache: join(config.cacheDir, "models"),
    documents: files,
    keep: config.keepData,
    log,
    allowUnprocessed: true,
  });
  let corpus: Corpus | undefined;
  try {
    corpus = readCorpus(library.dataDir, library.documents);
    const built = await buildVariants(corpus, embedder, library.dataDir, log);
    const names = new Map(corpus.documents.map((document) => [document.key, document.name]));
    const { rows, letters, sample } = founderSample(built, names, founder.seed);
    const report: FounderReport = {
      kind: "founder",
      run: runInfo(root, started),
      folder: founder.folder,
      files: files.length,
      ready: corpus.documents.length,
      notReady: files.length - corpus.documents.length,
      passages: library.passageCount,
      processingSeconds: library.processingSeconds,
      letters,
      sample,
      sheet: "founder-sample.csv",
    };
    return { report, sheet: founderSheet(rows) };
  } finally {
    corpus?.close();
    await library.close();
  }
}

test("grouping check", async () => {
  const started = new Date();
  const log = createLog(started.getTime());
  const config = readConfig(root);
  const grouping = readGroupingConfig();
  log(`Embedding model cache: ${config.cacheDir}`);

  if (grouping.founder) {
    const { report, sheet } = await founderRun(config, grouping.founder, started, log);
    const dir = await writeGroupingReports(
      config.resultsDir,
      "grouping-founder",
      report.run.startedAt,
      {
        "founder-sample.csv": sheet,
        "report.json": `${JSON.stringify(report, null, 2)}\n`,
        "report.md": founderMarkdown(report),
      },
    );
    console.log(
      `\nThe founder's sample: ${report.sample} Documents of ${report.ready}, three groupings each.\nJudge ${join(dir, report.sheet)} (eval/grouping/README.md).\n`,
    );
    return;
  }

  const report = await fixturesRun(config, grouping, started, log);
  const dir = await writeGroupingReports(config.resultsDir, "grouping", report.run.startedAt, {
    "report.json": `${JSON.stringify(report, null, 2)}\n`,
    "report.md": fixturesMarkdown(report),
  });
  console.log(fixturesSummary(report, dir, root));
  expect(report.failures, "The grouping check missed its bars (see the report).").toEqual([]);
});
