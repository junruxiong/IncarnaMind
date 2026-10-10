/**
 * The hard tier's report: report.json (everything), report.md (for people)
 * and a summary on the terminal, in a folder of its own under eval/results/.
 * Retrieval per difficulty, domain and language in every mode, the library
 * and what indexing it cost, and, with a chat model, Citations per difficulty.
 * Nothing here gates.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { join, relative } from "node:path";
import { CITATION_OUTCOMES, type CitationOutcome } from "../../lib/citations";
import { reviewerSheet } from "../../lib/report";
import type { RerankerInfo } from "../../lib/rerank";
import {
  GATING_MODE,
  HYBRID_RERANK_MODE,
  RANK_DEPTH,
  type RetrievalMode,
  TOP_K,
} from "../../lib/retrieval";
import type { HardCitations } from "./citations";
import type { FetchStatus } from "./fetch";
import type { KeywordTiming, QueryTiming } from "./keywordTiming";
import {
  DOMAIN_LABELS,
  DOMAINS,
  type Domain,
  type HardFormat,
  type HardLanguage,
} from "./manifest";
import { DIFFICULTIES, type Difficulty } from "./questions";
import {
  HARD_MODES,
  type HardModeSummary,
  type HardQuestionResult,
  type HardTally,
  modeResult,
  SEARCHED_DIFFICULTIES,
  summariseHard,
} from "./scoring";

export interface HardReport {
  run: {
    startedAt: string;
    seconds: number;
    commit: string;
    node: string;
    platform: string;
    cpu: string;
    memoryBytes: number;
  };
  library: {
    manifest: string;
    /** Documents in the manifest, and what became of each that isn't searched. */
    documents: number;
    added: number;
    leftOut: { key: string; status: FetchStatus | "failed" | "no-text"; reason: string }[];
    /** What the manifest downloads in all, and what this run fetched (none when cached). */
    downloadBytes: number;
    fetchedBytes: number;
    fetchSeconds: number;
    /** Documents per domain, language and format, as added. */
    composition: {
      domain: Domain;
      language: HardLanguage;
      format: HardFormat;
      documents: number;
    }[];
    sources: { id: string; name: string; licence: string; terms: string; documents: number }[];
  };
  indexing: {
    passages: number;
    embeddedPassages: number;
    /** From adding the files until keyword search covers every Document. */
    keywordSeconds: number;
    /** From adding the files until every Document is embedded. */
    readySeconds: number;
    /** Time the embedding model spent embedding Passages. */
    embeddingSeconds: number;
    /** The process's peak resident memory while indexing, and over the whole run (worker threads included). */
    peakRssBytes: number;
    peakRssBytesRun: number;
  };
  questions: {
    source: string;
    total: number;
    /** Per difficulty and language. */
    counts: Partial<Record<Difficulty, Partial<Record<HardLanguage, number>>>>;
    /** Questions left out: their Document isn't in the library, or their quote isn't where expected. */
    leftOut: { id: string; reasons: string[] }[];
  };
  retrieval: {
    topK: number;
    modes: { mode: RetrievalMode; label: string }[];
    results: HardQuestionResult[];
    summary: Partial<Record<RetrievalMode, HardModeSummary>>;
    rerankers: RerankerInfo[];
    /** Keyword search's own time per query at this size, ordered as the app orders it and by FTS5's rank. */
    keywordTiming: KeywordTiming | null;
  };
  citations: HardCitations | { skipped: string };
}

const fraction = ({ hits, total }: { hits: number; total: number }) =>
  total === 0 ? "–" : `${hits}/${total}`;
const percent = (value: number | null) => (value === null ? "–" : `${(value * 100).toFixed(0)}%`);
const share = ({ hits, total }: { hits: number; total: number }) =>
  total === 0 ? "–" : `${fraction({ hits, total })} (${percent(hits / total)})`;
const mb = (bytes: number) => `${(bytes / 1e6).toFixed(0)} MB`;
const gb = (bytes: number) => `${(bytes / 1e9).toFixed(2)} GB`;

export const DIFFICULTY_LABELS: Record<Difficulty, string> = {
  easy: "Easy (the Documents' own words)",
  paraphrase: "Paraphrase",
  "table-number": "Table number",
  "multi-page": "Multi-page",
  "cross-document": "Cross-document",
  unanswerable: "Unanswerable",
  "cross-lingual": "Cross-lingual (with a translated second query)",
  "near-duplicate": "Near-duplicate",
};

const labelOf = (report: HardReport, mode: RetrievalMode) =>
  report.retrieval.modes.find((each) => each.mode === mode)?.label ?? mode;

/** The cross-lingual Questions scored on their own query alone, without the translation. */
export function ownQueryOnly(results: readonly HardQuestionResult[]): HardQuestionResult[] {
  return results
    .filter((result) => result.difficulty === "cross-lingual")
    .map((result) => ({
      ...result,
      modes: Object.fromEntries(
        Object.entries(result.modes).map(([mode, each]) => [mode, modeResult(each?.ranks ?? [])]),
      ),
    }));
}

function modesTable(
  report: HardReport,
  rows: readonly {
    label: string;
    tally: (summary: HardModeSummary, mode: RetrievalMode) => HardTally | undefined;
  }[],
): string[] {
  const modes = report.retrieval.modes.filter(({ mode }) => report.retrieval.summary[mode]);
  const lines = [
    `| | ${modes.map(({ label }) => label).join(" | ")} |`,
    `|---|${modes.map(() => "---|").join("")}`,
  ];
  for (const row of rows) {
    const cells = modes.map(({ mode }) => {
      const summary = report.retrieval.summary[mode];
      const tally = summary && row.tally(summary, mode);
      return tally ? share(tally) : "–";
    });
    if (cells.every((cell) => cell === "–")) continue;
    lines.push(`| ${row.label} | ${cells.join(" | ")} |`);
  }
  return lines;
}

function retrievalSection(report: HardReport): string[] {
  const { retrieval } = report;
  const own = summariseHard(ownQueryOnly(retrieval.results));
  const multi = SEARCHED_DIFFICULTIES.filter(
    (difficulty) => difficulty === "multi-page" || difficulty === "cross-document",
  );
  const lines = [
    "## Retrieval",
    "",
    `A Question is a hit when every Passage its answer needs is in the top ${retrieval.topK}, each meeting the gating set's hit rule (the expected Document, covering the expected pages, holding the expected quote). Unanswerable Questions have nothing to find and are scored by the Citation part. A cross-lingual Question is also searched with its translation into its Documents' language, written by hand, as an Answer is told to search again: a Passage either search finds counts. Embeddings are on for this run, so the dense modes can be compared; ${labelOf(report, GATING_MODE)} is what the search Tool does by default.`,
    "",
    "### By difficulty",
    "",
    ...modesTable(report, [
      ...SEARCHED_DIFFICULTIES.map((difficulty) => ({
        label: DIFFICULTY_LABELS[difficulty],
        tally: (summary: HardModeSummary) => summary.byDifficulty[difficulty],
      })),
      {
        label: "Cross-lingual, its own query only",
        tally: (_summary: HardModeSummary, mode: RetrievalMode) =>
          own[mode]?.byDifficulty["cross-lingual"],
      },
      { label: "**All**", tally: (summary: HardModeSummary) => summary.all },
    ]),
    "",
    `Partly found (some, not all, of the Passages needed in the top ${retrieval.topK}): ${multi
      .map((difficulty) => {
        const cells = retrieval.modes
          .map(({ mode, label }) => {
            const tally = retrieval.summary[mode]?.byDifficulty[difficulty];
            return tally ? `${label} ${tally.partial}` : null;
          })
          .filter(Boolean);
        return `${difficulty}: ${cells.join(", ")}`;
      })
      .join("; ")}.`,
    "",
    "### By domain",
    "",
    ...modesTable(
      report,
      DOMAINS.map((domain) => ({
        label: DOMAIN_LABELS[domain],
        tally: (summary: HardModeSummary) => summary.byDomain[domain],
      })),
    ),
    "",
    "### By language of the Question",
    "",
    ...modesTable(report, [
      { label: "English", tally: (summary: HardModeSummary) => summary.byLanguage.en },
      { label: "Chinese", tally: (summary: HardModeSummary) => summary.byLanguage.zh },
    ]),
    "",
    "### Domain × difficulty",
    "",
  ];
  for (const { mode, label } of retrieval.modes) {
    const summary = retrieval.summary[mode];
    if (!summary) continue;
    const domains = DOMAINS.filter((domain) => summary.byDomain[domain]);
    lines.push(
      `**${label}**`,
      "",
      `| | ${domains.map((domain) => DOMAIN_LABELS[domain]).join(" | ")} | All |`,
      `|---|${domains.map(() => "---|").join("")}---|`,
      ...SEARCHED_DIFFICULTIES.filter((difficulty) => summary.byDifficulty[difficulty]).map(
        (difficulty) =>
          `| ${difficulty} | ${domains
            .map((domain) => {
              const tally = summary.byDomainAndDifficulty[`${domain} ${difficulty}`];
              return tally ? fraction(tally) : "–";
            })
            .join(" | ")} | ${fraction(summary.byDifficulty[difficulty] as HardTally)} |`,
      ),
      `| **All** | ${domains.map((domain) => fraction(summary.byDomain[domain] as HardTally)).join(" | ")} | ${fraction(summary.all)} |`,
      "",
    );
  }
  lines.push(...searchTime(report));
  const rank = (value: number | null | undefined) =>
    value === null || value === undefined ? "–" : value <= TOP_K ? String(value) : `(${value})`;
  lines.push(
    "### Per Question",
    "",
    `The rank of each Passage the answer needs, in the order expected; a rank in brackets is a near miss (6 to ${RANK_DEPTH}), "–" none in the top ${RANK_DEPTH}. "a / b": the Question's own query, then its translation.`,
    "",
    `| Question | Difficulty | Domain | ${retrieval.modes.map(({ label }) => label).join(" | ")} | Text |`,
    `|---|---|---|${retrieval.modes.map(() => "---|").join("")}---|`,
    ...retrieval.results.map((result) => {
      const cells = retrieval.modes.map(({ mode }) => {
        const each = result.modes[mode];
        if (!each) return "–";
        const own = each.ranks.map(rank).join(", ");
        return each.translatedRanks ? `${own} / ${each.translatedRanks.map(rank).join(", ")}` : own;
      });
      return `| ${result.id} | ${result.difficulty} | ${result.domain} | ${cells.join(" | ")} | ${result.question.replace(/\|/g, "\\|")} |`;
    }),
    "",
  );
  return lines;
}

/** Keyword search's own time per query, beside the reranker's per search. */
function searchTime(report: HardReport): string[] {
  const { keywordTiming, rerankers } = report.retrieval;
  if (!keywordTiming && rerankers.length === 0) return [];
  const ms = (value: number) => `${value < 10 ? value.toFixed(1) : value.toFixed(0)} ms`;
  const row = (label: string, timing: QueryTiming, extra = "") =>
    `| ${label} | ${ms(timing.median)} | ${ms(timing.p95)} | ${ms(timing.mean)} | ${ms(timing.max)} | ${extra} |`;
  return [
    "### Search time",
    "",
    ...(keywordTiming
      ? [
          `Keyword search alone, at ${keywordTiming.passages} Passages: the app's SQL (\`keywordSearch\`) keeping the best ${keywordTiming.limit}, over ${keywordTiming.bm25.queries} queries (the Questions and their translations), each timed once after a warm-up pass. FTS5 scores every Passage a query's words match before keeping the best: it has no top-k pruning. Both orders gave the same Passages for ${keywordTiming.sameResults} of ${keywordTiming.bm25.queries} queries.`,
          "",
        ]
      : []),
    "| Step | Median | 95th percentile | Mean | Slowest | |",
    "|---|---|---|---|---|---|",
    ...(keywordTiming
      ? [
          row("Keyword search, `ORDER BY bm25(passages_fts)` (the app's)", keywordTiming.bm25),
          row("Keyword search, `ORDER BY rank` (bm25() with the same weights)", keywordTiming.rank),
        ]
      : []),
    ...rerankers.map((each) =>
      row(
        `Reranking for ${labelOf(report, each.mode as RetrievalMode)}`,
        each.latency,
        `loading ${each.loadSeconds.toFixed(1)} s`,
      ),
    ),
    "",
  ];
}

const OUTCOME_LABELS: Record<CitationOutcome, string> = {
  found: '"Quote found"',
  "false-not-found": 'False "not found"',
  "wrong-page": '"Not found": other pages',
  "not-in-document": '"Not found": not in the Document',
  "page-range": '"Not found": page-range rule',
  "cant-check": '"Can\'t check"',
};

function citationsSection(report: HardReport): string[] {
  const { citations } = report;
  if ("skipped" in citations) return ["## Citations", "", `Skipped: ${citations.skipped}`, ""];
  const columns = DIFFICULTIES.filter((difficulty) => citations.byDifficulty[difficulty]);
  const row = (label: string, cell: (difficulty: Difficulty) => string) =>
    `| ${label} | ${columns.map(cell).join(" | ")} |`;
  const of = (difficulty: Difficulty) => citations.byDifficulty[difficulty];
  const ofAnswers = (part: number, difficulty: Difficulty) =>
    `${part}/${of(difficulty)?.answers ?? 0} (${percent(part / Math.max(1, of(difficulty)?.answers ?? 0))})`;
  return [
    "## Citations",
    "",
    `- Model: \`${citations.run.model}\`, ${citations.run.service ? `sent to ${citations.run.service}` : "on this computer"}. Each Question asked once, in a Mind of its own, with embeddings off as Users have them by default (keyword search, reranked). Reported, never gating.`,
    '- Figures as for the gating set (`eval/README.md`). "Cites a Document it needs": an Answer with a found Citation of an expected Document; for a near-duplicate Question, of the right version. For an unanswerable Question, the share answered without a Citation is the one that matters.',
    "",
    `| | ${columns.join(" | ")} |`,
    `|---|${columns.map(() => "---|").join("")}`,
    row("Answers (failed)", (d) => `${of(d)?.answers ?? 0} (${of(d)?.failedAnswers ?? 0})`),
    row("Citations", (d) => String(of(d)?.citations ?? 0)),
    ...CITATION_OUTCOMES.map((outcome) =>
      row(OUTCOME_LABELS[outcome], (d) => {
        const summary = of(d);
        return summary && summary.citations > 0
          ? `${summary.outcomes[outcome]} (${percent(summary.outcomes[outcome] / summary.citations)})`
          : "–";
      }),
    ),
    row("Sentences with a Citation", (d) => percent(of(d)?.coverage ?? null)),
    row("Cites a Document it needs", (d) =>
      d === "unanswerable" ? "–" : ofAnswers(of(d)?.citingExpected ?? 0, d),
    ),
    row("Cites another version (near-duplicate)", (d) =>
      d === "near-duplicate" ? ofAnswers(of(d)?.citingOtherVersion ?? 0, d) : "–",
    ),
    row("**Answered without a Citation**", (d) => ofAnswers(of(d)?.withoutCitation ?? 0, d)),
    "",
  ];
}

export function markdownReport(report: HardReport): string {
  const { run, library, indexing, questions } = report;
  const domainRows = DOMAINS.map((domain) => {
    const of = (language: HardLanguage) =>
      library.composition
        .filter((row) => row.domain === domain && row.language === language)
        .reduce((sum, row) => sum + row.documents, 0);
    const formats = [
      ...new Set(
        library.composition.filter((row) => row.domain === domain).map((row) => row.format),
      ),
    ];
    return `| ${DOMAIN_LABELS[domain]} | ${of("en")} | ${of("zh")} | ${formats.join(", ")} |`;
  });
  const lines = [
    "# IncarnaMind evaluation: the hard tier",
    "",
    "Reported, never gating (`eval/hard/README.md`).",
    "",
    `- Run: ${run.startedAt}, ${(run.seconds / 60).toFixed(0)} min, commit ${run.commit}`,
    `- Machine: ${run.platform}, ${run.cpu}, ${gb(run.memoryBytes)} of memory, Node ${run.node}`,
    "",
    "## Library",
    "",
    `\`${library.manifest}\`: ${library.documents} Documents, ${library.added} added. The manifest downloads ${gb(library.downloadBytes)} in all; this run fetched ${mb(library.fetchedBytes)} in ${library.fetchSeconds.toFixed(0)} s.`,
    "",
    "| Domain | English | Chinese | Formats |",
    "|---|---|---|---|",
    ...domainRows,
    "",
    "| Source | Licence or terms | Documents |",
    "|---|---|---|",
    ...library.sources.map(
      (source) => `| ${source.name} | [${source.licence}](${source.terms}) | ${source.documents} |`,
    ),
    "",
    ...(library.leftOut.length
      ? [
          "Left out:",
          "",
          ...library.leftOut.map((each) => `- ${each.key} (${each.status}): ${each.reason}`),
          "",
        ]
      : []),
    "## Indexing",
    "",
    `- ${indexing.passages} Passages, ${indexing.embeddedPassages} of them embedded.`,
    `- Keyword search covered every Document after ${indexing.keywordSeconds.toFixed(0)} s (text extraction, Passages and the keyword index); every Document was embedded after ${indexing.readySeconds.toFixed(0)} s.`,
    `- The embedding model spent ${indexing.embeddingSeconds.toFixed(0)} s embedding Passages${indexing.embeddingSeconds > 0 ? `: ${(indexing.embeddedPassages / indexing.embeddingSeconds).toFixed(1)} a second` : ""}.`,
    `- Peak memory (resident, worker threads included): ${gb(indexing.peakRssBytes)} while indexing, ${gb(indexing.peakRssBytesRun)} over the run.`,
    "",
    "## Questions",
    "",
    `\`${questions.source}\`: ${questions.total} Questions.`,
    "",
    "| Difficulty | English | Chinese |",
    "|---|---|---|",
    ...DIFFICULTIES.map(
      (difficulty) =>
        `| ${difficulty} | ${questions.counts[difficulty]?.en ?? 0} | ${questions.counts[difficulty]?.zh ?? 0} |`,
    ),
    "",
    ...(questions.leftOut.length
      ? [
          "Left out, not scored:",
          "",
          ...questions.leftOut.map((each) => `- ${each.id}: ${each.reasons.join(" ")}`),
          "",
        ]
      : []),
    ...retrievalSection(report),
    ...citationsSection(report),
  ];
  return lines.join("\n");
}

/** The summary printed at the end of a run. */
export function hardSummary(report: HardReport, reportDir: string, root: string): string {
  const { retrieval } = report;
  const modes = retrieval.modes.filter(({ mode }) => retrieval.summary[mode]);
  const width = Math.max(...SEARCHED_DIFFICULTIES.map((each) => each.length), 4) + 2;
  const lines = [
    "",
    `Hard tier, top ${retrieval.topK}: hits per difficulty (reported, never gating)`,
    `  ${"".padEnd(width)}${modes.map(({ label }) => label.padEnd(26)).join("")}`,
  ];
  for (const difficulty of [...SEARCHED_DIFFICULTIES, "all"] as const) {
    const cells = modes.map(({ mode }) => {
      const summary = retrieval.summary[mode];
      const tally = difficulty === "all" ? summary?.all : summary?.byDifficulty[difficulty];
      return (tally ? share(tally) : "–").padEnd(26);
    });
    lines.push(`  ${difficulty.padEnd(width)}${cells.join("")}`);
  }
  const { indexing } = report;
  lines.push(
    "",
    `  ${report.library.added} Documents, ${indexing.passages} Passages; keyword search ready after ${indexing.keywordSeconds.toFixed(0)} s, embedded after ${indexing.readySeconds.toFixed(0)} s; peak memory ${gb(indexing.peakRssBytesRun)}`,
  );
  const { citations } = report;
  if (!("skipped" in citations)) {
    const unanswerable = citations.byDifficulty.unanswerable;
    lines.push(
      `  Citations (${citations.run.model}): ${DIFFICULTIES.filter((d) => citations.byDifficulty[d])
        .map((d) => `${d} found ${percent(citations.byDifficulty[d]?.foundShare ?? null)}`)
        .join(
          ", ",
        )}${unanswerable ? `; unanswerable answered without a Citation ${unanswerable.withoutCitation}/${unanswerable.answers}` : ""}`,
    );
  }
  lines.push(`Reports: ${relative(root, reportDir)}/`, "");
  return lines.join("\n");
}

/** Writes the reports into a new folder and returns its path. */
export async function writeHardReports(report: HardReport, resultsDir: string): Promise<string> {
  const dir = join(resultsDir, `${report.run.startedAt.replace(/[:.]/g, "-")}-hard`);
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
  await writeFile(join(dir, "report.md"), markdownReport(report));
  if (!("skipped" in report.citations)) {
    await writeFile(join(dir, "reviewer-sheet.csv"), reviewerSheet(report.citations.run));
  }
  return dir;
}

/** The default modes, labelled as people read them. */
export const modeLabels = (keywordRerank: string, hybridRerank: string) =>
  HARD_MODES.map((mode) => ({
    mode,
    label: mode === GATING_MODE ? keywordRerank : mode === HYBRID_RERANK_MODE ? hybridRerank : mode,
  }));
