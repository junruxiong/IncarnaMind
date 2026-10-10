/**
 * The evaluation's outputs: report.json (everything), report.md (for people),
 * reviewer-sheet.csv (found quotes, for judging whether they support their
 * sentence; reviewer-sheet-formats.csv for the every-format set's), and a
 * summary on the terminal.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { join, relative } from "node:path";
import type { SearchMode } from "../../src/core";
import {
  CITATION_OUTCOMES,
  CITATION_TARGETS,
  type CitationGroup,
  type CitationOutcome,
  type CitationRun,
  type GroupSummary,
} from "./citations";
import type { FormatsReport } from "./formats";
import {
  type CandidateCounts,
  GATING_LABEL,
  GATING_MODE,
  HYBRID,
  KEYWORD_RERANK_DEPTH,
  KEYWORD_RERANK_LABEL,
  KEYWORD_RERANK_MODE,
  type ModeResult,
  type ModeSummary,
  type QuestionResult,
  RANK_DEPTH,
  RERANK_PER_LIST,
  type RetrievalMode,
  type RetrievalRun,
  rerankedLabel,
  rerankedSearchOf,
  SEARCH_MODES,
  type Tally,
} from "./retrieval";

export interface EvalReport {
  result: "pass" | "fail";
  /** Every target missed. */
  failures: string[];
  run: {
    startedAt: string;
    seconds: number;
    commit: string;
    node: string;
    platform: string;
    cpu: string;
  };
  evaluationSet: {
    source: string;
    hitRule: string;
    questions: { gating: { en: number; zh: number }; crossLingual: number };
  };
  documents: { key: string; name: string; pageCount: number | null }[];
  retrieval: {
    topK: number;
    gatingMode: RetrievalMode;
    /** The built-in model first; then a cloud model, when one was given. */
    runs: RetrievalRun[];
  };
  citations: CitationRun | { skipped: string };
  /** The every-format set (#70): reported per format, never gating. */
  formats?: FormatsReport;
}

const fraction = ({ hits, total }: Tally) => `${hits}/${total}`;
const percent = (value: number | null) => (value === null ? "–" : `${(value * 100).toFixed(1)}%`);
const pages = (from: number | null, to: number | null) =>
  from === null ? "" : to !== null && to !== from ? `${from}–${to}` : `${from}`;

const GROUPS: readonly [CitationGroup, string][] = [
  ["en", "English"],
  ["zh", "Chinese"],
  ["crossLingual", "Cross-lingual"],
];

const OUTCOME_LABELS: Record<CitationOutcome, string> = {
  found: '"Quote found"',
  "false-not-found": 'False "not found" (quote on the cited pages)',
  "wrong-page": '"Not found": quote on other pages',
  "not-in-document": '"Not found": quote not in the Document',
  "page-range": '"Not found": breaks the page-range rule',
  "cant-check": '"Can\'t check"',
};

/** A run's modes as the report lists them: the core's search modes, then the reranked ones. */
const modesIn = (run: RetrievalRun): RetrievalMode[] =>
  (Object.keys(run.summary) as RetrievalMode[]).sort(
    (a, b) =>
      (SEARCH_MODES.indexOf(a as SearchMode) + 1 || 99) -
      (SEARCH_MODES.indexOf(b as SearchMode) + 1 || 99),
  );

/** "hybrid", or "hybrid + <reranking model>" or "keyword + <reranking model>" for a reranked mode. */
function modeLabel(run: RetrievalRun, mode: RetrievalMode): string {
  const reranker = run.rerankers?.find((each) => each.mode === mode);
  const search = rerankedSearchOf(mode);
  if (reranker && search) return rerankedLabel(search, reranker.name);
  if (mode === GATING_MODE) return GATING_LABEL;
  return mode === KEYWORD_RERANK_MODE ? KEYWORD_RERANK_LABEL : mode;
}

function retrievalTable(runs: readonly RetrievalRun[]): string[] {
  const lines = [
    "| Embedding model | Mode | English | Chinese | Gating set | Cross-lingual | Cross-lingual, with a translated second query |",
    "|---|---|---|---|---|---|---|",
  ];
  for (const run of runs) {
    for (const mode of modesIn(run)) {
      const summary = run.summary[mode] as ModeSummary | undefined;
      if (!summary) continue;
      const gating = run.gating && mode === GATING_MODE;
      const label = gating ? `**${modeLabel(run, mode)} (gating)**` : modeLabel(run, mode);
      const translated = summary.crossLingualTranslated;
      const cell = (tally: Tally) => (gating ? `**${fraction(tally)}**` : fraction(tally));
      lines.push(
        `| ${run.embedding} | ${label} | ${cell(summary.en)} | ${cell(summary.zh)} | ${cell(summary.core)} | ${fraction(summary.crossLingual)} | ${translated ? fraction(translated) : "–"} |`,
      );
    }
  }
  return lines;
}

/** How many Passages a reranker saw per search, in hybrid's reranked modes or in keyword + rerank. */
export const candidatesLine = (counts: CandidateCounts) =>
  counts.search === "keyword"
    ? `Candidates per keyword + rerank search (keyword search's top ${counts.perList}, no vector search): ${counts.mean.toFixed(1)} on average, from ${counts.min} to ${counts.max}, over ${counts.searches} searches.`
    : `Candidates per reranked search (keyword search's top ${counts.perList} and vector search's top ${counts.perList}, each Passage once): ${counts.mean.toFixed(1)} on average, from ${counts.min} to ${counts.max}, over ${counts.searches} searches.`;

/** A run's candidate counts, hybrid's reranked modes' first. */
const candidateCountsOf = (run: RetrievalRun): CandidateCounts[] =>
  [run.rerankCandidates, run.keywordRerankCandidates].filter(
    (counts): counts is CandidateCounts => counts !== undefined,
  );

/** What each reranking candidate costs in each reranked mode: its download, and the time it adds to a search. */
function rerankerTable(run: RetrievalRun): string[] {
  if (!run.rerankers?.length) return [];
  const ms = (value: number) => `${value.toFixed(0)} ms`;
  return [
    "| Reranked mode | Licence | Download | Per search: mean | median | 95th percentile | slowest | Loading |",
    "|---|---|---|---|---|---|---|---|",
    ...run.rerankers.map(
      (each) =>
        `| ${modeLabel(run, each.mode as RetrievalMode)} | ${each.licence} | ${(each.downloadBytes / 1e6).toFixed(0)} MB | ${ms(each.latency.mean)} | ${ms(each.latency.median)} | ${ms(each.latency.p95)} | ${ms(each.latency.max)} | ${each.loadSeconds.toFixed(1)} s |`,
    ),
  ];
}

function perQuestionTable(runs: readonly RetrievalRun[]): string[] {
  const columns = runs.flatMap((run) => modesIn(run).map((mode) => ({ run, mode })));
  const header = columns.map(({ run, mode }) =>
    runs.length > 1
      ? `${modeLabel(run, mode)} (${run.gating ? "built-in" : "cloud"})`
      : modeLabel(run, mode),
  );
  const lines = [
    `| Question | ${header.join(" | ")} | Text |`,
    `|---|${columns.map(() => "---|").join("")}---|`,
  ];
  const rank = (result: ModeResult | undefined) => {
    if (!result?.rank) return "–";
    return result.hit ? String(result.rank) : `(${result.rank})`;
  };
  const first = runs[0] as RetrievalRun;
  first.questions.forEach((question, index) => {
    const cells = columns.map(({ run, mode }) => {
      const result = run.questions[index];
      const translated = result?.translated?.[mode];
      return translated
        ? `${rank(result?.modes[mode])} / ${rank(translated)}`
        : rank(result?.modes[mode]);
    });
    const id = question.crossLingual ? `${question.id} (cross-lingual)` : question.id;
    const text = question.translatedQuery
      ? `${question.question} / ${question.translatedQuery}`
      : question.question;
    lines.push(`| ${id} | ${cells.join(" | ")} | ${text} |`);
  });
  return lines;
}

/**
 * For each miss of the gating mode: what the top 5 were, and what each lacked.
 * `unit` names what the pages are: "p." for PDFs, "Unit" for any format.
 */
function misses(run: RetrievalRun, unit = "p."): string[] {
  const lines: string[] = [];
  const describe = (question: QuestionResult) =>
    (question.modes[GATING_MODE]?.top ?? []).map((passage, index) => {
      const lacks = [
        passage.rightDocument ? null : "other Document",
        passage.coversPages ? null : `other ${unit === "p." ? "pages" : "Units"}`,
        passage.hasQuote ? null : "no quote",
      ].filter(Boolean);
      return `${index + 1}. ${passage.documentName}, ${unit} ${pages(passage.pageFrom, passage.pageTo)}${lacks.length ? ` (${lacks.join(", ")})` : ""}`;
    });
  for (const question of run.questions) {
    if (question.modes[GATING_MODE]?.hit) continue;
    lines.push(`- **${question.id}** ${question.question}`);
    for (const line of describe(question)) lines.push(`  ${line}`);
  }
  return lines;
}

/**
 * Citation figures, a column per group of Answers. With `minCitations` (the
 * gating set's), each figure's target in a last column, and the reviewer's row.
 */
function citationTable(
  columns: readonly (readonly [string, GroupSummary])[],
  minCitations: number | null,
): string[] {
  const gating = minCitations !== null;
  const row = (label: string, cell: (summary: GroupSummary) => string, target = "") =>
    `| ${label} | ${columns.map(([, summary]) => cell(summary)).join(" | ")} |${gating ? ` ${target} |` : ""}`;
  const outcome = (key: CitationOutcome) => (summary: GroupSummary) =>
    summary.citations > 0
      ? `${summary.outcomes[key]} (${percent(summary.outcomes[key] / summary.citations)})`
      : "0";
  return [
    `| | ${columns.map(([label]) => label).join(" | ")} |${gating ? " Target (per language) |" : ""}`,
    `|---|${columns.map(() => "---|").join("")}${gating ? "---|" : ""}`,
    row("Answers (failed)", (summary) => `${summary.answers} (${summary.failedAnswers})`),
    row("Citations", (summary) => String(summary.citations), `at least ${minCitations}`),
    row(
      OUTCOME_LABELS.found,
      (summary) => percent(summary.foundShare),
      `at least ${percent(CITATION_TARGETS.found)}`,
    ),
    row(
      OUTCOME_LABELS["false-not-found"],
      (summary) => percent(summary.falseNotFoundShare),
      `at most ${percent(CITATION_TARGETS.falseNotFound)}`,
    ),
    ...CITATION_OUTCOMES.filter((key) => key !== "found" && key !== "false-not-found").map((key) =>
      row(OUTCOME_LABELS[key], outcome(key)),
    ),
    row(
      "Sentences with a Citation (coverage)",
      (summary) => `${percent(summary.coverage)} (${summary.citedSentences}/${summary.sentences})`,
      "reported",
    ),
    row(
      "Markers without records (removed)",
      (summary) => String(summary.droppedMarkers),
      "reported",
    ),
    row(
      "Records without markers (dropped)",
      (summary) => String(summary.droppedRecords),
      "reported",
    ),
    ...(gating
      ? [
          row(
            "Found quotes that support their sentence",
            () => "reviewer sheet",
            `at least ${percent(CITATION_TARGETS.supports)}`,
          ),
        ]
      : []),
  ];
}

/** The every-format set (#70): per format, per hard place and per Question. Never gating. */
function formatsSection(formats: FormatsReport): string[] {
  const { retrieval } = formats;
  const top = retrieval.summary[GATING_MODE];
  const notSearchable = formats.documents.filter((document) => document.status !== "ready");
  const lines = [
    "## Every format (reported, not gating)",
    "",
    `\`${formats.source}\`: ${retrieval.questions.length} Questions over ${formats.documents.length} Documents in Word, PowerPoint, Excel, CSV, Markdown, plain text and PDF, in English and Chinese, about the hard places in each (#70). They are searched in a library of their own, with the gating set's hit rule; the expected pages are the Units a Citation would cite. Known gaps, text the readers don't index today (Word comments, scanned pages), are counted apart. No bar is set yet: \`eval/README.md\` proposes one per format.`,
    "",
    `| Format | Documents | Questions (English / Chinese), known gaps apart | ${GATING_LABEL}: English | Chinese | All | ${HYBRID}: All | ${KEYWORD_RERANK_LABEL}: All | Known gaps found |`,
    "|---|---|---|---|---|---|---|---|---|",
    ...formats.formats.map((format) => {
      const reranked = format.retrieval[GATING_MODE];
      const hybrid = format.retrieval[HYBRID];
      const keyword = format.retrieval[KEYWORD_RERANK_MODE];
      if (!reranked || !hybrid)
        return `| ${format.label} | ${format.documents} | – | – | – | – | – | – | – |`;
      return `| ${format.label} | ${format.documents} | ${reranked.en.total} / ${reranked.zh.total} | ${fraction(reranked.en)} | ${fraction(reranked.zh)} | **${fraction(reranked.all)}** | ${fraction(hybrid.all)} | ${keyword ? fraction(keyword.all) : "–"} | ${fraction(format.knownGaps)} |`;
    }),
    "",
    ...(top
      ? [
          `Cross-lingual, every format: ${fraction(top.crossLingual)}${top.crossLingualTranslated ? `, with a translated second query ${fraction(top.crossLingualTranslated)}` : ""}.`,
          "",
        ]
      : []),
    ...(notSearchable.length
      ? [
          `Not searchable: ${notSearchable.map((document) => `${document.name} (${document.status})`).join(", ")}.`,
          "",
        ]
      : []),
    "### By hard place",
    "",
    `Hits of ${GATING_LABEL}, known gaps included.`,
    "",
    "| Format | Place | English | Chinese | |",
    "|---|---|---|---|---|",
    ...formats.formats.flatMap((format) =>
      format.places.map(
        (place) =>
          `| ${format.label} | ${place.place} | ${fraction(place.hits.en)} | ${fraction(place.hits.zh)} | ${place.knownGap ? `known gap: ${place.knownGap}` : ""} |`,
      ),
    ),
    "",
  ];
  const { citations, crossLingualCitations } = formats;
  if ("skipped" in citations) {
    lines.push("### Citations per format", "", `Skipped: ${citations.skipped}`, "");
  } else {
    const columns = [
      ...formats.formats.flatMap((format) =>
        format.citations ? [[format.label, format.citations] as const] : [],
      ),
      ...(crossLingualCitations ? [["Cross-lingual", crossLingualCitations] as const] : []),
    ];
    lines.push(
      "### Citations per format",
      "",
      `- Model: \`${citations.model}\`, ${citations.service ? `sent to ${citations.service}` : "on this computer"}; each Question asked once, in a Mind of its own. Reported, not gating.`,
      '- The figures are defined as for the gating set; a Citation\'s "pages" are the Units it cites.',
      "",
      ...citationTable(columns, null),
      "",
    );
  }
  lines.push(
    "### Per Question",
    "",
    "Ranks as for the gating set.",
    "",
    ...perQuestionTable([retrieval]),
    "",
    `### Misses of ${GATING_LABEL}`,
    "",
    ...(misses(retrieval, "Unit").length ? misses(retrieval, "Unit") : ["None."]),
    "",
  );
  return lines;
}

function markdownReport(report: EvalReport, reportDir: string, root: string): string {
  const { run, retrieval, citations } = report;
  const builtIn = retrieval.runs[0] as RetrievalRun;
  const sheet = relative(root, join(reportDir, "reviewer-sheet.csv"));
  const lines = [
    "# IncarnaMind evaluation",
    "",
    `**Result: ${report.result === "pass" ? "pass" : "fail"}**`,
    ...(report.failures.length ? ["", ...report.failures.map((failure) => `- ${failure}`)] : []),
    "",
    `- Run: ${run.startedAt}, ${run.seconds.toFixed(0)} s, commit ${run.commit}`,
    `- Machine: ${run.platform}, ${run.cpu}, Node ${run.node}`,
    `- Evaluation set: \`${report.evaluationSet.source}\`, ${report.evaluationSet.questions.gating.en} English and ${report.evaluationSet.questions.gating.zh} Chinese gating Questions, ${report.evaluationSet.questions.crossLingual} cross-lingual`,
    `- Documents: ${report.documents.length} (${report.documents.map((document) => document.name).join(", ")})`,
    "",
    "## Retrieval",
    "",
    `Top-${retrieval.topK} hit rate through the core's \`searchPassages\`. A Question is a hit when one of the top ${retrieval.topK} Passages belongs to the expected Document, covers the expected pages and contains the expected quote, both normalised (ADR-0009). The gate is what the search Tool does by default, ${GATING_LABEL}: hybrid search with the built-in embedding model, reranked by the built-in reranking model. It must find at least 80% overall and in each language (32 of 40, and 16 of 20 per language, with today's set). The plain search modes, keyword + rerank, the other reranking candidates, cross-lingual Questions and cloud embedding models are reported only.`,
    "",
    `- **Reranked modes** ("${HYBRID} + model"): what the search Tool hands a reranker, keyword search's top ${RERANK_PER_LIST} and vector search's top ${RERANK_PER_LIST}, each Passage once, reordered by a reranking model.`,
    `- **Keyword + rerank** ("keyword + model"): keyword search's top ${KEYWORD_RERANK_DEPTH}, with no vector search, reordered by the same reranking models: as many candidates as the most the search Tool hands one.`,
    "- **With a translated second query:** the cross-lingual Questions that have a hand-written translation into their Document's language are also searched with it, as an Answer is told to search again in the Documents' language. A hit in either search's top 5 counts. The translation is written by hand, so this is the most the approach can bring.",
    "",
    ...retrievalTable(retrieval.runs),
    "",
    ...retrieval.runs.map(
      (each) =>
        `- ${each.embedding}: ${each.passageCount} Passages, added and processed (text extraction and embedding) in ${each.processingSeconds.toFixed(0)} s.`,
    ),
    "",
    ...(builtIn.rerankers?.length
      ? [
          "### Reranking models",
          "",
          ...candidateCountsOf(builtIn).flatMap((counts) => [candidatesLine(counts), ""]),
          "Time to rerank one search's candidates on this machine, one Passage at a time on a worker thread, after the first search (which loads the model; each mode opens it afresh).",
          "",
          ...rerankerTable(builtIn),
          "",
        ]
      : []),
    "### Per Question",
    "",
    `The rank of the first Passage that meets the hit rule. Only the top ${retrieval.topK} count as hits: a rank in brackets is a near miss, within the top ${RANK_DEPTH}, and "–" means none in the top ${RANK_DEPTH}. For a cross-lingual Question with a translated query, "a / b" is the rank for the Question, then for its translation.`,
    "",
    ...perQuestionTable(retrieval.runs),
    "",
    `### Misses of ${GATING_LABEL} (gating)`,
    "",
    ...(misses(builtIn).length ? misses(builtIn) : ["None."]),
    "",
    "## Citation quality",
    "",
  ];
  if ("skipped" in citations) {
    lines.push(`Skipped: ${citations.skipped}`, "");
  } else {
    lines.push(
      `- Model: \`${citations.model}\`, ${citations.service ? `sent to ${citations.service}` : "on this computer"}. ${citations.gating ? "It is the gating model." : "A local model: reported, not gating."}`,
      `- Each Question asked in a Mind of its own; ${citations.rounds} round${citations.rounds === 1 ? "" : "s"} (more rounds ask a language's gating Questions again until it has ${citations.minCitations} Citations).`,
      `- False "not found": the check said "not found", but the quote is on the cited pages once both are compared by letters and digits only, ignoring case, accents and punctuation. The share is of all Citations.`,
      "- Coverage counts every sentence of an Answer (headings and code left out) as drawn from Documents, so it is a lower bound: sentences that only say what the Documents don't cover count as uncited.",
      `- Reviewer sheet: \`${sheet}\`. Mark each found quote "y" if it supports its sentence, "n" if not; the target is ${percent(CITATION_TARGETS.supports)} "y".`,
      "",
      ...citationTable(
        GROUPS.map(([group, label]) => [label, citations.summary[group]] as const),
        citations.minCitations,
      ),
      "",
    );
  }
  if (report.formats) lines.push(...formatsSection(report.formats));
  lines.push(
    "## Citation-check cases",
    "",
    "The check's cases (hyphenation, ligatures, full-width punctuation, CJK text, quotes across a page break, the page-range rule, in English and Chinese) are unit tests that run with `npm test`: see `eval/README.md`.",
    "",
  );
  return lines.join("\n");
}

const csvCell = (value: string | number) => {
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

/** Found quotes, one row each, with an empty column for the reviewer's judgement. */
export function reviewerSheet(run: CitationRun): string {
  const rows: (string | number)[][] = [
    [
      "question_id",
      "language",
      "round",
      "question",
      "answer_sentence",
      "quote",
      "document",
      "page",
      "supports (y/n)",
      "note",
    ],
  ];
  for (const answer of run.answers) {
    for (const citation of answer.citations) {
      if (citation.outcome !== "found") continue;
      rows.push([
        answer.questionId,
        answer.crossLingual ? `${answer.language} (cross-lingual)` : answer.language,
        answer.round,
        answer.question,
        citation.sentence,
        citation.quote,
        citation.documentName,
        pages(citation.pageFrom, citation.pageTo),
        "",
        "",
      ]);
    }
  }
  // A byte-order mark, so spreadsheet apps read the Chinese text as UTF-8.
  return `\uFEFF${rows.map((row) => row.map(csvCell).join(",")).join("\r\n")}\r\n`;
}

/** Writes the reports into a new folder and returns its path. */
export async function writeReports(
  report: EvalReport,
  resultsDir: string,
  root: string,
): Promise<string> {
  const dir = join(resultsDir, report.run.startedAt.replace(/[:.]/g, "-"));
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
  await writeFile(join(dir, "report.md"), markdownReport(report, dir, root));
  if (!("skipped" in report.citations)) {
    await writeFile(join(dir, "reviewer-sheet.csv"), reviewerSheet(report.citations));
  }
  if (report.formats && !("skipped" in report.formats.citations)) {
    await writeFile(
      join(dir, "reviewer-sheet-formats.csv"),
      reviewerSheet(report.formats.citations),
    );
  }
  return dir;
}

/** The summary printed at the end of a run. */
export function terminalSummary(report: EvalReport, reportDir: string, root: string): string {
  const lines = ["", `Retrieval, top ${report.retrieval.topK} (hit rule of ADR-0009)`];
  for (const run of report.retrieval.runs) {
    lines.push(`  ${run.embedding}${run.gating ? "" : " (reported, not gating)"}`);
    for (const mode of modesIn(run)) {
      const summary = run.summary[mode];
      if (!summary) continue;
      const label =
        run.gating && mode === GATING_MODE
          ? `${modeLabel(run, mode)} (gating)`
          : modeLabel(run, mode);
      const translated = summary.crossLingualTranslated;
      lines.push(
        `    ${label.padEnd(16)} English ${fraction(summary.en).padEnd(6)} Chinese ${fraction(summary.zh).padEnd(6)} gating set ${fraction(summary.core).padEnd(6)} cross-lingual ${fraction(summary.crossLingual)}${translated ? `, with a translated second query ${fraction(translated)}` : ""}`,
      );
    }
    for (const counts of candidateCountsOf(run)) lines.push(`    ${candidatesLine(counts)}`);
    for (const each of run.rerankers ?? []) {
      lines.push(
        `    ${modeLabel(run, each.mode as RetrievalMode)}: ${(each.downloadBytes / 1e6).toFixed(0)} MB, ${each.latency.mean.toFixed(0)} ms per search on average (95th percentile ${each.latency.p95.toFixed(0)} ms)`,
      );
    }
  }
  const { citations } = report;
  if ("skipped" in citations) {
    lines.push("", `Citation quality: skipped. ${citations.skipped}`);
  } else {
    lines.push(
      "",
      `Citation quality, ${citations.model}${citations.gating ? " (gating)" : " (local, not gating)"}`,
    );
    for (const [group, label] of GROUPS) {
      const summary = citations.summary[group];
      lines.push(
        `  ${label.padEnd(14)} ${String(summary.citations).padStart(3)} Citations, found ${percent(summary.foundShare)}, false "not found" ${percent(summary.falseNotFoundShare)}, coverage ${percent(summary.coverage)}, dropped ${summary.droppedMarkers} markers and ${summary.droppedRecords} records`,
      );
    }
  }
  if (report.formats) lines.push(...formatsSummary(report.formats));
  lines.push("", `Result: ${report.result}`);
  for (const failure of report.failures) lines.push(`  - ${failure}`);
  lines.push(`Reports: ${relative(root, reportDir)}/`, "");
  return lines.join("\n");
}

/** The every-format set's lines of the terminal summary. */
function formatsSummary(formats: FormatsReport): string[] {
  const lines = [
    "",
    `Every format (reported, not gating): ${GATING_LABEL}, top 5, known gaps apart`,
  ];
  for (const format of formats.formats) {
    const reranked = format.retrieval[GATING_MODE];
    const keyword = format.retrieval[KEYWORD_RERANK_MODE];
    const citations = format.citations;
    lines.push(
      `  ${format.label.padEnd(24)} English ${fraction(reranked?.en ?? { hits: 0, total: 0 }).padEnd(6)} Chinese ${fraction(reranked?.zh ?? { hits: 0, total: 0 }).padEnd(6)} known gaps ${fraction(format.knownGaps)}${keyword ? `; ${KEYWORD_RERANK_LABEL} ${fraction(keyword.all)}` : ""}${citations ? `; ${citations.citations} Citations, found ${percent(citations.foundShare)}, false "not found" ${percent(citations.falseNotFoundShare)}, coverage ${percent(citations.coverage)}` : ""}`,
    );
  }
  const top = formats.retrieval.summary[GATING_MODE];
  if (top) {
    lines.push(
      `  ${"Cross-lingual".padEnd(24)} ${fraction(top.crossLingual)}${top.crossLingualTranslated ? `, with a translated second query ${fraction(top.crossLingualTranslated)}` : ""}`,
    );
  }
  return lines;
}
