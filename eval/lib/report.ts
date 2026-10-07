/**
 * The evaluation's outputs: report.json (everything), report.md (for people),
 * reviewer-sheet.csv (found quotes, for judging whether they support their
 * sentence), and a summary on the terminal.
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
import {
  GATING_MODE,
  type ModeSummary,
  type QuestionResult,
  RANK_DEPTH,
  type RetrievalRun,
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
    gatingMode: SearchMode;
    /** The built-in model first; then a cloud model, when one was given. */
    runs: RetrievalRun[];
  };
  citations: CitationRun | { skipped: string };
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

function retrievalTable(runs: readonly RetrievalRun[]): string[] {
  const lines = [
    "| Embedding model | Mode | English | Chinese | Gating set | Cross-lingual |",
    "|---|---|---|---|---|---|",
  ];
  for (const run of runs) {
    for (const mode of SEARCH_MODES) {
      const summary = run.summary[mode] as ModeSummary | undefined;
      if (!summary) continue;
      const gating = run.gating && mode === GATING_MODE;
      const label = gating ? `**${mode} (gating)**` : mode;
      lines.push(
        `| ${run.embedding} | ${label} | ${fraction(summary.en)} | ${fraction(summary.zh)} | ${gating ? `**${fraction(summary.core)}**` : fraction(summary.core)} | ${fraction(summary.crossLingual)} |`,
      );
    }
  }
  return lines;
}

function perQuestionTable(runs: readonly RetrievalRun[]): string[] {
  const columns = runs.flatMap((run) =>
    SEARCH_MODES.filter((mode) => run.summary[mode]).map((mode) => ({ run, mode })),
  );
  const header = columns.map(({ run, mode }) =>
    runs.length > 1 ? `${mode} (${run.gating ? "built-in" : "cloud"})` : mode,
  );
  const lines = [
    `| Question | ${header.join(" | ")} | Text |`,
    `|---|${columns.map(() => "---|").join("")}---|`,
  ];
  const first = runs[0] as RetrievalRun;
  first.questions.forEach((question, index) => {
    const cells = columns.map(({ run, mode }) => {
      const result = run.questions[index]?.modes[mode];
      if (!result?.rank) return "–";
      return result.hit ? String(result.rank) : `(${result.rank})`;
    });
    const id = question.crossLingual ? `${question.id} (cross-lingual)` : question.id;
    lines.push(`| ${id} | ${cells.join(" | ")} | ${question.question} |`);
  });
  return lines;
}

/** For each miss of the gating mode: what the top 5 were, and what each lacked. */
function misses(run: RetrievalRun): string[] {
  const lines: string[] = [];
  const describe = (question: QuestionResult) =>
    (question.modes[GATING_MODE]?.top ?? []).map((passage, index) => {
      const lacks = [
        passage.rightDocument ? null : "other Document",
        passage.coversPages ? null : "other pages",
        passage.hasQuote ? null : "no quote",
      ].filter(Boolean);
      return `${index + 1}. ${passage.documentName}, p. ${pages(passage.pageFrom, passage.pageTo)}${lacks.length ? ` (${lacks.join(", ")})` : ""}`;
    });
  for (const question of run.questions) {
    if (question.modes[GATING_MODE]?.hit) continue;
    lines.push(`- **${question.id}** ${question.question}`);
    for (const line of describe(question)) lines.push(`  ${line}`);
  }
  return lines;
}

function citationTable(run: CitationRun): string[] {
  const row = (label: string, cell: (summary: GroupSummary) => string, target = "") =>
    `| ${label} | ${GROUPS.map(([group]) => cell(run.summary[group])).join(" | ")} | ${target} |`;
  const outcome = (key: CitationOutcome) => (summary: GroupSummary) =>
    summary.citations > 0
      ? `${summary.outcomes[key]} (${percent(summary.outcomes[key] / summary.citations)})`
      : "0";
  return [
    `| | ${GROUPS.map(([, label]) => label).join(" | ")} | Target (per language) |`,
    `|---|${GROUPS.map(() => "---|").join("")}---|`,
    row("Answers (failed)", (summary) => `${summary.answers} (${summary.failedAnswers})`),
    row("Citations", (summary) => String(summary.citations), `at least ${run.minCitations}`),
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
    row(
      "Found quotes that support their sentence",
      () => "reviewer sheet",
      `at least ${percent(CITATION_TARGETS.supports)}`,
    ),
  ];
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
    `Top-${retrieval.topK} hit rate through the core's \`searchPassages\`. A Question is a hit when one of the top ${retrieval.topK} Passages belongs to the expected Document, covers the expected pages and contains the expected quote, both normalised (ADR-0009). The gate is ${GATING_MODE} search with the built-in model: at least 80% overall and in each language (32 of 40, and 16 of 20 per language, with today's set). Cross-lingual Questions and cloud embedding models are reported only.`,
    "",
    ...retrievalTable(retrieval.runs),
    "",
    ...retrieval.runs.map(
      (each) =>
        `- ${each.embedding}: ${each.passageCount} Passages, added and processed (text extraction and embedding) in ${each.processingSeconds.toFixed(0)} s.`,
    ),
    "",
    "### Per Question",
    "",
    `The rank of the first Passage that meets the hit rule. Only the top ${retrieval.topK} count as hits: a rank in brackets is a near miss, within the top ${RANK_DEPTH}, and "–" means none in the top ${RANK_DEPTH}.`,
    "",
    ...perQuestionTable(retrieval.runs),
    "",
    `### Misses of ${GATING_MODE} search with the built-in model`,
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
      ...citationTable(citations),
      "",
    );
  }
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
  return dir;
}

/** The summary printed at the end of a run. */
export function terminalSummary(report: EvalReport, reportDir: string, root: string): string {
  const lines = ["", `Retrieval, top ${report.retrieval.topK} (hit rule of ADR-0009)`];
  for (const run of report.retrieval.runs) {
    lines.push(`  ${run.embedding}${run.gating ? "" : " (reported, not gating)"}`);
    for (const mode of SEARCH_MODES) {
      const summary = run.summary[mode];
      if (!summary) continue;
      const label = run.gating && mode === GATING_MODE ? `${mode} (gating)` : mode;
      lines.push(
        `    ${label.padEnd(16)} English ${fraction(summary.en).padEnd(6)} Chinese ${fraction(summary.zh).padEnd(6)} gating set ${fraction(summary.core).padEnd(6)} cross-lingual ${fraction(summary.crossLingual)}`,
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
  lines.push("", `Result: ${report.result}`);
  for (const failure of report.failures) lines.push(`  - ${failure}`);
  lines.push(`Reports: ${relative(root, reportDir)}/`, "");
  return lines.join("\n");
}
