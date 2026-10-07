/**
 * The grouping check's outputs, as the evaluation writes its own: a folder
 * under eval/results/ with report.json (everything) and report.md (for
 * people), plus founder-sample.csv for the founder's library, and a summary
 * on the terminal.
 */
import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { arch, cpus, platform } from "node:os";
import { join, relative } from "node:path";
import type { KMeansRun } from "../../../src/core/topics/grouping";
import type { IncrementalResult } from "./incremental";
import {
  BARS,
  type CaseResult,
  type CasesScore,
  type GroupingSummary,
  type NameRowScore,
} from "./scoring";
import type { Form, Language } from "./set";
import type { KMeansTiming, MeansTiming } from "./timing";
import { median } from "./timing";
import type { VariantId } from "./variants";

export interface RunInfo {
  startedAt: string;
  seconds: number;
  commit: string;
  node: string;
  platform: string;
  cpu: string;
}

export function runInfo(root: string, started: Date): RunInfo {
  const git = (...args: string[]) =>
    execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
  let commit = "unknown";
  try {
    const changed = git("status", "--porcelain") !== "";
    commit = `${git("rev-parse", "--short", "HEAD")}${changed ? " (with uncommitted changes)" : ""}`;
  } catch {
    // Not a git checkout.
  }
  const cpu = cpus();
  return {
    startedAt: started.toISOString(),
    seconds: (Date.now() - started.getTime()) / 1000,
    commit,
    node: process.version,
    platform: `${platform()} ${arch()}`,
    cpu: `${cpu[0]?.model.trim() ?? "unknown CPU"} (${cpu.length} cores)`,
  };
}

export interface GroupingRun {
  k: number;
  seconds: number;
  runs: KMeansRun[];
  summary: GroupingSummary;
  choosing: CasesScore;
  heldOut: CasesScore;
  nameRows: NameRowScore[];
}

export interface VariantResult extends GroupingRun {
  id: VariantId;
  label: string;
  cost: string;
  /** Documents with a vector. */
  documents: number;
  /** Building the vectors: the extra embeddings this variant needs and their time. */
  extraEmbeddings: number;
  buildSeconds: number;
}

export interface ClassifierResult {
  name: string;
  local: boolean;
  seconds: number;
  perDocumentMs: number;
  failed: number;
  firstError: string | null;
  summary: GroupingSummary;
  choosing: CasesScore;
  heldOut: CasesScore;
}

export type Classifiers =
  | { skipped: string }
  | {
      proposer: string;
      topics: string[];
      proposeSeconds: number;
      runs: ClassifierResult[];
      ranking: { name: string; reason: string } | null;
    };

export interface FixturesReport {
  kind: "fixtures";
  result: "pass" | "fail";
  failures: string[];
  run: RunInfo;
  set: {
    source: string;
    subjects: number;
    documents: {
      key: string;
      name: string;
      subject: string;
      language: Language;
      form: Form;
      passages: number;
      role: string;
    }[];
  };
  processing: { passages: number; seconds: number };
  variants: VariantResult[];
  chosen: { id: VariantId; reason: string };
  bars: {
    pairs: { hits: number; total: number; needed: number };
    decks: { hits: number; total: number; needed: number };
  };
  incremental: IncrementalResult;
  classifiers: Classifiers;
  timing: { kMeans: KMeansTiming; means: MeansTiming } | { skipped: string };
  recommendation: string[];
}

export interface FounderReport {
  kind: "founder";
  run: RunInfo;
  folder: string;
  files: number;
  ready: number;
  notReady: number;
  passages: number;
  processingSeconds: number;
  /** The letter each grouping has in the sheet. */
  letters: Record<
    string,
    {
      id: VariantId;
      label: string;
      k: number;
      topics: number;
      ungrouped: number;
      buildSeconds: number;
    }
  >;
  sample: number;
  sheet: string;
}

const fraction = ({ hits, total }: { hits: number; total: number }) => `${hits}/${total}`;
const percent = (value: number) => `${(value * 100).toFixed(0)}%`;
const ms = (value: number) => `${value.toFixed(0)} ms`;

/** The exact steps for the founder's 30-Document sample, which only the User can run. */
export const FOUNDER_PROCEDURE = [
  "1. In your own checkout of IncarnaMind, on this branch, with dependencies installed (`npm ci`).",
  '2. Run the check on your library folder (read only; nothing in it changes, and everything is indexed in a temporary data folder): `INCARNAMIND_EVAL_GROUPING_FOLDER="/path/to/your/library" INCARNAMIND_EVAL_GROUPING_LIMIT=300 npm run eval:grouping`. The limit draws 300 files at random; leave it out to group them all (about 25 Passages a second to embed, twice over for the name-free variant).',
  "3. Open `founder-sample.csv` in the new `eval/results/grouping-founder-…/` folder. It has 30 random Documents, each three times, once per grouping (A, B, C), with its Topic shown as the Documents nearest the Topic's centre.",
  '4. For each row, write `y` in "right Topic? (y/n)" if the Document belongs with those Documents (for "Not grouped yet": if it really fits none), `n` if not.',
  "5. Count the `y` rows of each letter; then look up which letter is which variant in `report.json` (`letters`). The bar is at least 21 of 30 (70%) for the winning variant.",
];

function caseLine(each: CaseResult): string {
  return `${each.documents.join(" + ")}: ${each.hit ? "hit" : `miss (${each.why})`}`;
}

function variantTable(variants: readonly VariantResult[], chosen: VariantId): string[] {
  const lines = [
    "| Document vector | Choosing: pairs | Choosing: decks | **Held-out: pairs** | **Held-out: decks** | Purity | Topics (ungrouped) | Extra cost |",
    "|---|---|---|---|---|---|---|---|",
  ];
  for (const variant of variants) {
    const label = variant.id === chosen ? `**${variant.label} (chosen)**` : variant.label;
    lines.push(
      `| ${label} | ${fraction(variant.choosing.pairs)} | ${fraction(variant.choosing.decks)} | ${fraction(variant.heldOut.pairs)} | ${fraction(variant.heldOut.decks)} | ${percent(variant.summary.purity)} | ${variant.summary.topics} (${variant.summary.ungrouped}) | ${variant.extraEmbeddings} embeddings, ${variant.buildSeconds.toFixed(1)} s |`,
    );
  }
  return lines;
}

function nameRowTable(variants: readonly VariantResult[]): string[] {
  const rows = variants[0]?.nameRows ?? [];
  const lines = [
    `| Document vector | ${rows.map((row) => `${row.label} (${row.documents.length} Documents, ${row.pairs} pairs on different subjects)`).join(" | ")} |`,
    `|---|${rows.map(() => "---|").join("")}`,
  ];
  for (const variant of variants) {
    lines.push(
      `| ${variant.label} | ${variant.nameRows.map((row) => `${row.together} together`).join(" | ")} |`,
    );
  }
  return lines;
}

function classifierSection(classifiers: Classifiers): string[] {
  if ("skipped" in classifiers) return [`Skipped: ${classifiers.skipped}`];
  const lines = [
    `The Topic list (${classifiers.topics.length}), proposed by ${classifiers.proposer} from the titles in ${classifiers.proposeSeconds.toFixed(1)} s: ${classifiers.topics.join("; ")}.`,
    "",
    "| Classifier | Choosing | **Held-out: pairs** | **Held-out: decks** | Purity | Time per Document | Failed |",
    "|---|---|---|---|---|---|---|",
    ...classifiers.runs.map(
      (run) =>
        `| ${run.name}${run.local ? " (on this computer)" : ""} | ${fraction(run.choosing)} | ${fraction(run.heldOut.pairs)} | ${fraction(run.heldOut.decks)} | ${percent(run.summary.purity)} | ${ms(run.perDocumentMs)} | ${run.failed}${run.firstError ? ` (${run.firstError})` : ""} |`,
    ),
  ];
  if (classifiers.ranking) {
    lines.push("", `Best: ${classifiers.ranking.name}, by ${classifiers.ranking.reason}.`);
  }
  return lines;
}

function timingSection(timing: FixturesReport["timing"]): string[] {
  if ("skipped" in timing) return [`Skipped: ${timing.skipped}`];
  const { kMeans, means } = timing;
  return [
    `- **k-means at ${kMeans.documents.toLocaleString("en")} Documents** (${kMeans.dimensions} dimensions, generated around 60 directions; k = ${kMeans.k}, 3 seeded runs, as the grouping worker runs it): ${kMeans.seconds.toFixed(2)} s. Iterations per run: ${kMeans.runs.map((run) => run.iterations).join(", ")}.`,
    `- **The means on the core's thread at ${means.passages.toLocaleString("en")} Passages** (${means.documents.toLocaleString("en")} Documents, ${means.textLength} characters of text per Passage, a ${means.databaseMb.toFixed(0)} MB database): loading the index and computing the means took ${means.coldMs.map(ms).join(", ")} (median ${ms(median(means.coldMs))}); the means alone, with the index loaded, ${means.warmMs.map(ms).join(", ")}. The bar is ${means.barMs} ms: **${means.passes ? "met" : "missed"}**.`,
  ];
}

function incrementalSection(incremental: IncrementalResult): string[] {
  const { initial, lateArrivals, corrections, regroup, checks } = incremental;
  return [
    `First grouping without the late arrivals: ${initial.documents} Documents, k = ${initial.k}, ${initial.topics} Topics, ${initial.ungrouped} not grouped yet.`,
    "",
    "| Late arrival | Expected | Placed | Similarity / threshold | |",
    "|---|---|---|---|---|",
    ...lateArrivals.map(
      (arrival) =>
        `| ${arrival.key} | ${arrival.expected} | ${arrival.outcome} | ${arrival.similarity === null ? "–" : `${arrival.similarity.toFixed(3)} / ${arrival.threshold?.toFixed(3)}`} | ${arrival.correct ? "right" : "wrong"} |`,
    ),
    "",
    "The User's corrections:",
    "",
    ...corrections.map((line) => `- ${line}`),
    "",
    `Regroup: a pool of ${regroup.pool} Documents, ${regroup.reclustered ? `re-clustered with k = ${regroup.k}` : "only placed (under 12)"}; ${regroup.kept} Topics kept their id, ${regroup.created} new, ${regroup.removed} removed, ${regroup.ungrouped} not grouped yet.`,
    "",
    ...checks.map((check) => `- ${check.kept ? "Kept" : "**Lost**"}: ${check.description}.`),
    "",
    `After the Regroup (for information): choosing pairs ${fraction(incremental.afterRegroup.choosing.pairs)}, decks ${fraction(incremental.afterRegroup.choosing.decks)}; held-out pairs ${fraction(incremental.afterRegroup.heldOut.pairs)}, decks ${fraction(incremental.afterRegroup.heldOut.decks)}.`,
  ];
}

export function fixturesMarkdown(report: FixturesReport): string {
  const chosen = report.variants.find(
    (variant) => variant.id === report.chosen.id,
  ) as VariantResult;
  const { run } = report;
  const roles = report.set.documents;
  return [
    "# Grouping check",
    "",
    `**Result: ${report.result}**`,
    ...(report.failures.length ? ["", ...report.failures.map((failure) => `- ${failure}`)] : []),
    "",
    `- Run: ${run.startedAt}, ${run.seconds.toFixed(0)} s, commit ${run.commit}`,
    `- Machine: ${run.platform}, ${run.cpu}, Node ${run.node}`,
    `- Fixture set: \`${report.set.source}\`, ${roles.length} Documents on ${report.set.subjects} subjects, ${report.processing.passages} Passages, added and processed in ${report.processing.seconds.toFixed(0)} s with the built-in model.`,
    "- Every Document is grouped together, as one library, with at least as many clusters as subjects. The variant is chosen on the choosing set; the pass bars count on the held-out set only.",
    "",
    "## Recommendation",
    "",
    ...report.recommendation.map((line) => `- ${line}`),
    "",
    "## The three Document vectors",
    "",
    ...variantTable(report.variants, report.chosen.id),
    "",
    `Chosen: **${chosen.label}**, by ${report.chosen.reason}. A near-tie (within one case) goes to the cheaper variant.`,
    "",
    `Grouping took ${report.variants.map((variant) => `${variant.seconds.toFixed(2)} s (${variant.label})`).join(", ")}, k = ${chosen.k}.`,
    "",
    "### Documents whose names share a part (R3)",
    "",
    "How many pairs of Documents on different subjects, whose names share the part, land in one Topic. Fewer is better: names pull them together.",
    "",
    ...nameRowTable(report.variants),
    "",
    `### The held-out bars, with ${chosen.label}`,
    "",
    `- English–Chinese pairs in one Topic: **${fraction(report.bars.pairs)}**, the bar is ${report.bars.pairs.needed} (${percent(BARS.pairs)}).`,
    `- Decks and spreadsheets with a Document on their subject: **${fraction(report.bars.decks)}**, the bar is ${report.bars.decks.needed} (${percent(BARS.decks)}).`,
    "- The founder's 30-Document sample: pending, it needs the User (see below).",
    "",
    "Every case, with the chosen variant:",
    "",
    ...[...chosen.heldOut.pairs.cases, ...chosen.heldOut.decks.cases].map(
      (each) => `- held-out ${caseLine(each)}`,
    ),
    ...[...chosen.choosing.pairs.cases, ...chosen.choosing.decks.cases].map(
      (each) => `- choosing ${caseLine(each)}`,
    ),
    "",
    "Its Topics, by subject:",
    "",
    ...chosen.summary.contents.map((contents, index) => `${index + 1}. ${contents}`),
    "",
    "## Incremental cases, with the chosen variant",
    "",
    ...incrementalSection(report.incremental),
    "",
    "## The classifier fallback (R0)",
    "",
    ...classifierSection(report.classifiers),
    "",
    "## Timing",
    "",
    ...timingSection(report.timing),
    "",
    "## The founder's 30-Document sample",
    "",
    "Pending: only the User can run it, on their own library.",
    "",
    ...FOUNDER_PROCEDURE,
    "",
    "## The fixture set",
    "",
    "| Document | Name | Subject | Language | Form | Passages | Case |",
    "|---|---|---|---|---|---|---|",
    ...roles.map(
      (document) =>
        `| ${document.key} | ${document.name} | ${document.subject} | ${document.language} | ${document.form} | ${document.passages} | ${document.role} |`,
    ),
    "",
  ].join("\n");
}

export function founderMarkdown(report: FounderReport): string {
  const { run } = report;
  return [
    "# Grouping check: the founder's sample",
    "",
    `- Run: ${run.startedAt}, ${run.seconds.toFixed(0)} s, commit ${run.commit}`,
    `- Machine: ${run.platform}, ${run.cpu}, Node ${run.node}`,
    `- Library: ${report.files} files, ${report.ready} indexed (${report.notReady} not added, without text, or failed), ${report.passages} Passages, in ${report.processingSeconds.toFixed(0)} s.`,
    "",
    `The sheet \`${report.sheet}\` has ${report.sample} Documents, each once per grouping. The letters are mapped to the variants in report.json (\`letters\`): look after judging.`,
    "",
    ...FOUNDER_PROCEDURE.slice(2),
    "",
  ].join("\n");
}

/** Writes the reports (and the sheet) into a new folder and returns its path. */
export async function writeGroupingReports(
  resultsDir: string,
  name: string,
  startedAt: string,
  files: Record<string, string>,
): Promise<string> {
  const dir = join(resultsDir, `${name}-${startedAt.replace(/[:.]/g, "-")}`);
  await mkdir(dir, { recursive: true });
  for (const [file, contents] of Object.entries(files)) await writeFile(join(dir, file), contents);
  return dir;
}

export function fixturesSummary(report: FixturesReport, dir: string, root: string): string {
  const lines = ["", "Grouping check (held-out bars with the chosen variant)"];
  for (const variant of report.variants) {
    lines.push(
      `  ${variant.label.padEnd(50)} choosing ${fraction(variant.choosing).padEnd(6)} held-out pairs ${fraction(variant.heldOut.pairs)}, decks ${fraction(variant.heldOut.decks)}${variant.id === report.chosen.id ? "  (chosen)" : ""}`,
    );
  }
  if (!("skipped" in report.timing)) {
    const { kMeans, means } = report.timing;
    lines.push(
      `  k-means at ${kMeans.documents} Documents: ${kMeans.seconds.toFixed(2)} s; means at ${means.passages} Passages: ${ms(median(means.coldMs))} cold (bar ${means.barMs} ms), ${ms(median(means.warmMs))} warm`,
    );
  }
  lines.push("", `Result: ${report.result}`);
  for (const failure of report.failures) lines.push(`  - ${failure}`);
  lines.push(`Reports: ${relative(root, dir)}/`, "");
  return lines.join("\n");
}
