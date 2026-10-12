/**
 * The checks evaluation's report (#86): the bars, Find support and verify
 * figures per language and format, and every miss, for people (report.md),
 * the terminal, and everything as JSON (report.json).
 */
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { PipelineUsage } from "./pipeline";
import type {
  FindSupportScore,
  FoundEvidence,
  Gate,
  Miss,
  PrecisionRecall,
  Rate,
  SentenceResult,
  VerifiedDocument,
  VerifyScore,
} from "./scoring";
import type { ChecksSet, LabelledQuote } from "./set";

export interface ChecksRun {
  startedAt: string;
  seconds: number;
  commit: string;
  model: string;
  /** What extracted the claims and checked them: the baseline until #96 and #97 exist. */
  pipeline: string;
}

export interface ChecksReport {
  run: ChecksRun;
  set: {
    findSupport: number;
    verify: number;
    received: number;
    confirmed: number;
    review: ChecksSet["review"];
  };
  /** Whether a missed bar fails the run. */
  gating: boolean;
  gates: Gate[];
  findSupport: FindSupportScore;
  verify: VerifyScore;
  usage: PipelineUsage;
  results: { findSupport: SentenceResult[]; verify: VerifiedDocument[] };
}

const percent = (value: number | null) => (value === null ? "–" : `${(value * 100).toFixed(1)}%`);

const ofRate = (value: Rate) => `${value.hits}/${value.total} (${percent(value.share)})`;

const row = (cells: readonly string[]) =>
  `| ${cells.map((cell) => cell.replace(/\|/g, "\\|").replace(/\n/g, " ")).join(" | ")} |`;

const prCells = (value: PrecisionRecall) => [
  percent(value.precision),
  percent(value.recall),
  `${value.truePositives} / ${value.falsePositives} / ${value.falseNegatives}`,
];

const where = (quote: FoundEvidence | LabelledQuote | null) => {
  if (!quote) return "–";
  const units = quote.units ? `Units ${quote.units[0]}–${quote.units[1]}` : "";
  const location = "location" in quote ? quote.location : units;
  return `${quote.document}, ${location}: "${quote.quote}"`;
};

function missTable(misses: readonly Miss[]): string[] {
  if (misses.length === 0) return ["None."];
  return [
    row(["Id", "Language", "Label", "Verdict", "Text", "Found", "Labelled evidence", "Why"]),
    row(["---", "---", "---", "---", "---", "---", "---", "---"]),
    ...misses.map((miss) =>
      row([
        miss.id ?? `(no label, ${miss.document})`,
        miss.language,
        miss.label ?? "–",
        miss.verdict,
        miss.text,
        where(miss.found),
        where(miss.expected),
        miss.reason ?? "",
      ]),
    ),
  ];
}

export function markdownReport(report: ChecksReport): string {
  const { run, findSupport, verify } = report;
  const lines = [
    "# Checks evaluation: Find support and verify (#86)",
    "",
    `Run ${run.startedAt} at ${run.commit}, ${Math.round(run.seconds)} s, with ${run.model}. Claims extracted and checked by ${run.pipeline}.`,
    "",
    `Set: ${report.set.findSupport} Find support sentences and ${report.set.verify} claims in ${report.set.received} received documents; ${report.set.confirmed} labels confirmed${report.set.review.by ? ` by ${report.set.review.by} on ${report.set.review.at}` : " (none reviewed yet)"}.`,
    "",
    report.gating ? "## Bars" : "## Bars (reported only: they gate once #96 and #97 exist)",
    "",
    row(["Bar", "Value", "Needed", "Met"]),
    row(["---", "---", "---", "---"]),
    ...report.gates.map((gate) =>
      row([gate.name, percent(gate.value), `≥ ${percent(gate.bar)}`, gate.met ? "yes" : "no"]),
    ),
    "",
    "## Find support",
    "",
    row(["", '"Supported" precision', "Recall", "True / false positives / misses"]),
    row(["---", "---", "---", "---"]),
    row(["All", ...prCells(findSupport.supported)]),
    row(["English", ...prCells(findSupport.byLanguage.en)]),
    row(["Chinese", ...prCells(findSupport.byLanguage.zh)]),
    ...Object.entries(findSupport.byFormat).map(([format, value]) =>
      row([format, ...prCells(value)]),
    ),
    "",
    `- Rightly supported, on the labelled Passage: ${ofRate(findSupport.labelledEvidence)}.`,
    `- "Contradicted" (reported): precision ${percent(findSupport.contradicted.precision)}, recall ${percent(findSupport.contradicted.recall)}.`,
    `- Not checked: ${findSupport.notChecked}.`,
    "",
    "## Verify",
    "",
    `- Claims extracted: ${ofRate(verify.extracted)}; English ${ofRate(verify.extractedByLanguage.en)}, Chinese ${ofRate(verify.extractedByLanguage.zh)}.`,
    `- Extracted claims that match no label: ${verify.unlabelled}. Not checked: ${verify.notChecked}.`,
    "",
    row(["", '"Contradicted" precision', "Recall", "True / false positives / misses"]),
    row(["---", "---", "---", "---"]),
    row(["All", ...prCells(verify.contradicted)]),
    row(["English", ...prCells(verify.contradictedByLanguage.en)]),
    row(["Chinese", ...prCells(verify.contradictedByLanguage.zh)]),
    row(['"Supported" (reported)', ...prCells(verify.supported)]),
    "",
    `Model calls: ${report.usage.calls}, ${report.usage.inputTokens} input and ${report.usage.outputTokens} output tokens.`,
    "",
    "## Misses",
    "",
    "### Find support",
    "",
    ...missTable(findSupport.misses),
    "",
    "### Verify",
    "",
    "Labelled claims not extracted, claims checked otherwise than labelled, and claims called contradicted that match no label.",
    "",
    ...missTable(verify.misses),
    "",
  ];
  return lines.join("\n");
}

export function summaryLines(report: ChecksReport, dir: string): string[] {
  const { findSupport, verify } = report;
  return [
    `Checks (#86), ${report.gating ? "gating" : "reported only until #96 and #97"}:`,
    ...report.gates.map(
      (gate) =>
        `  ${gate.met ? "met" : "MISSED"}  ${gate.name}: ${percent(gate.value)} (≥ ${percent(gate.bar)})`,
    ),
    `  Find support misses: ${findSupport.misses.length}; verify misses: ${verify.misses.length}`,
    `  Report: ${join(dir, "report.md")}`,
  ];
}

export async function writeChecksReport(report: ChecksReport, resultsDir: string): Promise<string> {
  const dir = join(resultsDir, `${report.run.startedAt.replace(/[:.]/g, "-")}-checks`);
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
  await writeFile(join(dir, "report.md"), markdownReport(report));
  return dir;
}
