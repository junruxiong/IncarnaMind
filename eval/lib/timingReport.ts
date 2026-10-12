/** How the reports show where Answers' time went (see ./timing). */
import type { HandedSummary } from "./handed";
import { CALLS_SHOWN, type Spread, type TimingSummary } from "./timing";

interface Column {
  label: string;
  timing?: TimingSummary | null;
  handed?: HandedSummary | null;
}

const secondsText = (value: number) => `${value.toFixed(value < 10 ? 2 : 1)} s`;

/** "5.80 s (95th 10.1 s)", or "–". */
export const spreadText = (spread: Spread | null | undefined) =>
  spread ? `${secondsText(spread.median)} (95th ${secondsText(spread.p95)})` : "–";

const callLabel = (index: number) =>
  index < CALLS_SHOWN ? `Model call ${index + 1}` : `Model calls after the ${CALLS_SHOWN}`;

/** A table of where the time went, a column per group of Answers. */
export function timingTable(columns: readonly Column[]): string[] {
  if (columns.every((column) => !column.timing)) return [];
  const row = (label: string, cell: (column: Column) => string) =>
    `| ${label} | ${columns.map(cell).join(" | ")} |`;
  const timed = (pick: (timing: TimingSummary) => string) => (column: Column) =>
    column.timing ? pick(column.timing) : "–";
  const orders = Array.from({ length: CALLS_SHOWN + 1 }, (_, index) => index);
  return [
    `| Median (95th percentile) | ${columns.map((column) => column.label).join(" | ")} |`,
    `|---|${columns.map(() => "---|").join("")}`,
    row(
      "Answers timed",
      timed((timing) => String(timing.answers)),
    ),
    row(
      "Computer's load when asked (median)",
      timed((timing) => timing.loadAverage?.toFixed(1) ?? "–"),
    ),
    row(
      "**Whole Answer**",
      timed((timing) => spreadText(timing.whole)),
    ),
    row(
      "First text that stays in the Answer",
      timed((timing) => spreadText(timing.firstText)),
    ),
    row(
      "Each model call",
      timed((timing) => spreadText(timing.modelCall)),
    ),
    row(
      "Wait for a model call's first byte",
      timed((timing) => spreadText(timing.modelCallWait)),
    ),
    ...orders.map((index) =>
      row(
        callLabel(index),
        timed((timing) => {
          const call = timing.modelCallsByOrder[index];
          return call?.duration
            ? `${spreadText(call.duration)}, wait ${secondsText(call.wait?.median ?? 0)}`
            : "–";
        }),
      ),
    ),
    row(
      "Each search",
      timed((timing) => spreadText(timing.search)),
    ),
    row(
      "Reranker, in a search that reranked",
      timed((timing) => spreadText(timing.reranker)),
    ),
    row(
      "Model calls an Answer (mean)",
      timed((timing) => timing.modelCallsPerAnswer.toFixed(1)),
    ),
    row(
      "Searches an Answer (mean)",
      timed((timing) => timing.searchesPerAnswer.toFixed(1)),
    ),
    row(
      "Answers written twice",
      timed((timing) => `${timing.writtenTwice}/${timing.answers}`),
    ),
    row(
      "Tokens a model call: input (cached), output (mean)",
      timed((timing) =>
        timing.tokensPerCall
          ? `${timing.tokensPerCall.input.toFixed(0)} (${timing.tokensPerCall.cachedInput.toFixed(0)}), ${timing.tokensPerCall.output.toFixed(0)}`
          : "–",
      ),
    ),
    row("A search handed every Passage the Question needs", (column) =>
      column.handed ? `${column.handed.handed}/${column.handed.answers}` : "–",
    ),
    row("The Answer cites every Passage the Question needs", (column) =>
      column.handed ? `${column.handed.cited}/${column.handed.answers}` : "–",
    ),
  ];
}

/** The timing table of some groups of Answers, with its heading; nothing without recorded Answers. */
export function timingSection(groups: readonly Column[], heading = "###"): string[] {
  const table = timingTable(groups);
  if (table.length === 0) return [];
  return [
    `${heading} Where the time went`,
    "",
    "Each Answer's model calls, searches and reranker are timed from the moment its Question is asked (`report.json` keeps every Answer's own). The first text counts only text that stayed in the Answer: a preamble taken back doesn't. \"Written twice\" means a paragraph of the Answer appears again word for word. A search \"handed\" a Passage when the model was given it, whether or not the Answer cites it. Times depend on the computer's load: compare runs with a similar load.",
    "",
    ...table,
    "",
  ];
}

/** One line of the terminal summary for a group, or null without timings. */
export function timingLine(timing: TimingSummary | null | undefined): string | null {
  if (!timing) return null;
  const first = timing.firstText ? `, first text ${secondsText(timing.firstText.median)}` : "";
  const reranker = timing.reranker
    ? `, reranker ${secondsText(timing.reranker.median)} a search`
    : "";
  return `time: ${spreadText(timing.whole)} an Answer${first}, ${timing.modelCallsPerAnswer.toFixed(1)} model calls (${secondsText(timing.modelCall?.median ?? 0)} each), ${timing.searchesPerAnswer.toFixed(1)} searches${reranker}${timing.writtenTwice > 0 ? `, ${timing.writtenTwice} written twice` : ""}, load ${timing.loadAverage?.toFixed(1) ?? "–"}`;
}
