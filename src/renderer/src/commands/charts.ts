import type { Editor } from "@tiptap/core";
import { isInTable } from "@tiptap/pm/tables";
import { CHART_KINDS, type ChartKind } from "../../../shared/charts";
import type { MessageKey } from "../../../shared/i18n";
import { chartOfSelection, chartOfTable, insertChart, tableAround } from "../editor/chartCommands";
import { chartHandles } from "../editor/chartHandles";
import { noteHolding } from "../editor/noteKeymap";
import { languageNow } from "../i18n";
import type { Command, CommandContext } from "./registry";

/** The note the command acts in: the keymap gives it the Block at the cursor as its focus. */
function noteOf(ctx: CommandContext): Editor | null {
  const editor = ctx.focus?.kind === "block" ? noteHolding(ctx.focus.id) : null;
  return editor && !editor.isDestroyed && editor.isEditable ? editor : null;
}

/** The chart the command acts on: the one whose menu it is, or the Block selected in the note. */
const chartOf = (ctx: CommandContext) =>
  ctx.focus && (ctx.focus.kind === "chart" || ctx.focus.kind === "block")
    ? (chartHandles.get(ctx.focus.id) ?? null)
    : null;

const KIND_COMMANDS: Record<ChartKind, MessageKey> = {
  bar: "chart.command.bar",
  line: "chart.command.line",
  pie: "chart.command.pie",
  scatter: "chart.command.scatter",
};

/**
 * Charts (#436): one from the table the cursor is in (also in the table's
 * menu), or from the cited figures in the selected lines; and, for a chart,
 * its kind, its values as a table, an update from its table, Rename and
 * Delete, in its right-click menu and in ⌘K. Each is undone with ⌘Z.
 */
export const chartCommands: readonly Command[] = [
  {
    id: "chart.fromTable",
    label: "chart.fromTable",
    group: "main",
    appliesTo: ["block"],
    reversible: true,
    when: (ctx) => {
      const editor = noteOf(ctx);
      return editor !== null && isInTable(editor.state);
    },
    enabled(ctx) {
      const editor = noteOf(ctx);
      return editor && chartOfTable(editor.state, languageNow())
        ? true
        : { reason: "chart.fromTable.none" };
    },
    run(ctx) {
      const editor = noteOf(ctx);
      const chart = editor && chartOfTable(editor.state, languageNow());
      if (editor && chart) insertChart(editor, chart);
    },
  },
  {
    id: "chart.fromFigures",
    label: "chart.fromFigures",
    group: "main",
    appliesTo: ["block"],
    reversible: true,
    when: (ctx) => {
      const editor = noteOf(ctx);
      return editor !== null && !editor.state.selection.empty && !tableAround(editor.state);
    },
    enabled(ctx) {
      const editor = noteOf(ctx);
      return editor && chartOfSelection(editor.state, languageNow())
        ? true
        : { reason: "chart.fromFigures.none" };
    },
    run(ctx) {
      const editor = noteOf(ctx);
      const chart = editor && chartOfSelection(editor.state, languageNow());
      if (editor && chart) insertChart(editor, chart);
    },
  },
  {
    id: "chart.table",
    label: (ctx, t) => t(chartOf(ctx)?.asTable() ? "chart.showAsChart" : "chart.showAsTable"),
    group: "main",
    appliesTo: ["chart", "block"],
    // A way of looking at it: the same command puts it back.
    reversible: true,
    when: (ctx) => chartOf(ctx) !== null,
    run: (ctx) => chartOf(ctx)?.toggleTable(),
  },
  ...CHART_KINDS.map(
    (kind): Command => ({
      id: `chart.kind.${kind}`,
      label: KIND_COMMANDS[kind],
      group: "main",
      appliesTo: ["chart", "block"],
      reversible: true,
      // The kind it is already isn't offered.
      when: (ctx) => {
        const chart = chartOf(ctx);
        return chart !== null && chart.kind() !== kind;
      },
      enabled: (ctx) => (chartOf(ctx)?.fits(kind) ? true : { reason: "chart.scatter.needsTwo" }),
      run: (ctx) => chartOf(ctx)?.setKind(kind),
    }),
  ),
  {
    id: "chart.update",
    label: "chart.update.command",
    group: "main",
    appliesTo: ["chart", "block"],
    reversible: true,
    when: (ctx) => chartOf(ctx)?.canUpdate() ?? false,
    run: (ctx) => chartOf(ctx)?.update(),
  },
  {
    id: "chart.rename",
    label: "chart.rename",
    rowKeys: "F2",
    group: "edit",
    appliesTo: ["chart", "block"],
    reversible: true,
    when: (ctx) => chartOf(ctx) !== null,
    run: (ctx) => chartOf(ctx)?.rename(),
  },
];
