/**
 * Making and changing charts in the editor (#436): a chart from the table the
 * cursor is in, or from the cited figures in the selected lines, inserted
 * after the Block they are in; and its kind, its columns, its title and an
 * update from its table. Each is an Undo step of its own.
 */
import type { Editor } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, NodeSelection } from "@tiptap/pm/state";
import { TableMap } from "@tiptap/pm/tables";
import {
  BLOCK_ID_ATTRIBUTE,
  CHART_NODE,
  CITATION_NODE,
  type CitationAttributes,
  FOOTNOTE_NODE,
  PROPOSAL_ATTRIBUTE,
  PROPOSAL_INSERT_MARK,
} from "../../../core/api";
import {
  type ChartAttributes,
  type ChartData,
  type ChartKind,
  chartFromFigures,
  chartFromTable,
  drawnSeries,
  readChart,
  type SourceGrid,
  type SourceLine,
  sameChartData,
} from "../../../shared/charts";
import { readProposalAttribute } from "../../../shared/proposals";
import { asOwnUndoStep } from "./selectionCommands";

/** The keys a Citation keeps (`CitationAttributes`): a chart keeps a copy of each value's. */
const CITATION_KEYS: readonly (keyof CitationAttributes)[] = [
  "passageId",
  "documentId",
  "documentName",
  "contentHash",
  "pageFrom",
  "pageTo",
  "location",
  "quote",
  "check",
  "checkReason",
  "verdict",
];

function citationOf(node: ProseMirrorNode): CitationAttributes {
  const copy: Record<string, unknown> = {};
  for (const key of CITATION_KEYS) {
    const value: unknown = node.attrs[key];
    if (value !== undefined) copy[key] = value;
  }
  return copy as unknown as CitationAttributes;
}

/** Added by a Proposal still waiting: not the Mind's text yet. */
const proposed = (node: ProseMirrorNode) =>
  readProposalAttribute(node.attrs[PROPOSAL_ATTRIBUTE])?.kind === "insert" ||
  node.marks.some((mark) => mark.type.name === PROPOSAL_INSERT_MARK);

/** The text of a cell or a line, without its Citations' markers or its footnotes, and its first Citation. */
function readText(node: ProseMirrorNode): SourceLine {
  let text = "";
  let citation: CitationAttributes | null = null;
  node.descendants((child) => {
    if (proposed(child)) return false;
    if (child.type.name === CITATION_NODE) {
      citation ??= citationOf(child);
      return false;
    }
    if (child.type.name === FOOTNOTE_NODE) return false;
    if (child.isText) text += child.text ?? "";
    else if (child.type.name === "hardBreak" || (child.isBlock && text !== "")) text += " ";
    return true;
  });
  return { text: text.replace(/\s+/g, " ").trim(), citation };
}

/** A table laid out on its grid, as a chart reads it: a cell spanning rows or columns is in each of its slots. */
export function tableGrid(table: ProseMirrorNode): SourceGrid {
  const map = TableMap.get(table);
  const seen = new Set<number>();
  const grid: SourceGrid[number][] = [];
  for (let row = 0; row < map.height; row++) {
    const slots: SourceGrid[number][number][] = [];
    for (let column = 0; column < map.width; column++) {
      const offset = map.map[row * map.width + column];
      const cell = offset === undefined ? null : table.nodeAt(offset);
      if (!cell || offset === undefined) {
        slots.push(null);
        continue;
      }
      slots.push({
        ...readText(cell),
        header: cell.type.name === "tableHeader",
        continued: seen.has(offset),
      });
      seen.add(offset);
    }
    grid.push(slots);
  }
  return grid;
}

/** The table the selection is in, and where it starts. */
export function tableAround(state: EditorState): { node: ProseMirrorNode; pos: number } | null {
  const { $from } = state.selection;
  for (let depth = $from.depth; depth > 0; depth--) {
    const node = $from.node(depth);
    if (node.type.name === "table") return { node, pos: $from.before(depth) };
  }
  return null;
}

/** The text of the heading nearest above a top-level Block: the chart's title until the User names it. */
function headingAbove(doc: ProseMirrorNode, topLevel: number): string {
  let title = "";
  doc.forEach((node, _offset, index) => {
    if (index < topLevel && node.type.name === "heading") title = node.textContent;
  });
  return title;
}

/** The chart the table the cursor is in makes, or null when no column of it holds a number. */
export function chartOfTable(state: EditorState, language: string): ChartAttributes | null {
  const table = tableAround(state);
  if (!table) return null;
  const id = table.node.attrs[BLOCK_ID_ATTRIBUTE];
  return chartFromTable(tableGrid(table.node), {
    blockId: typeof id === "string" ? id : null,
    title: headingAbove(state.doc, state.doc.resolve(table.pos).index(0)),
    language,
  });
}

/** The lines of text the selection touches: each paragraph and heading, a list item's text included. */
function selectedLines(state: EditorState): SourceLine[] {
  const { from, to } = state.selection;
  const lines: SourceLine[] = [];
  state.doc.nodesBetween(from, to, (node) => {
    if (node.type.name === "table" || node.type.name === CHART_NODE) return false;
    if (!node.isTextblock) return true;
    lines.push(readText(node));
    return false;
  });
  return lines;
}

/** The chart the cited figures in the selected lines make, or null when none of them holds a figure. */
export function chartOfSelection(state: EditorState, language: string): ChartAttributes | null {
  const { selection } = state;
  if (selection.empty || tableAround(state)) return null;
  const lines = selectedLines(state);
  return chartFromFigures(lines, {
    title: headingAbove(state.doc, state.doc.resolve(selection.from).index(0)),
    language,
  });
}

/** The chart as the node's attributes. */
const attributesOf = (chart: ChartAttributes) => ({
  kind: chart.kind,
  title: chart.title,
  drawn: chart.drawn,
  data: chart.data,
  source: chart.source,
});

/**
 * Inserts a chart after the top-level Block the selection ends in (a table in
 * an Answer: after the Answer), selects it and brings it into view.
 */
export function insertChart(editor: Editor, chart: ChartAttributes): boolean {
  if (editor.view.composing) return false;
  const { state } = editor;
  const $end = state.selection.$to;
  const after = $end.depth > 0 ? $end.after(1) : $end.pos;
  const type = state.schema.nodes[CHART_NODE];
  if (!type) return false;
  return asOwnUndoStep(editor, () => {
    const node = type.create(attributesOf(chart));
    const tr = state.tr.insert(after, node);
    tr.setSelection(NodeSelection.create(tr.doc, after)).scrollIntoView();
    editor.view.dispatch(tr);
    editor.commands.focus();
    return true;
  });
}

/** Changes a chart's attributes in one step: its kind, the columns drawn, its title, or its data. */
export function changeChart(
  editor: Editor,
  pos: number,
  change: Partial<Pick<ChartAttributes, "kind" | "title" | "drawn" | "data">>,
): boolean {
  const node = editor.state.doc.nodeAt(pos);
  if (!node || node.type.name !== CHART_NODE || !editor.isEditable) return false;
  return asOwnUndoStep(editor, () => {
    editor.view.dispatch(
      editor.state.tr.setNodeMarkup(pos, undefined, { ...node.attrs, ...change }),
    );
    return true;
  });
}

/**
 * The columns drawn once `series` is turned on or off, keeping the others in
 * the table's order: a pie draws only the one chosen, a scatter chart two at
 * most, the first kept with the one chosen.
 */
export function toggledSeries(chart: ChartAttributes, series: number): number[] {
  const all = chart.data.series.map((_, index) => index);
  if (chart.kind === "pie") return [series];
  const drawn = chart.drawn ?? all;
  if (drawn.includes(series)) return drawn.filter((each) => each !== series);
  const kept = chart.kind === "scatter" ? drawnSeries(chart).slice(0, 1) : drawn;
  return all.filter((each) => each === series || kept.includes(each));
}

/** The kind a chart switches to, and the columns it then draws: a scatter chart needs two. */
export function withKind(
  chart: ChartAttributes,
  kind: ChartKind,
): Pick<ChartAttributes, "kind" | "drawn"> {
  const all = chart.data.series.map((_, index) => index);
  const drawn = chart.drawn ?? all;
  if (kind === "scatter" && drawn.length < 2) {
    return { kind, drawn: [...new Set([...drawn, ...all])].slice(0, 2).sort((a, b) => a - b) };
  }
  return { kind, drawn: chart.drawn };
}

/** The table a chart was made from, where it is now in the Mind, if it is still there. */
export function chartTable(
  doc: ProseMirrorNode,
  chart: Pick<ChartAttributes, "source">,
): ProseMirrorNode | null {
  const { source } = chart;
  if (source.kind !== "table" || !source.blockId) return null;
  let found: ProseMirrorNode | null = null;
  doc.descendants((node) => {
    if (found) return false;
    if (node.type.name === "table" && node.attrs[BLOCK_ID_ATTRIBUTE] === source.blockId) {
      found = node;
      return false;
    }
    return node.isBlock && !node.isTextblock;
  });
  return found;
}

/**
 * What the chart's table would give now, when it differs from what the chart
 * draws: the data to update it with. Null when the table is gone, unchanged,
 * or no longer has a column of numbers.
 */
export function tableUpdate(
  doc: ProseMirrorNode,
  attributes: Readonly<Record<string, unknown>>,
  language: string,
): ChartData | null {
  const chart = readChart(attributes);
  const table = chartTable(doc, chart);
  if (!table) return null;
  const fresh = chartFromTable(tableGrid(table), {
    blockId: chart.source.kind === "table" ? chart.source.blockId : null,
    title: chart.title,
    language,
  });
  if (!fresh) return null;
  return sameChartData(fresh.data, chart.data) ? null : fresh.data;
}
