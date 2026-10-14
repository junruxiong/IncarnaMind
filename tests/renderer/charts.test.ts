import {
  Editor,
  getSchema,
  getTextBetween,
  getTextSerializersFromSchema,
  type JSONContent,
} from "@tiptap/core";
import { DOMSerializer, DOMParser as ProseMirrorDOMParser } from "@tiptap/pm/model";
import { NodeSelection } from "@tiptap/pm/state";
import { parseHTML } from "linkedom";
import { describe, expect, onTestFinished, test } from "vitest";
import { CHART_NODE, CITATION_NODE } from "../../src/core";
import {
  changeChart,
  chartOfSelection,
  chartOfTable,
  insertChart,
  tableUpdate,
  toggledSeries,
  withKind,
} from "../../src/renderer/src/editor/chartCommands";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { type ChartAttributes, readChart } from "../../src/shared/charts";

const text = (value: string): JSONContent => ({ type: "text", text: value });
const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });
const citation = (page: number): JSONContent => ({
  type: CITATION_NODE,
  attrs: {
    passageId: "p",
    documentId: "acme",
    documentName: "Acme",
    pageFrom: page,
    pageTo: page,
    quote: `Revenue on page ${page}`,
    check: "found",
  },
});
const cell = (type: "tableHeader" | "tableCell", ...content: JSONContent[]): JSONContent => ({
  type,
  attrs: { colspan: 1, rowspan: 1 },
  content: [paragraph(...content)],
});

/** Revenue by quarter: a heading row, then a figure and its Citation in each cell but the last. */
const QUARTERS: JSONContent = {
  type: "table",
  attrs: { id: "table-1" },
  content: [
    {
      type: "tableRow",
      content: [
        cell("tableHeader", text("Quarter")),
        cell("tableHeader", text("2024")),
        cell("tableHeader", text("2025")),
      ],
    },
    {
      type: "tableRow",
      content: [
        cell("tableCell", text("Q1")),
        cell("tableCell", text("289.4 "), citation(1)),
        cell("tableCell", text("341.5 "), citation(2)),
      ],
    },
    {
      type: "tableRow",
      content: [
        cell("tableCell", text("Q2")),
        cell("tableCell", text("301.2 "), citation(1)),
        cell("tableCell", text("362.0")),
      ],
    },
  ],
};

function editorWith(blocks: JSONContent[]): Editor {
  const editor = new Editor({
    element: null,
    extensions: noteExtensions(),
    content: { type: "doc", content: blocks },
  });
  onTestFinished(() => editor.destroy());
  return editor;
}

/** Puts the cursor in the first cell whose text starts with `start`. */
function cursorIn(editor: Editor, start: string): void {
  let at = -1;
  editor.state.doc.descendants((node, pos) => {
    if (at < 0 && node.isTextblock && node.textContent.startsWith(start)) at = pos + 1;
    return at < 0;
  });
  editor.commands.setTextSelection(at);
}

/** The top-level Blocks' types, an Answer's as "answer[…]" with its own. */
function outline(editor: Editor): string[] {
  const types: string[] = [];
  editor.state.doc.forEach((node) => {
    if (node.type.name !== "answer") types.push(node.type.name);
    else {
      const inner: string[] = [];
      node.forEach((child) => {
        inner.push(child.type.name);
      });
      types.push(`answer[${inner.join(",")}]`);
    }
  });
  return types;
}

/** The chart in the editor, and where it is. */
function chartAt(editor: Editor): { chart: ChartAttributes; pos: number } {
  let found: { chart: ChartAttributes; pos: number } | null = null;
  editor.state.doc.descendants((node, pos) => {
    if (!found && node.type.name === CHART_NODE) found = { chart: readChart(node.attrs), pos };
    return !found;
  });
  if (!found) throw new Error("No chart in the editor.");
  return found;
}

describe("Chart from table", () => {
  test("inserts a bar chart of the table after it, selected, its values with their cells and Citations", () => {
    const editor = editorWith([
      { type: "heading", attrs: { level: 2 }, content: [text("Quarter by quarter")] },
      QUARTERS,
      paragraph(text("After")),
    ]);
    cursorIn(editor, "Q2");
    const chart = chartOfTable(editor.state, "en");
    expect(chart).not.toBeNull();
    expect(insertChart(editor, chart as ChartAttributes)).toBe(true);

    expect(outline(editor)).toEqual(["heading", "table", CHART_NODE, "paragraph"]);
    const { chart: inserted, pos } = chartAt(editor);
    expect(editor.state.selection).toBeInstanceOf(NodeSelection);
    expect(editor.state.selection.from).toBe(pos);
    // Named after the heading above it, until the User names it.
    expect(inserted.title).toBe("Quarter by quarter");
    expect(inserted.source).toEqual({ kind: "table", blockId: "table-1" });
    expect(inserted.data.series).toEqual(["2024", "2025"]);
    const q1 = inserted.data.rows[0]?.values[1];
    expect(q1).toMatchObject({
      shown: "341.5",
      value: 341.5,
      from: { kind: "cell", row: 1, column: 2 },
    });
    expect(q1?.citation).toMatchObject({ documentName: "Acme", pageFrom: 2, check: "found" });
    // The Citation is kept as a Citation keeps itself: no editor-only attributes.
    expect(Object.keys(q1?.citation ?? {})).not.toContain("proposal");
    expect(inserted.data.rows[1]?.values[1]?.citation).toBeNull();
  });

  test("a table in an Answer gets its chart after the Answer", () => {
    const editor = editorWith([
      { type: "question", attrs: { id: "q" }, content: [text("Revenue?")] },
      { type: "answer", attrs: { questionId: "q", status: "done" }, content: [QUARTERS] },
      paragraph(),
    ]);
    cursorIn(editor, "Q1");
    insertChart(editor, chartOfTable(editor.state, "en") as ChartAttributes);
    expect(outline(editor)).toEqual(["question", "answer[table]", CHART_NODE, "paragraph"]);
  });

  test("isn't offered outside a table, nor for a table with no column of numbers", () => {
    const editor = editorWith([
      paragraph(text("Text")),
      {
        type: "table",
        content: [
          {
            type: "tableRow",
            content: [cell("tableHeader", text("Claim")), cell("tableHeader", text("Where"))],
          },
          {
            type: "tableRow",
            content: [cell("tableCell", text("Spring")), cell("tableCell", text("Docs"))],
          },
        ],
      },
    ]);
    cursorIn(editor, "Text");
    expect(chartOfTable(editor.state, "en")).toBeNull();
    cursorIn(editor, "Spring");
    expect(chartOfTable(editor.state, "en")).toBeNull();
  });
});

describe("Chart from selected figures", () => {
  test("charts the figure in each selected line, with the line's Citation", () => {
    const editor = editorWith([
      paragraph(text("Q1 2025: $341.5 million "), citation(2)),
      {
        type: "bulletList",
        content: [{ type: "listItem", content: [paragraph(text("Q2 2025: $362.0 million"))] }],
      },
      paragraph(text("No figure here.")),
    ]);
    editor.commands.setTextSelection({ from: 1, to: editor.state.doc.content.size - 1 });
    const chart = chartOfSelection(editor.state, "en") as ChartAttributes;
    expect(chart.source).toEqual({ kind: "figures" });
    expect(chart.data.rows.map((row) => [row.label, row.values[0]?.shown])).toEqual([
      ["Q1 2025", "$341.5 million"],
      ["Q2 2025", "$362.0 million"],
    ]);
    expect(chart.data.rows[0]?.values[0]?.citation?.pageFrom).toBe(2);
    insertChart(editor, chart);
    expect(outline(editor)).toEqual(["paragraph", "bulletList", "paragraph", CHART_NODE]);
  });

  test("isn't offered with nothing selected, or no figure in what is", () => {
    const editor = editorWith([paragraph(text("No figure here."))]);
    expect(chartOfSelection(editor.state, "en")).toBeNull();
    editor.commands.setTextSelection({ from: 1, to: 10 });
    expect(chartOfSelection(editor.state, "en")).toBeNull();
  });
});

describe("changing a chart in place", () => {
  const made = () => {
    const editor = editorWith([QUARTERS, paragraph()]);
    cursorIn(editor, "Q1");
    insertChart(editor, chartOfTable(editor.state, "en") as ChartAttributes);
    return editor;
  };

  test("its kind, its columns, and its title, each a change of its own", () => {
    const editor = made();
    let { chart, pos } = chartAt(editor);
    changeChart(editor, pos, withKind(chart, "line"));
    expect(chartAt(editor).chart.kind).toBe("line");

    // A column off, and on again in its place in the table.
    ({ chart, pos } = chartAt(editor));
    changeChart(editor, pos, { drawn: toggledSeries(chart, 0) });
    expect(chartAt(editor).chart.drawn).toEqual([1]);
    ({ chart, pos } = chartAt(editor));
    changeChart(editor, pos, { drawn: toggledSeries(chart, 0) });
    expect(chartAt(editor).chart.drawn).toEqual([0, 1]);

    // A pie draws the one chosen.
    ({ chart, pos } = chartAt(editor));
    changeChart(editor, pos, withKind(chart, "pie"));
    ({ chart, pos } = chartAt(editor));
    expect(toggledSeries(chart, 0)).toEqual([0]);

    changeChart(editor, pos, { title: "Revenue, $ million" });
    expect(chartAt(editor).chart.title).toBe("Revenue, $ million");
  });

  test("a scatter chart takes two columns, keeping the first chosen", () => {
    const editor = made();
    const { chart } = chartAt(editor);
    expect(withKind({ ...chart, drawn: [1] }, "scatter")).toEqual({
      kind: "scatter",
      drawn: [0, 1],
    });
    expect(toggledSeries({ ...chart, kind: "scatter", drawn: [0, 1] }, 1)).toEqual([0]);
  });

  test("knows when its table changed since, and what it would draw now", () => {
    const editor = made();
    const node = () => editor.state.doc.nodeAt(chartAt(editor).pos);
    expect(tableUpdate(editor.state.doc, node()?.attrs ?? {}, "en")).toBeNull();

    // A figure in the table is corrected.
    let at = -1;
    editor.state.doc.descendants((child, pos) => {
      if (at < 0 && child.isText && child.text === "362.0") at = pos;
      return at < 0;
    });
    editor.view.dispatch(editor.state.tr.insertText("365.0", at, at + 5));
    const update = tableUpdate(editor.state.doc, node()?.attrs ?? {}, "en");
    expect(update?.rows[1]?.values[1]?.shown).toBe("365.0");

    changeChart(editor, chartAt(editor).pos, { data: update ?? undefined });
    expect(chartAt(editor).chart.data.rows[1]?.values[1]?.value).toBe(365);
    expect(tableUpdate(editor.state.doc, node()?.attrs ?? {}, "en")).toBeNull();
  });
});

describe("a chart copied", () => {
  test("as HTML keeps everything it holds, and reads as its title and a table elsewhere", () => {
    const editor = editorWith([QUARTERS, paragraph()]);
    cursorIn(editor, "Q1");
    insertChart(editor, chartOfTable(editor.state, "en") as ChartAttributes);
    const { pos } = chartAt(editor);
    const node = editor.state.doc.nodeAt(pos);
    if (!node) throw new Error("No chart.");

    const { document } = parseHTML("<!doctype html><html><body></body></html>");
    const schema = getSchema(noteExtensions());
    const dom = DOMSerializer.fromSchema(schema).serializeNode(node, {
      document,
    }) as unknown as HTMLElement;
    expect(dom.querySelector("table")?.textContent).toContain("Q1289.4341.5");

    document.body.append(dom);
    const parsed = ProseMirrorDOMParser.fromSchema(schema).parse(document.body as unknown as Node);
    const copy = parsed.firstChild;
    expect(copy?.type.name).toBe(CHART_NODE);
    expect(readChart(copy?.attrs ?? {})).toEqual(readChart(node.attrs));
  });

  test("as text is its title and its values, each with its source", () => {
    const editor = editorWith([QUARTERS, paragraph()]);
    cursorIn(editor, "Q1");
    insertChart(editor, chartOfTable(editor.state, "en") as ChartAttributes);
    const { pos } = chartAt(editor);
    const node = editor.state.doc.nodeAt(pos);
    // What the clipboard's text is, as Tiptap writes it from each node's `renderText`.
    const copied = getTextBetween(
      editor.state.doc,
      { from: pos, to: pos + (node?.nodeSize ?? 0) },
      { textSerializers: getTextSerializersFromSchema(editor.schema) },
    );
    expect(copied).toBe(
      [
        "Quarter\t2024\t2025",
        "Q1\t289.4 [Acme, p. 1]\t341.5 [Acme, p. 2]",
        "Q2\t301.2 [Acme, p. 1]\t362.0",
      ].join("\n"),
    );
  });
});
