import { Editor, type JSONContent } from "@tiptap/core";
import { CellSelection } from "@tiptap/pm/tables";
import { describe, expect, onTestFinished, test } from "vitest";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { noteSlashItems } from "../../src/renderer/src/editor/slashItems";
import {
  emptyTable,
  keyLabel,
  runTableAction,
  TABLE_ACTIONS,
  type TableActionId,
  TableKeys,
  tableSpot,
} from "../../src/renderer/src/editor/tableCommands";

const paragraph = (text = ""): JSONContent =>
  text ? { type: "paragraph", content: [{ type: "text", text }] } : { type: "paragraph" };

/** A headless editor holding these top-level Blocks, with the table's keys. */
function editorWith(blocks: JSONContent[]): Editor {
  const editor = new Editor({
    element: null,
    extensions: [...noteExtensions(), TableKeys],
    content: { type: "doc", content: blocks },
  });
  onTestFinished(() => editor.destroy());
  return editor;
}

/** Each row's cells' text, a heading cell's marked with "#". */
function grid(editor: Editor): string[][] {
  const rows: string[][] = [];
  editor.state.doc.descendants((node) => {
    if (node.type.name !== "tableRow") return true;
    const cells: string[] = [];
    node.forEach((cell) => {
      cells.push(`${cell.type.name === "tableHeader" ? "#" : ""}${cell.textContent}`);
    });
    rows.push(cells);
    return false;
  });
  return rows;
}

/** The position inside the text of the cell at this row and column. */
function inCell(editor: Editor, row: number, column: number): number {
  let at = -1;
  let rowIndex = -1;
  editor.state.doc.descendants((node, pos) => {
    if (node.type.name === "tableRow") {
      rowIndex++;
      if (rowIndex !== row) return false;
      let offset = pos + 1;
      node.forEach((cell, cellOffset, index) => {
        if (index === column) at = pos + 1 + cellOffset + 2;
        offset += cell.nodeSize;
      });
      return false;
    }
    return true;
  });
  return at;
}

const action = (id: TableActionId) => {
  const found = TABLE_ACTIONS.find((each) => each.id === id);
  if (!found) throw new Error(`No table action ${id}.`);
  return found;
};

/** A table with a heading row, its cells named by row and column: "a1", "b2"… */
const named = (): JSONContent => {
  const table = emptyTable(3, 2);
  table.content?.forEach((row, rowIndex) => {
    row.content?.forEach((cell, column) => {
      cell.content = [paragraph(`${"ab"[column]}${rowIndex + 1}`)];
    });
  });
  return table;
};

describe("inserting a table", () => {
  test('"/" › Table puts a heading row over two rows of three cells, with the cursor in the first', () => {
    const editor = editorWith([paragraph("Before"), paragraph("/tab")]);
    const item = noteSlashItems.find((each) => each.id === "table");
    if (!item) throw new Error("No table in the slash menu.");
    const end = editor.state.doc.content.size - 1;
    editor.commands.setTextSelection(end);
    item.run(editor, { from: end - 4, to: end });

    const types: string[] = [];
    editor.state.doc.forEach((node) => {
      types.push(node.type.name);
    });
    // The line it was typed on becomes the table; a line after it to write on.
    expect(types).toEqual(["paragraph", "table", "paragraph"]);
    expect(grid(editor)).toEqual([
      ["#", "#", "#"],
      ["", "", ""],
      ["", "", ""],
    ]);
    expect(editor.state.selection.$head.parent.type.name).toBe("paragraph");
    expect(editor.state.selection.$head.node(-1).type.name).toBe("tableHeader");
    expect(tableSpot(editor.state)).toMatchObject({ rows: 1, columns: 1, headerRow: true });
  });

  test("the table menu's actions add and delete rows and columns where the cursor is, and switch the heading row", () => {
    const editor = editorWith([named(), paragraph("After")]);
    editor.commands.setTextSelection(inCell(editor, 1, 0));
    expect(editor.state.selection.$head.parent.textContent).toBe("a2");

    runTableAction(editor, action("rowBelow"));
    runTableAction(editor, action("columnRight"));
    expect(grid(editor)).toEqual([
      ["#a1", "#", "#b1"],
      ["a2", "", "b2"],
      ["", "", ""],
      ["a3", "", "b3"],
    ]);
    runTableAction(editor, action("rowAbove"));
    runTableAction(editor, action("columnLeft"));
    expect(grid(editor)[0]).toEqual(["#", "#a1", "#", "#b1"]);
    expect(grid(editor).map((row) => row[1])).toEqual(["#a1", "", "a2", "", "a3"]);

    // The row and the column the cursor is in go.
    editor.commands.setTextSelection(inCell(editor, 1, 0));
    runTableAction(editor, action("deleteRow"));
    editor.commands.setTextSelection(inCell(editor, 0, 0));
    runTableAction(editor, action("deleteColumn"));
    expect(grid(editor).map((row) => row.join(","))).toEqual([
      "#a1,#,#b1",
      "a2,,b2",
      ",,",
      "a3,,b3",
    ]);

    runTableAction(editor, action("headerRow"));
    expect(grid(editor)[0]).toEqual(["a1", "", "b1"]);
    expect(tableSpot(editor.state)?.headerRow).toBe(false);
    runTableAction(editor, action("headerRow"));
    expect(tableSpot(editor.state)?.headerRow).toBe(true);

    runTableAction(editor, action("deleteTable"));
    const types: string[] = [];
    editor.state.doc.forEach((node) => {
      types.push(node.type.name);
    });
    expect(types).not.toContain("table");
    expect(editor.state.doc.textContent).toBe("After");
  });

  test("a selection of cells counts its rows and columns; the table's last row or column can't be deleted", () => {
    const editor = editorWith([named()]);
    const first = inCell(editor, 0, 0) - 2;
    const last = inCell(editor, 2, 0) - 2;
    editor.view.dispatch(
      editor.state.tr.setSelection(CellSelection.create(editor.state.doc, first, last)),
    );
    expect(tableSpot(editor.state)).toMatchObject({
      rows: 3,
      columns: 1,
      canDeleteRows: false,
      canDeleteColumns: true,
    });
    expect(runTableAction(editor, action("deleteRow"))).toBe(false);
    expect(grid(editor)).toHaveLength(3);
    expect(runTableAction(editor, action("deleteColumn"))).toBe(true);
    expect(grid(editor)).toEqual([["#b1"], ["b2"], ["b3"]]);
    expect(tableSpot(editor.state)?.canDeleteColumns).toBe(false);
  });

  test("outside a table there is no table menu", () => {
    const editor = editorWith([paragraph("Plain")]);
    editor.commands.setTextSelection(2);
    expect(tableSpot(editor.state)).toBeNull();
  });

  test("the menu shows each key as the platform writes it", () => {
    expect(["⌘⌥↓", "Ctrl+Alt+↓"]).toContain(keyLabel("Mod-Alt-ArrowDown"));
    expect(["⌘/", "Ctrl+/"]).toContain(keyLabel("Mod-/"));
  });
});
