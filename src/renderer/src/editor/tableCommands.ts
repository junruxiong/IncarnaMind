import { type Editor, Extension, type JSONContent } from "@tiptap/core";
import { type EditorState, Plugin, PluginKey, TextSelection } from "@tiptap/pm/state";
import { CellSelection, cellAround, isInTable, rowIsHeader, selectedRect } from "@tiptap/pm/tables";
import { yUndoPluginKey } from "@tiptap/y-tiptap";
import type { CommandId, Keys } from "../../../shared/commands";
import type { MessageKey } from "../../../shared/i18n";

/** What the table menu can do to the table the cursor is in. */
export type TableActionId =
  | "rowAbove"
  | "rowBelow"
  | "columnLeft"
  | "columnRight"
  | "headerRow"
  | "deleteRow"
  | "deleteColumn"
  | "deleteTable";

export interface TableAction {
  id: TableActionId;
  /** Its command in the command list (../commands/tables), which binds its keys and is in ⌘K. */
  command: CommandId;
  label: MessageKey;
  /** Its keys, as an accelerator, when it has some. */
  keys?: Keys;
  run(editor: Editor): boolean;
}

/**
 * The table menu's actions, in its order: adding rows and columns (with
 * their keys, as in Confluence), the heading row, then deleting, last.
 */
export const TABLE_ACTIONS: readonly TableAction[] = [
  {
    id: "rowAbove",
    command: "table.rowAbove",
    label: "editor.table.rowAbove",
    keys: "CmdOrCtrl+Alt+Up",
    run: (editor) => editor.commands.addRowBefore(),
  },
  {
    id: "rowBelow",
    command: "table.rowBelow",
    label: "editor.table.rowBelow",
    keys: "CmdOrCtrl+Alt+Down",
    run: (editor) => editor.commands.addRowAfter(),
  },
  {
    id: "columnLeft",
    command: "table.columnLeft",
    label: "editor.table.columnLeft",
    keys: "CmdOrCtrl+Alt+Left",
    run: (editor) => editor.commands.addColumnBefore(),
  },
  {
    id: "columnRight",
    command: "table.columnRight",
    label: "editor.table.columnRight",
    keys: "CmdOrCtrl+Alt+Right",
    run: (editor) => editor.commands.addColumnAfter(),
  },
  {
    id: "headerRow",
    command: "table.headerRow",
    label: "editor.table.headerRow",
    run: (editor) => editor.commands.toggleHeaderRow(),
  },
  {
    id: "deleteRow",
    command: "table.deleteRow",
    label: "editor.table.deleteRow",
    run: (editor) => editor.commands.deleteRow(),
  },
  {
    id: "deleteColumn",
    command: "table.deleteColumn",
    label: "editor.table.deleteColumn",
    run: (editor) => editor.commands.deleteColumn(),
  },
  {
    id: "deleteTable",
    command: "table.delete",
    label: "editor.table.delete",
    run: (editor) => editor.commands.deleteTable(),
  },
];

/** The keys that open the table menu, as ⌘/ opens the menu over selected text. */
export const TABLE_MENU_KEYS: Keys = "CmdOrCtrl+/";

/** Runs a change as an Undo step of its own, rather than merged with the typing just before or after it. */
export function asOwnUndoStep<T>(editor: Editor, run: () => T): T {
  const undo = yUndoPluginKey.getState(editor.state)?.undoManager;
  undo?.stopCapturing();
  const result = run();
  undo?.stopCapturing();
  return result;
}

/**
 * Runs a table action as its own Undo step, unless an input method is
 * composing (what it is writing isn't in the cell yet), and puts the focus
 * back in the table. Whether it could be done.
 */
export function runTableAction(editor: Editor, action: TableAction): boolean {
  if (editor.view.composing) return false;
  const done = asOwnUndoStep(editor, () => action.run(editor));
  // Back in the cell, from the menu: the next key goes there.
  editor.commands.focus();
  return done;
}

/** The table the cursor is in, as its menu needs it. */
export interface TableSpot {
  /** The position just before the cell the menu sits on: the cursor's, or a cell selection's head. */
  cell: number;
  /** How many rows and columns the selection covers. */
  rows: number;
  columns: number;
  /** Whether the table's first row is its heading row. */
  headerRow: boolean;
  /** Whether the selected rows, or columns, are not all the table has: the last one can't go. */
  canDeleteRows: boolean;
  canDeleteColumns: boolean;
}

/** Where the cursor is in a table, or null when it isn't in one. */
export function tableSpot(state: EditorState): TableSpot | null {
  if (!isInTable(state)) return null;
  const { selection } = state;
  const $cell =
    selection instanceof CellSelection ? selection.$headCell : cellAround(selection.$head);
  if (!$cell) return null;
  const rect = selectedRect(state);
  return {
    cell: $cell.pos,
    rows: rect.bottom - rect.top,
    columns: rect.right - rect.left,
    headerRow: rowIsHeader(rect.map, rect.table, 0),
    canDeleteRows: rect.bottom - rect.top < rect.map.height,
    canDeleteColumns: rect.right - rect.left < rect.map.width,
  };
}

export const sameSpot = (a: TableSpot | null, b: TableSpot | null) =>
  a === b ||
  (a !== null &&
    b !== null &&
    a.cell === b.cell &&
    a.rows === b.rows &&
    a.columns === b.columns &&
    a.headerRow === b.headerRow &&
    a.canDeleteRows === b.canDeleteRows &&
    a.canDeleteColumns === b.canDeleteColumns);

/** Why a table action can't be done here now, or null. */
export function unavailable(id: TableActionId, spot: TableSpot): MessageKey | null {
  if (id === "deleteRow" && !spot.canDeleteRows) return "editor.table.lastRow";
  if (id === "deleteColumn" && !spot.canDeleteColumns) return "editor.table.lastColumn";
  return null;
}

/** Whether the table menu opens on ⌘/: the cursor is in a table, with no text selected (that's the selection menu's). */
export function menuKeyApplies(state: EditorState): boolean {
  const { selection } = state;
  return isInTable(state) && (selection.empty || selection instanceof CellSelection);
}

/** A new table: a heading row over empty rows, each cell an empty paragraph. */
export function emptyTable(rows = 3, columns = 3): JSONContent {
  const row = (type: "tableHeader" | "tableCell"): JSONContent => ({
    type: "tableRow",
    content: Array.from({ length: columns }, () => ({ type, content: [{ type: "paragraph" }] })),
  });
  return {
    type: "table",
    content: [row("tableHeader"), ...Array.from({ length: rows - 1 }, () => row("tableCell"))],
  };
}

export interface TableKeysStorage {
  /**
   * Opens the table menu (see `TableMenu`, which sets it), on the cell the
   * cursor is in; `buttons`: the mouse buttons still held, for a right-click.
   */
  openMenu: ((buttons: number) => void) | null;
}

declare module "@tiptap/core" {
  interface Storage {
    tableKeys: TableKeysStorage;
  }
}

/**
 * The table's own keys in the editor, and its right-click. Tiptap's Table
 * moves between cells with Tab and Shift-Tab, and adds a row at the last
 * cell; here they also never take the focus out of the Mind (at the first
 * cell, say). A right-click in a cell puts the cursor there, unless it is on
 * the selection, and opens the table menu. The menu's own keys are commands
 * (../commands/tables), in the one keymap.
 */
export const TableKeys = Extension.create<Record<string, never>, TableKeysStorage>({
  name: "tableKeys",
  // After Tiptap's Table (and lists in cells): their Tab and Shift-Tab go first.
  priority: 50,

  addStorage() {
    return { openMenu: null };
  },

  addKeyboardShortcuts() {
    const inTable = () => isInTable(this.editor.state);
    return { Tab: inTable, "Shift-Tab": inTable };
  },

  addProseMirrorPlugins() {
    const storage = this.storage;
    return [
      new Plugin({
        key: new PluginKey("tableRightClick"),
        props: {
          handleDOMEvents: {
            contextmenu: (view, event) => {
              const open = storage.openMenu;
              if (!open || !view.editable) return false;
              const at = view.posAtCoords({ left: event.clientX, top: event.clientY });
              const $cell = at && cellAround(view.state.doc.resolve(at.pos));
              if (!at || !$cell) return false;
              event.preventDefault();
              const { selection } = view.state;
              let onSelection = false;
              if (selection instanceof CellSelection) {
                selection.forEachCell((_cell, pos) => {
                  if (pos === $cell.pos) onSelection = true;
                });
              } else {
                const cellEnd = $cell.pos + ($cell.nodeAfter?.nodeSize ?? 0);
                onSelection = selection.from > $cell.pos && selection.to < cellEnd;
              }
              if (!onSelection) {
                view.dispatch(
                  view.state.tr.setSelection(TextSelection.near(view.state.doc.resolve(at.pos))),
                );
              }
              view.focus();
              open(event.buttons);
              return true;
            },
          },
        },
      }),
    ];
  },
});
