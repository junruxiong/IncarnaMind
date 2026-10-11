import { type Editor, Extension, isMacOS, type JSONContent } from "@tiptap/core";
import { type EditorState, Plugin, PluginKey, TextSelection } from "@tiptap/pm/state";
import { CellSelection, cellAround, isInTable, rowIsHeader, selectedRect } from "@tiptap/pm/tables";
import { yUndoPluginKey } from "@tiptap/y-tiptap";
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
  label: MessageKey;
  /** Its key, in ProseMirror's names ("Mod-Alt-ArrowUp"), when it has one. */
  key?: string;
  run(editor: Editor): boolean;
}

/**
 * The table menu's actions, in its order: adding rows and columns (with
 * their keys, as in Confluence), the heading row, then deleting, last.
 */
export const TABLE_ACTIONS: readonly TableAction[] = [
  {
    id: "rowAbove",
    label: "editor.table.rowAbove",
    key: "Mod-Alt-ArrowUp",
    run: (editor) => editor.commands.addRowBefore(),
  },
  {
    id: "rowBelow",
    label: "editor.table.rowBelow",
    key: "Mod-Alt-ArrowDown",
    run: (editor) => editor.commands.addRowAfter(),
  },
  {
    id: "columnLeft",
    label: "editor.table.columnLeft",
    key: "Mod-Alt-ArrowLeft",
    run: (editor) => editor.commands.addColumnBefore(),
  },
  {
    id: "columnRight",
    label: "editor.table.columnRight",
    key: "Mod-Alt-ArrowRight",
    run: (editor) => editor.commands.addColumnAfter(),
  },
  {
    id: "headerRow",
    label: "editor.table.headerRow",
    run: (editor) => editor.commands.toggleHeaderRow(),
  },
  {
    id: "deleteRow",
    label: "editor.table.deleteRow",
    run: (editor) => editor.commands.deleteRow(),
  },
  {
    id: "deleteColumn",
    label: "editor.table.deleteColumn",
    run: (editor) => editor.commands.deleteColumn(),
  },
  {
    id: "deleteTable",
    label: "editor.table.delete",
    run: (editor) => editor.commands.deleteTable(),
  },
];

const KEY_SYMBOLS: Readonly<Record<string, string>> = {
  ArrowUp: "↑",
  ArrowDown: "↓",
  ArrowLeft: "←",
  ArrowRight: "→",
};

/** A key as a menu shows it: ⌘⌥↑ on macOS, Ctrl+Alt+↑ elsewhere. */
export function keyLabel(key: string): string {
  const parts = key.split("-");
  const last = parts.pop() ?? "";
  const symbol = KEY_SYMBOLS[last] ?? last;
  if (isMacOS()) {
    const mac: Readonly<Record<string, string>> = { Mod: "⌘", Alt: "⌥", Shift: "⇧" };
    return `${parts.map((part) => mac[part] ?? part).join("")}${symbol}`;
  }
  const other: Readonly<Record<string, string>> = { Mod: "Ctrl", Alt: "Alt", Shift: "Shift" };
  return [...parts.map((part) => other[part] ?? part), symbol].join("+");
}

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

/** The key that opens the table menu, as ⌘/ opens the menu over selected text. */
export const TABLE_MENU_KEY = "Mod-/";

/**
 * The table's keys and its right-click. In a table: the menu's keys add rows
 * and columns, ⌘/ opens the menu, and Tab and Shift-Tab never take the focus
 * out of the Mind (Tiptap's Table moves between cells with them, and adds a
 * row at the last cell). A right-click in a cell puts the cursor there,
 * unless it is on the selection, and opens the menu.
 */
export const TableKeys = Extension.create<Record<string, never>, TableKeysStorage>({
  name: "tableKeys",
  // After Tiptap's Table (and lists in cells): its Tab and Shift-Tab go first; these only keep the focus.
  priority: 50,

  addStorage() {
    return { openMenu: null };
  },

  addKeyboardShortcuts() {
    const inTable = () => isInTable(this.editor.state);
    const shortcuts: Record<string, () => boolean> = {
      [TABLE_MENU_KEY]: () => {
        const { selection } = this.editor.state;
        // Over selected text, ⌘/ is the selection menu's.
        if (!inTable() || !(selection.empty || selection instanceof CellSelection)) return false;
        const open = this.storage.openMenu;
        if (!open) return false;
        open(0);
        return true;
      },
      Tab: inTable,
      "Shift-Tab": inTable,
    };
    for (const action of TABLE_ACTIONS) {
      if (action.key)
        shortcuts[action.key] = () => inTable() && runTableAction(this.editor, action);
    }
    return shortcuts;
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
