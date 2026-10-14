import type { Editor } from "@tiptap/core";
import { COMPARISON_CELL_NODE, COMPARISON_TABLE_NODE } from "../../../core/api";
import { comparisonQuestionsOf, MAX_COMPARISON_QUESTIONS } from "../../../shared/comparisons";

/** Where a comparison table is in the editor, by its id; null when it isn't there. */
export function comparisonTableAt(editor: Editor, tableId: string): number | null {
  let found: number | null = null;
  editor.state.doc.forEach((block, offset) => {
    if (block.type.name === COMPARISON_TABLE_NODE && block.attrs.id === tableId) found = offset;
  });
  return found;
}

/**
 * "+ Question" (#98): a column for `question` at the table's end, and in each
 * row a cell waiting to be filled, as one change ⌘Z takes back. False when
 * the table is gone or has as many columns as a table holds.
 */
export function addComparisonColumn(editor: Editor, tableId: string, question: string): boolean {
  const pos = comparisonTableAt(editor, tableId);
  const table = pos === null ? null : editor.state.doc.nodeAt(pos);
  const text = question.replace(/\s+/g, " ").trim();
  if (pos === null || !table || !text) return false;
  const questions = comparisonQuestionsOf(table.attrs);
  if (questions.length >= MAX_COMPARISON_QUESTIONS) return false;
  const { schema } = editor.state;
  const id = crypto.randomUUID();
  const tr = editor.state.tr.setNodeAttribute(pos, "questions", [...questions, { id, text }]);
  table.forEach((row, offset) => {
    const end = pos + 1 + offset + row.nodeSize - 1;
    const cell = schema.nodes[COMPARISON_CELL_NODE]?.create(
      { questionId: id, status: "waiting" },
      schema.nodes.paragraph?.create(),
    );
    if (cell) tr.insert(tr.mapping.map(end), cell);
  });
  editor.view.dispatch(tr);
  return true;
}
