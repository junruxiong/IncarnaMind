/**
 * When an edit the User asked for on the text applies at once (#880),
 * and what it leaves for them to decide:
 *
 * - Paragraphs of the selection the new text left out stay as they were,
 *   unless the User asked for something to go ("delete…", "remove…"); the
 *   User removes them with one click if they meant to.
 * - New text that can't be mapped onto the selection (none of its
 *   paragraphs is like one of the selection's) and would remove most of it
 *   doesn't apply: what was meant can't be told, so it waits as Proposals.
 */
import type { EditAction } from "../api";

/** Words that ask for text to go. */
const REMOVAL =
  /\b(delete|deletes|deleting|remove|removes|removing|cut|drop|erase|get rid of|take out|strip)\b|删除|删掉|删去|去掉|去除|移除|拿掉/i;

/** Whether the User asked for text to go, so what the new text leaves out goes too. */
export function asksToRemove(action: EditAction, instruction: string): boolean {
  return action === "instruction" && REMOVAL.test(instruction);
}

/** Beyond this share of the selected text going, new text that can't be mapped onto it waits. */
export const MOST_OF_IT = 0.5;

/**
 * Whether an edit waits for the User instead of applying: its new text
 * isn't like any of the selection's paragraphs (`mapped`), and what it would
 * remove, less what it adds, is more than half of the selected text, in
 * characters. A translation is mapped by its order, and never waits.
 */
export function waitsForUser(edit: {
  action: EditAction;
  mapped: boolean;
  removed: number;
  added: number;
  selected: number;
}): boolean {
  if (edit.mapped || edit.action === "translate" || edit.selected === 0) return false;
  return (edit.removed - edit.added) / edit.selected > MOST_OF_IT;
}
