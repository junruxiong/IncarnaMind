import type { EditAction } from "../api";

const REMOVAL =
  /\b(delete|deletes|deleting|remove|removes|removing|cut|drop|erase|get rid of|take out|strip)\b|删除|删掉|删去|去掉|去除|移除|拿掉/i;

/** Unless asked for, what an edit's new text leaves out is kept, as a model may return only what it changed (#880). */
export function asksToRemove(action: EditAction, instruction: string): boolean {
  return action === "instruction" && REMOVAL.test(instruction);
}

export const MOST_OF_IT = 0.5;

/** What was meant can't be told when new text maps onto none of the selection and removes most of it: it waits. */
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
