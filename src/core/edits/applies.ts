import type { EditAction } from "../api";

const REMOVE =
  /\b(?:delete|remove|cut|drop|erase|get rid of|take out|strip)\b|删除|删掉|删去|去掉|去除|移除|拿掉/i;
/** Whole parts of the text, not words in it: "remove the jargon" doesn't take paragraphs out. */
const PART =
  /\b(?:paragraphs?|sections?|sentences?|lines?|bullets?|items?|points?|headings?|parts?|this|that|these|those|it|them|all|everything|rest)\b|段|句|部分|这|那|全部|所有|其余|标题|条/i;
const NOT = /\b(?:don'?t|do not|never|without|keep|leave)\b|不要|别|保留|不用|不必|勿/i;

/** Unless asked for, what an edit's new text leaves out is kept, as a model may return only what it changed (#880). */
export function asksToRemove(action: EditAction, instruction: string): boolean {
  if (action !== "instruction" || NOT.test(instruction)) return false;
  const verb = REMOVE.exec(instruction);
  if (!verb) return false;
  // The object, not where it is: "the jargon in this paragraph", "第二段里的口语化表达".
  const object = (words: string) =>
    (words.split(/\b(?:in|of|from|within)\b|从/)[0] ?? "").split("的").at(-1) ?? "";
  const after = instruction.slice(verb.index + verb[0].length);
  // Chinese puts the object before the verb: 把第二段删掉.
  const before = /[把将](.*)$/u.exec(instruction.slice(0, verb.index))?.[1] ?? "";
  return PART.test(object(after)) || PART.test(object(before));
}

export const MOST_OF_IT = 0.5;

/** New text maps onto the selection when at least half of it is found there. */
export function mapsOnto(found: number, total: number): boolean {
  return total === 0 || found / total >= MOST_OF_IT;
}

/** What was meant can't be told when new text maps onto none of the selection and removes most of it: it waits. */
export function waitsForUser(edit: {
  action: EditAction;
  removing: boolean;
  mapped: boolean;
  removed: number;
  added: number;
  selected: number;
}): boolean {
  if (edit.removing || edit.mapped || edit.action === "translate" || edit.selected === 0) {
    return false;
  }
  return (edit.removed - edit.added) / edit.selected > MOST_OF_IT;
}
