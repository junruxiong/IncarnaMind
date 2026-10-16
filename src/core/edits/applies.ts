import type { EditAction } from "../api";

const REMOVE =
  /\b(?:delete|deleting|remove|removing|cut|cutting|drop|dropping|erase|get rid of|take out|strip)\b|删除|删掉|删去|去掉|去除|移除|拿掉/giu;
/**
 * A part of the text by its kind: "the second paragraph", "the Methods section",
 * 第二段, 最后一句. Words in the text ("these filler words", 所有形容词) are no
 * part, so taking them out never takes out what the model left out.
 */
const PART =
  /\b(?:paragraphs?|sections?|headings?|sentences?|bullets?|bullet points?|tables?)\b|段落|章节|句子|标题|表格|要点|(?:第[\d一二三四五六七八九十百]+|最后[\d一二两三四五六七八九十]*|[前后][\d一二两三四五六七八九十]+|[这那此该本上下][一二两]?)[段节句条]/iu;
/** Saying not to, before the verb in its own clause: "but keep the rest" after it doesn't undo it. */
const NOT =
  /\b(?:don'?t|do not|never|without|no need to|not)\b|不要|别|不用|不必|无需|不需要|没必要|勿/iu;
const CLAUSE = /[,;:.!?，；：。！？]|\b(?:but|then)\b|但是?/iu;
/** "the first and second paragraphs" is one part; "the jargon and tighten the paragraph" isn't. */
const ORDINAL_AND =
  /\b(?:and|or)\s+(?=(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|last|next|\d+(?:st|nd|rd|th)?)\b)/giu;

/** Unless asked for, what an edit's new text leaves out is kept, as a model may return only what it changed (#880). */
export function asksToRemove(action: EditAction, instruction: string): boolean {
  if (action !== "instruction") return false;
  const asked = instruction.replace(ORDINAL_AND, "");
  for (const verb of asked.matchAll(REMOVE)) {
    const at = verb.index ?? 0;
    const opening = asked.slice(0, at).split(CLAUSE).at(-1) ?? "";
    if (NOT.test(opening)) continue;
    const rest = (asked.slice(at + verb[0].length).split(CLAUSE)[0] ?? "").split(/\band\b/iu)[0];
    // The object, not where it is: "the adjectives across all paragraphs", "第二段里的口语化表达".
    const object = (words: string) =>
      (
        words.split(
          /\b(?:in|of|from|within|across|throughout|over|on|at|for|each|every|per)\b|从|每|各/iu,
        )[0] ?? ""
      )
        .split("的")
        .at(-1) ?? "";
    // Chinese puts the object before the verb: 把第二段删掉.
    const before = /[把将]([^把将]*)$/u.exec(opening)?.[1] ?? "";
    if (PART.test(object(rest ?? "")) || PART.test(object(before))) return true;
  }
  return false;
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
