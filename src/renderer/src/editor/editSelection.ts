/**
 * Which text a selection in the note is, as Edit with AI changes it (#93): in
 * paragraphs and headings, by their Block ids and offsets into their text
 * (`TextPlace`), as the core reads it.
 */
import type { EditorState } from "@tiptap/pm/state";
import type { TextPlace } from "../../../core/api";

/** The text a selection covers, as Edit with AI changes it. */
export interface EditSelection {
  from: TextPlace;
  to: TextPlace;
  /** As the core compares it: an inline node as U+FFFC, a line per paragraph or heading. */
  text: string;
  /** Where it is in the editor. */
  range: { from: number; to: number };
}

const TEXT_BLOCKS: ReadonlySet<string> = new Set(["paragraph", "heading"]);
/** What Edit with AI doesn't change: a Question, code and formulas on their own line. */
const NOT_EDITED: ReadonlySet<string> = new Set(["question", "codeBlock", "blockMath"]);

/**
 * The text from `from` to `to` (the selection by default) as Edit with AI
 * changes it, or null: nothing but spaces is selected, or it starts or ends
 * outside a paragraph or heading (in a Question, in code), or an Answer
 * being written holds it.
 */
export function selectionForEdit(
  state: EditorState,
  range: { from: number; to: number } = state.selection,
): EditSelection | null {
  const { from, to } = range;
  if (from >= to) return null;
  const blocks: { pos: number; size: number; id: string; text: string }[] = [];
  let refused = false;
  state.doc.nodesBetween(from, to, (node, pos) => {
    if (refused) return false;
    if (node.type.name === "answer" && node.attrs.status === "streaming") refused = true;
    if (refused || NOT_EDITED.has(node.type.name)) return false;
    if (!node.isTextblock) return true;
    const id: unknown = node.attrs.id;
    if (TEXT_BLOCKS.has(node.type.name) && typeof id === "string") {
      const start = pos + 1;
      const size = node.content.size;
      const text = node.textBetween(Math.max(0, from - start), Math.min(size, to - start), "", "￼");
      blocks.push({ pos: start, size, id, text });
    }
    return false;
  });
  const first = blocks[0];
  const last = blocks.at(-1);
  if (refused || !first || !last) return null;
  const text = blocks.map((block) => block.text).join("\n");
  if (text.trim() === "") return null;
  const offset = (block: typeof first, at: number) =>
    Math.min(block.size, Math.max(0, at - block.pos));
  return {
    from: { blockId: first.id, offset: offset(first, from) },
    to: { blockId: last.id, offset: offset(last, to) },
    text,
    range: { from, to },
  };
}
