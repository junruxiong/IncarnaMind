import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { FOOTNOTE_NODE, QUESTION_BLOCK } from "../../../core/api";
import { countText, type TextCount } from "../../../shared/wordCount";

/**
 * The top-level Block types the count leaves out. Questions are what the User
 * asks, not part of what they write. Captions and the Bibliography would be
 * listed here (and offered as an option in the count's menu) once the editor
 * has such nodes. Footnotes are inline, and left out by `visibleText`.
 */
export const UNCOUNTED_BLOCKS: readonly string[] = [QUESTION_BLOCK];

/** A leaf's text: a line break is one, and a Citation marker or a formula is none. */
const leafText = (leaf: ProseMirrorNode) => (leaf.type.name === "hardBreak" ? "\n" : "");

/**
 * `doc.textBetween(from, to, "\n", leafText)`, but without the text a
 * footnote holds: only its marker is in the line, and a marker counts as none.
 */
function visibleText(doc: ProseMirrorNode, from: number, to: number): string {
  let text = "";
  let first = true;
  doc.nodesBetween(from, to, (node, pos) => {
    if (node.type.name === FOOTNOTE_NODE) return false;
    const nodeText = node.isText
      ? (node.text ?? "").slice(Math.max(from, pos) - pos, to - pos)
      : node.isLeaf
        ? leafText(node)
        : "";
    if (node.isBlock && ((node.isLeaf && nodeText) || node.isTextblock)) {
      if (first) first = false;
      else text += "\n";
    }
    text += nodeText;
    return true;
  });
  return text;
}

/**
 * The text of the Mind (or of the range `from`–`to` of it), Block by Block with
 * a line break between, without Citation markers, formulas or the uncounted Blocks.
 */
export function countableText(
  doc: ProseMirrorNode,
  range: { from: number; to: number } | null = null,
  uncounted: readonly string[] = UNCOUNTED_BLOCKS,
): string {
  const parts: string[] = [];
  doc.forEach((block, offset) => {
    if (uncounted.includes(block.type.name)) return;
    const start = offset;
    const end = offset + block.nodeSize;
    const from = range ? Math.max(range.from, start) : start;
    const to = range ? Math.min(range.to, end) : end;
    if (from >= to) return;
    parts.push(visibleText(doc, from, to));
  });
  return parts.join("\n");
}

export interface MindCount {
  /** The whole Mind. */
  total: TextCount;
  /** The selection, when there is one. */
  selection: TextCount | null;
}

/** Counts the Mind and, if the selection holds text, the selection. */
export function countMind(
  doc: ProseMirrorNode,
  selection: { from: number; to: number },
  uncounted: readonly string[] = UNCOUNTED_BLOCKS,
): MindCount {
  const selected =
    selection.from === selection.to ? null : countText(countableText(doc, selection, uncounted));
  return {
    total: countText(countableText(doc, null, uncounted)),
    selection: selected && selected.characters > 0 ? selected : null,
  };
}
