/**
 * "Add to note" (#201): an Answer in the thread, or the part of it the User
 * selected, goes into a Mind's note as Notes, Citations and all, each
 * Citation checked again where it lands. Into the Mind open in the editor it
 * goes at the cursor through the editor (so ⌘Z undoes it there too); into
 * another Mind, at its end through the core. Either way the Blocks keep a link
 * back to the Answer (src/shared/blockOrigins.ts).
 */
import type { Editor } from "@tiptap/core";
import { Fragment, type Node as ProseMirrorNode, type Slice } from "@tiptap/pm/model";
import type { EditorView } from "@tiptap/pm/view";
import { ANSWER_BLOCK, BLOCK_ID_ATTRIBUTE, NOTE_BLOCK_TYPES } from "../../../core/api";
import {
  type BlockJSON,
  type Recheck,
  recheckCitations,
  withoutIds,
} from "../../../shared/noteBlocks";
import { findBlock } from "./blockCommands";
import { type QuestionPlace, questionPlace } from "./composerAsk";

/** The Answer with this id in the thread's editor, and where its content starts and ends. */
function answerRange(doc: ProseMirrorNode, answerId: string) {
  const found = findBlock(doc, answerId);
  if (!found || found.node.type.name !== ANSWER_BLOCK) return null;
  return { node: found.node, from: found.pos + 1, to: found.pos + found.node.nodeSize - 1 };
}

/** The slice's nodes with the Answers it is open into unwrapped: Notes only. */
function notesOf(slice: Slice): BlockJSON[] {
  const nodes: ProseMirrorNode[] = [];
  slice.content.forEach((node) => {
    if (node.type.name !== ANSWER_BLOCK) {
      nodes.push(node);
      return;
    }
    node.content.forEach((child) => {
      nodes.push(child);
    });
  });
  return nodes
    .filter((node) => (NOTE_BLOCK_TYPES as readonly string[]).includes(node.type.name))
    .map((node) => node.toJSON() as BlockJSON);
}

/** The part of the Answer the User selected in the thread, if any of it is. */
function selectedIn(
  view: EditorView,
  from: number,
  to: number,
): { from: number; to: number } | null {
  const selection = view.dom.ownerDocument.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null;
  const range = selection.getRangeAt(0);
  if (!view.dom.contains(range.commonAncestorContainer)) return null;
  try {
    const start = Math.max(from, view.posAtDOM(range.startContainer, range.startOffset));
    const end = Math.min(to, view.posAtDOM(range.endContainer, range.endOffset));
    return start < end ? { from: start, to: end } : null;
  } catch {
    return null;
  }
}

/**
 * What adding the Answer puts into a note: the part of it selected in the
 * thread, or all of it, as Notes with their Citations. Null when the Answer
 * isn't there or has nothing to add.
 */
export function answerBlocks(view: EditorView, answerId: string): BlockJSON[] | null {
  const answer = answerRange(view.state.doc, answerId);
  if (!answer) return null;
  const selected = selectedIn(view, answer.from, answer.to);
  const blocks = selected
    ? // From the top, so the Answer around a selection inside one paragraph comes too.
      notesOf(view.state.doc.slice(selected.from, selected.to, true))
    : notesOf(view.state.doc.slice(answer.from - 1, answer.to + 1));
  const withText = blocks.filter((block) => JSON.stringify(block).includes('"text"'));
  return withText.length > 0 ? withText : null;
}

/** The Blocks without their ids, their Citations checked again (by `recheck`, the core's) now. */
export const checkedBlocks = (
  blocks: readonly BlockJSON[],
  recheck: Recheck,
): Promise<BlockJSON[]> => recheckCitations(withoutIds(blocks), recheck);

/** What adding put into the open Mind, so Undo can take exactly that back. */
export interface AddedHere {
  /** The new top-level Blocks' ids, in order. */
  blockIds: string[];
  /** The id of the empty line they took the place of, which Undo puts back; null if none. */
  emptyLine: string | null;
}

/**
 * Puts checked Blocks into the open Mind at `place` (see `questionPlace`):
 * at the cursor, or at the end. Null if the editor couldn't take them.
 */
export function addAtPlace(
  note: Editor,
  blocks: readonly BlockJSON[],
  place: QuestionPlace,
): AddedHere | null {
  const replaced = place.to > place.from ? note.state.doc.nodeAt(place.from) : null;
  const emptyLine = replaced?.attrs[BLOCK_ID_ATTRIBUTE];
  const ids = blocks.map(() => crypto.randomUUID());
  let nodes: ProseMirrorNode[];
  try {
    nodes = blocks.map((block, index) =>
      note.schema.nodeFromJSON({
        ...block,
        attrs: { ...block.attrs, [BLOCK_ID_ATTRIBUTE]: ids[index] },
      }),
    );
  } catch {
    return null;
  }
  const done = note.commands.command(({ tr }) => {
    tr.replaceWith(place.from, place.to, Fragment.from(nodes));
    return true;
  });
  return done
    ? { blockIds: ids, emptyLine: typeof emptyLine === "string" ? emptyLine : null }
    : null;
}

/** Where "Add to note" puts Blocks in the open Mind: at its cursor if it is there, else at its end. */
export const placeInNote = (note: Editor, atCursor: boolean): QuestionPlace =>
  questionPlace(note.state.doc, atCursor ? note.state.selection : null);

/** Takes the Blocks with these ids back out of the open Mind, and puts back the empty line they took. */
export function removeFromNote(note: Editor, added: AddedHere): void {
  note.commands.command(({ tr, state }) => {
    // From the end, so each position stays right.
    const found = added.blockIds
      .map((id) => findBlock(tr.doc, id))
      .filter((block) => block !== null)
      .sort((a, b) => b.pos - a.pos);
    if (found.length === 0) return false;
    for (const { pos, node } of found) tr.delete(pos, pos + node.nodeSize);
    const line = added.emptyLine
      ? state.schema.nodes.paragraph?.create({ [BLOCK_ID_ATTRIBUTE]: added.emptyLine })
      : null;
    const at = found.at(-1)?.pos;
    if (line && at !== undefined && !findBlock(tr.doc, added.emptyLine ?? "")) {
      tr.insert(Math.min(at, tr.doc.content.size), line);
    }
    return true;
  });
}
