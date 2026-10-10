/**
 * Putting a Question asked in the composer into the note (DESIGN.md,
 * Composer): one grey line at the cursor, or at the end of the Mind when the
 * cursor isn't in it; the core then writes its Answer right below it.
 */
import type { Editor, JSONContent } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { NodeSelection, type Selection, TextSelection } from "@tiptap/pm/state";
import {
  ANSWER_BLOCK,
  BLOCK_ID_ATTRIBUTE,
  type ChatModelChoice,
  QUESTION_BLOCK,
  type QuestionAttributes,
  type SearchScope,
} from "../../../core/api";
import { scopeAttributesOf } from "../../../shared/searchScope";
import { findBlock } from "./blockCommands";

/** Where a Question goes: it replaces `from`–`to` (an empty line), or goes in at `from` when they are equal. */
export interface QuestionPlace {
  from: number;
  to: number;
}

const isEmptyLine = (node: ProseMirrorNode | null | undefined) =>
  node?.type.name === "paragraph" && node.content.size === 0;

/** At the end of the Mind: on its empty last line, or after its last Block. */
function atEnd(doc: ProseMirrorNode): QuestionPlace {
  const size = doc.content.size;
  const last = doc.lastChild;
  return isEmptyLine(last) && last
    ? { from: size - last.nodeSize, to: size }
    : { from: size, to: size };
}

/**
 * Where the Question asked now goes, given the Mind's selection, or null when
 * the cursor isn't in the Mind: at its end then. On an empty line it takes the
 * line's place; in any other Block it goes after the Block, never splitting
 * the text; in a Question, after that Question's Answer.
 */
export function questionPlace(doc: ProseMirrorNode, selection: Selection | null): QuestionPlace {
  if (!selection) return atEnd(doc);
  const { $to } = selection;
  // Between Blocks, or a whole Block selected: right there, or after it.
  if ($to.depth === 0) {
    const at = selection instanceof NodeSelection ? selection.to : $to.pos;
    return { from: at, to: at };
  }
  const index = $to.index(0);
  const block = doc.child(index);
  const start = $to.before(1);
  const end = start + block.nodeSize;
  if (isEmptyLine(block)) return { from: start, to: end };
  if (block.type.name === QUESTION_BLOCK && index + 1 < doc.childCount) {
    const next = doc.child(index + 1);
    const answers =
      next.type.name === ANSWER_BLOCK && next.attrs.questionId === block.attrs[BLOCK_ID_ATTRIBUTE];
    if (answers) return { from: end + next.nodeSize, to: end + next.nodeSize };
  }
  return { from: end, to: end };
}

/** A Question's text as its inline content: lines joined by line breaks. */
export function questionContent(text: string): JSONContent[] {
  const content: JSONContent[] = [];
  text.split(/\r\n|\r|\n/).forEach((line, index) => {
    if (index > 0) content.push({ type: "hardBreak" });
    if (line) content.push({ type: "text", text: line });
  });
  return content;
}

/** What a Question asked from the composer carries besides its text. */
export interface QuestionToAsk {
  /** Its Block ID, made here so it can be asked at once. */
  id: string;
  text: string;
  /** The Mind's model; null follows the default. */
  model: ChatModelChoice | null;
  scope: SearchScope;
  /** The Skill chosen for it with "/", if any. */
  skill: string | null;
}

/** The Question's stored attributes (`QuestionAttributes`). */
export function questionAttributes(question: QuestionToAsk): QuestionAttributes {
  return {
    id: question.id,
    providerId: question.model?.providerId ?? null,
    modelId: question.model?.modelId ?? null,
    ...scopeAttributesOf(question.scope),
    forcedSkill: question.skill,
  };
}

/** A Question put in the note: where, and whether it took an empty line's place (see `takeBackQuestion`). */
export interface PlacedQuestion {
  id: string;
  pos: number;
  replacedEmptyLine: boolean;
}

/**
 * Puts the Question into the Mind at `place` (see `questionPlace`), without
 * moving the focus there: the composer keeps it. The selection stays where it
 * was, mapped past the new line.
 */
export function placeQuestion(
  editor: Editor,
  question: QuestionToAsk,
  place: QuestionPlace,
): PlacedQuestion | null {
  const type = editor.schema.nodes[QUESTION_BLOCK];
  if (!type) return null;
  const node = editor.schema.nodeFromJSON({
    type: QUESTION_BLOCK,
    attrs: questionAttributes(question),
    content: questionContent(question.text),
  });
  const done = editor.commands.command(({ tr }) => {
    tr.replaceWith(place.from, place.to, node);
    return true;
  });
  if (!done) return null;
  return { id: question.id, pos: place.from, replacedEmptyLine: place.to > place.from };
}

/**
 * Takes a Question that couldn't be asked back out of the Mind (its text goes
 * back to the composer): the empty line it replaced comes back. Leaves it if
 * the User has changed it meanwhile, or it is gone.
 */
export function takeBackQuestion(editor: Editor, placed: PlacedQuestion, text: string): void {
  const found = findBlock(editor.state.doc, placed.id);
  if (!found || found.node.type.name !== QUESTION_BLOCK) return;
  if (found.node.textBetween(0, found.node.content.size, "\n", "\n") !== text) return;
  const from = found.pos;
  const to = from + found.node.nodeSize;
  editor.commands.command(({ tr, state }) => {
    const paragraph = state.schema.nodes.paragraph?.create();
    if (placed.replacedEmptyLine && paragraph) tr.replaceWith(from, to, paragraph);
    else tr.delete(from, to);
    return true;
  });
}

/**
 * Resolves once the Answer is in the editor's document (true), or after
 * `timeoutMs` if it never comes (false). The core writes it as it asks, but
 * its change can reach the editor a moment after the reply to the ask.
 */
export function whenAnswerShown(editor: Editor, answerId: string, timeoutMs = 3_000) {
  if (findBlock(editor.state.doc, answerId)) return Promise.resolve(true);
  return new Promise<boolean>((resolve) => {
    const check = () => {
      if (findBlock(editor.state.doc, answerId)) done(true);
    };
    const timer = setTimeout(() => done(false), timeoutMs);
    function done(found: boolean) {
      clearTimeout(timer);
      editor.off("update", check);
      resolve(found);
    }
    editor.on("update", check);
  });
}

/**
 * Puts the Mind's cursor on an empty line right after an Answer, adding the
 * line unless one is there already: the next Question asked goes there, and
 * Esc in the composer comes back to write there. The focus stays where it is.
 */
export function cursorBelowAnswer(editor: Editor, answerId: string): boolean {
  const answer = findBlock(editor.state.doc, answerId);
  if (!answer) return false;
  const after = answer.pos + answer.node.nodeSize;
  const next = editor.state.doc.nodeAt(after);
  return editor.commands.command(({ tr, state }) => {
    if (!isEmptyLine(next)) {
      const paragraph = state.schema.nodes.paragraph?.create();
      if (!paragraph) return false;
      tr.insert(after, paragraph);
    }
    tr.setSelection(TextSelection.create(tr.doc, after + 1));
    return true;
  });
}
