/**
 * Add to note (#201): an Answer in the assistant, or the part of it selected,
 * goes into a Mind's note as Notes, Citations and all, its words AI text, each
 * Citation checked again where it lands, with a link back to the Answer and
 * its conversation (src/shared/blockOrigins.ts).
 *
 * - Into the Mind the assistant reads, open in its editor: at its cursor, or
 *   at its end, through the editor, as one step of its Undo, so ⌘Z in the
 *   note takes it back too.
 * - Into another Mind: at its end, through the core. With no Mind in focus
 *   (`openIt`), it is shown too, in a new tab beside the focused one if it isn't open,
 *   never replacing it.
 * - Into a new Mind: made from the Answer, named as the conversation, in its
 *   Folder, opened in a new tab beside the focused one, the Answer waiting
 *   there as changes to accept or reject.
 *
 * Each says what it did under the Answer, with Undo (and ⌘Z), and what went
 * in at once is recorded as AI use (#292); a change, once it is accepted.
 */
import type { Editor } from "@tiptap/core";
import { Fragment, type Node as ProseMirrorNode } from "@tiptap/pm/model";
import { Selection } from "@tiptap/pm/state";
import {
  BLOCK_ID_ATTRIBUTE,
  type BlockOrigin,
  type Conversation,
  type ConversationAnswer,
} from "../../core/api";
import { messageReference } from "../../core/references";
import { asGenerated } from "../../shared/aiText";
import type { NodeJSON } from "../../shared/blockJson";
import { setOrigins } from "../../shared/blockOrigins";
import { mindName } from "../../shared/naming";
import { citedTables, recheckCitations, textLength, withoutIds } from "../../shared/noteBlocks";
import { readMindHandle } from "./commands/handles";
import { conversationLabel, questionOf } from "./conversations";
import { core } from "./core";
import { flashBlocks } from "./editor/blockOrigins";
import { asOneStep, offerEditorUndo } from "./editor/editorUndo";
import { notePlace } from "./editor/notePlace";
import { translateNow as t } from "./i18n";
import { useAppStore } from "./store";
import { answerUndoPlace, useUndo } from "./undo";

/** Where Add to note puts an Answer: a Mind, by id, or a new Mind made from it. */
export type AddTarget = { kind: "mind"; mindId: string } | { kind: "new" };

/** What is added, and from where. */
export interface AddedAnswer {
  answer: ConversationAnswer;
  conversation: Conversation | undefined;
  /** The Notes to add: the part of the Answer selected, or all of it. */
  blocks: NodeJSON[];
}

const nameOfMind = (mindId: string) => {
  const mind = useAppStore.getState().minds.find((each) => each.id === mindId);
  return (mind && mindName(mind)) || t("mind.untitled");
};

const originOf = ({ answer }: AddedAnswer): BlockOrigin => ({
  from: messageReference(answer.conversationId, answer.id),
  question: questionOf(answer)?.text ?? "",
});

/**
 * Puts checked Blocks into the open Mind where its cursor is (or at its end),
 * as one step of the note's Undo. Returns their new ids, or null when the
 * editor couldn't take them.
 */
function addInEditor(editor: Editor, blocks: readonly NodeJSON[], atCursor: boolean) {
  const place = notePlace(editor.state.doc, atCursor ? editor.state.selection : null);
  const ids = blocks.map(() => crypto.randomUUID());
  let nodes: ProseMirrorNode[];
  try {
    nodes = blocks.map((block, index) =>
      editor.schema.nodeFromJSON({
        ...block,
        attrs: { ...block.attrs, [BLOCK_ID_ATTRIBUTE]: ids[index] },
      }),
    );
  } catch (error) {
    // Blocks this editor can't hold: nothing goes in, and the User is told.
    useAppStore.getState().reportError(error);
    return null;
  }
  const done = asOneStep(editor, () =>
    editor.commands.command(({ tr }) => {
      const fragment = Fragment.from(nodes);
      tr.replaceWith(place.from, place.to, fragment);
      // The cursor goes after what came in, so the next Answer added goes after it, in order.
      const end = Math.min(place.from + fragment.size, tr.doc.content.size);
      tr.setSelection(Selection.near(tr.doc.resolve(end), -1));
      return true;
    }),
  );
  if (!done) return null;
  const first = editor.view.nodeDOM(place.from);
  if (first instanceof HTMLElement) first.scrollIntoView({ block: "nearest" });
  return ids;
}

/** Into the Mind the assistant reads, through its editor; false when it isn't open in one. */
async function addToReadMind(added: AddedAnswer, mindId: string): Promise<boolean> {
  const read = readMindHandle.current;
  if (!read || read.mindId !== mindId || read.editor.isDestroyed) return false;
  const checked = asGenerated(
    await recheckCitations(citedTables(withoutIds(added.blocks)), (input) =>
      core.recheckCitation(input),
    ),
  );
  const { editor, doc } = read;
  if (editor.isDestroyed) return false;
  const ids = addInEditor(editor, checked, read.cursorInMind());
  if (!ids) return true;
  setOrigins(doc, ids, originOf(added));
  flashBlocks(editor, ids);
  const { model } = added.answer;
  // After the Mind's own update, so the record finds the Blocks there.
  core
    .recordAiUse([
      {
        mindId,
        kind: "answer-added",
        providerId: model?.providerId ?? null,
        modelId: model?.modelId ?? null,
        characters: textLength(checked),
        blockIds: ids,
      },
    ])
    .catch(useAppStore.getState().reportError);
  offerEditorUndo(
    editor,
    t("answer.added", { mind: nameOfMind(mindId) }),
    answerUndoPlace(added.answer.id),
  );
  return true;
}

/** Into a Mind that isn't the one the assistant reads, at its end, through the core. */
async function addToOtherMind(added: AddedAnswer, mindId: string, openIt: boolean): Promise<void> {
  const { blockIds } = await core.addToNote({
    mindId,
    blocks: added.blocks,
    origin: originOf(added),
    model: added.answer.model,
  });
  useUndo.getState().push({
    message: t("answer.added", { mind: nameOfMind(mindId) }),
    undo: () => core.undoAddToNote({ mindId, blockIds }),
    shownAt: answerUndoPlace(added.answer.id),
  });
  if (openIt) await openBeside(mindId);
}

/** Shows a Mind where it is open, else in a new tab beside the focused one. */
const openBeside = (mindId: string) =>
  useAppStore.getState().changeLayout({
    type: "open",
    item: { kind: "mind", mindId },
    where: { in: "place" },
  });

/**
 * Into a new Mind: named as the conversation, in its Folder, the Answer in it
 * as changes to accept or reject, in a new tab beside the focused one. Undo
 * deletes it.
 */
async function addToNewMind(added: AddedAnswer): Promise<void> {
  const title = conversationLabel(added.conversation);
  const mind = await core.createMind({
    title,
    folderId: added.conversation?.folderId ?? null,
  });
  await core.addToNote({
    mindId: mind.id,
    blocks: added.blocks,
    origin: originOf(added),
    model: added.answer.model,
    proposed: true,
  });
  await openBeside(mind.id);
  useUndo.getState().push({
    message: t("answer.addedToNew", { mind: mindName(mind) || title }),
    undo: () => core.deleteMind(mind.id),
    shownAt: answerUndoPlace(added.answer.id),
  });
}

/**
 * Adds an Answer to a Mind or a new Mind (see the module's comment). `openIt`:
 * no Mind is in focus, so a Mind chosen from the list is opened too.
 */
export async function addAnswer(
  added: AddedAnswer,
  target: AddTarget,
  openIt = false,
): Promise<void> {
  if (added.blocks.length === 0) return;
  try {
    if (target.kind === "new") await addToNewMind(added);
    else if (!(await addToReadMind(added, target.mindId))) {
      await addToOtherMind(added, target.mindId, openIt);
    }
  } catch (error) {
    useAppStore.getState().reportError(error);
  }
}
