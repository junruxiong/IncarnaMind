import type { Editor } from "@tiptap/core";
import type { EditAction } from "../../../core/api";
import { translate } from "../../../shared/i18n";
import { useComposer } from "../composer";
import { selectionForEdit } from "../editor/editSelection";
import { noteHolding } from "../editor/noteKeymap";
import { pendingAtSelection, pendingIn } from "../editor/proposalReview";
import {
  regenerateChange,
  requestEdit,
  resolveProposalsIn,
  startEditing,
  useEdits,
} from "../editWithAi";
import { useAppStore } from "../store";
import { noteHandle } from "./handles";
import type { Command, CommandContext } from "./registry";

/*
 * Proposals (#93) and Edit with AI, from the keyboard and ⌘K. ⌘↵ and ⌘⌫ act
 * in the note, on the change the cursor is in or the changes a selection
 * touches (a section, as in Word); anywhere else they do what they always do.
 */

/** The note the cursor's Block is in, while the cursor or the selection is on a Proposal. */
function noteOnProposal(ctx: CommandContext): Editor | null {
  const editor = ctx.focus?.kind === "block" ? noteHolding(ctx.focus.id) : null;
  return editor && !editor.isDestroyed && pendingAtSelection(editor.state).length > 0
    ? editor
    : null;
}

function resolveAt(ctx: CommandContext, outcome: "accept" | "reject"): void {
  const editor = noteOnProposal(ctx);
  const note = noteHandle.current;
  if (!editor || note?.editor !== editor) return;
  const ids = pendingAtSelection(editor.state).map((each) => each.id);
  resolveProposalsIn(editor, note.mindId, ids, outcome);
}

/** The change out of date the cursor is on, in the Mind last written in, while no request runs there. */
function outOfDateAt(ctx: CommandContext) {
  const editor = noteOnProposal(ctx);
  const note = noteHandle.current;
  if (!editor || note?.editor !== editor || useEdits.getState().pending[note.mindId]) return null;
  const proposal = pendingAtSelection(editor.state).find((each) => each.outOfDate);
  return proposal ? { note, proposal } : null;
}

/** The Mind last written in, while it has Proposals waiting. */
function withProposals() {
  const note = noteHandle.current;
  return note && !note.editor.isDestroyed && pendingIn(note.editor.state).length > 0 ? note : null;
}

function resolveAll(outcome: "accept" | "reject"): void {
  const note = withProposals();
  if (!note) return;
  resolveProposalsIn(note.editor, note.mindId, "all", outcome);
}

/** The Mind last written in, while it has text selected that Edit with AI can change. */
function withSelection() {
  const note = noteHandle.current;
  if (!note || note.editor.isDestroyed || useEdits.getState().pending[note.mindId]) return null;
  return selectionForEdit(note.editor.state) ? note : null;
}

/** An AI action of the selection menu, on the selected text. */
const editAction = (
  action: Extract<EditAction, "improve" | "shorten" | "explain">,
  label: "editor.ai.improve" | "editor.ai.shorten" | "editor.ai.explain",
): Command => ({
  id: `edit.${action}`,
  label,
  group: "main",
  // Its changes wait as Proposals: rejecting them leaves the text as it was.
  reversible: true,
  when: () => withSelection() !== null,
  run() {
    const note = withSelection();
    if (note) void requestEdit(note.editor, note.mindId, note.doc, action);
  },
});

export const proposalCommands: readonly Command[] = [
  {
    id: "proposal.accept",
    label: "command.proposal.accept",
    shortcut: { keys: "CmdOrCtrl+Enter" },
    group: "main",
    appliesTo: ["block"],
    // ⌘Z, or Undo on its line, takes it back.
    reversible: true,
    when: (ctx) => noteOnProposal(ctx) !== null,
    run: (ctx) => resolveAt(ctx, "accept"),
  },
  {
    id: "proposal.reject",
    label: "command.proposal.reject",
    shortcut: { keys: "CmdOrCtrl+Backspace", inText: true },
    group: "main",
    appliesTo: ["block"],
    reversible: true,
    when: (ctx) => noteOnProposal(ctx) !== null,
    run: (ctx) => resolveAt(ctx, "reject"),
  },
  {
    // "Regenerate" under a change out of date, from the keyboard and ⌘K.
    id: "proposal.regenerate",
    label: "command.proposal.regenerate",
    group: "main",
    appliesTo: ["block"],
    // Its new changes wait as Proposals, and rejecting them leaves the text as it was.
    reversible: true,
    when: (ctx) => outOfDateAt(ctx) !== null,
    run(ctx) {
      const found = outOfDateAt(ctx);
      if (found) {
        const { editor, mindId, doc } = found.note;
        void regenerateChange(editor, mindId, doc, found.proposal);
      }
    },
  },
  {
    id: "proposal.acceptAll",
    label: "command.proposal.acceptAll",
    group: "main",
    reversible: true,
    when: () => withProposals() !== null,
    run: () => resolveAll("accept"),
  },
  {
    id: "proposal.rejectAll",
    label: "command.proposal.rejectAll",
    group: "main",
    reversible: true,
    when: () => withProposals() !== null,
    run: () => resolveAll("reject"),
  },
  {
    // ⌘J with text selected does it too (see "composer.focus").
    id: "edit.withAi",
    label: "editor.ai.edit",
    group: "main",
    // Nothing changes until the User says what, and then only as Proposals.
    reversible: true,
    when: () => withSelection() !== null,
    run() {
      const note = withSelection();
      if (note && startEditing(note.editor, note.mindId)) useComposer.getState().requestFocus();
    },
  },
  editAction("improve", "editor.ai.improve"),
  editAction("shorten", "editor.ai.shorten"),
  {
    // From ⌘K the language is said in the composer: "Translate into …".
    id: "edit.translate",
    label: "editor.ai.translate",
    group: "main",
    reversible: true,
    when: () => withSelection() !== null,
    run() {
      const note = withSelection();
      if (!note || !startEditing(note.editor, note.mindId)) return;
      const language = useAppStore.getState().settings?.language ?? "en";
      useComposer
        .getState()
        .update(note.mindId, { text: translate(language, "composer.edit.translateInto") });
      useComposer.getState().requestFocus();
    },
  },
  editAction("explain", "editor.ai.explain"),
];
