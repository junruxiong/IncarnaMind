import type { ClaimCheckAction } from "../../../core/api";
import { documentsAdded, useAddedDocuments } from "../addedDocuments";
import { checkSentences, useChecks } from "../checks";
import { selectionForEdit } from "../editor/editSelection";
import { startMindCheck } from "../mindCheck";
import { noteHandle } from "./handles";
import type { Command } from "./registry";

/** The Mind last written in, while it has text selected to check and no check of it runs. */
function withSentences() {
  const note = noteHandle.current;
  if (!note || note.editor.isDestroyed || useChecks.getState().running[note.mindId]) return null;
  return selectionForEdit(note.editor.state) ? note : null;
}

/** Find support and Check against Documents (#96), on the selected sentences, from ⌘K. */
const checkCommand = (
  action: ClaimCheckAction,
  label: "editor.ai.findSupport" | "editor.ai.checkAgainstDocuments",
): Command => ({
  id: action === "find-support" ? "check.findSupport" : "check.againstDocuments",
  label,
  group: "main",
  // It changes no words: its marks go with Check again, or when the sentence is edited.
  reversible: true,
  when: () => withSentences() !== null,
  run() {
    const note = withSentences();
    if (note) void checkSentences(note.editor, note.mindId, note.doc, action);
  },
});

/** The Mind last written in, while Documents added since its checks could change some (#993). */
function withAdded(): string | null {
  const note = noteHandle.current;
  if (!note || note.editor.isDestroyed || useChecks.getState().running[note.mindId]) return null;
  const added = useAddedDocuments.getState().byMind[note.mindId];
  return documentsAdded(added).length > 0 ? note.mindId : null;
}

export const checkCommands: readonly Command[] = [
  checkCommand("find-support", "editor.ai.findSupport"),
  checkCommand("check-against-documents", "editor.ai.checkAgainstDocuments"),
  {
    id: "check.addedDocuments",
    label: "checks.added.command",
    group: "main",
    reversible: true,
    when: () => withAdded() !== null,
    run() {
      const mindId = withAdded();
      if (mindId) void startMindCheck(mindId, "added");
    },
  },
];
