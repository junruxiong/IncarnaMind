import { checkSentences, useChecks } from "../checks";
import { selectionForEdit } from "../editor/editSelection";
import { noteHandle } from "./handles";
import type { Command } from "./registry";

/** The Mind last written in, while it has text selected to check and no check of it runs. */
function withSentences() {
  const note = noteHandle.current;
  if (!note || note.editor.isDestroyed || useChecks.getState().running[note.mindId]) return null;
  return selectionForEdit(note.editor.state) ? note : null;
}

/** Check against Documents (#96), on the selected sentences, from ⌘K; "find support" finds it too. */
export const checkCommands: readonly Command[] = [
  {
    id: "check.againstDocuments",
    label: "editor.ai.checkAgainstDocuments",
    keywords: ["editor.ai.checkAgainstDocuments.keywords"],
    group: "main",
    // It changes no words: its marks go with Check again, or when the sentence is edited.
    reversible: true,
    when: () => withSentences() !== null,
    run() {
      const note = withSentences();
      if (note) void checkSentences(note.editor, note.mindId, note.doc, "check-against-documents");
    },
  },
];
