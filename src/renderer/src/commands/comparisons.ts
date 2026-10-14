import { startCompareDocuments } from "../comparisons";
import { currentMindIdOf } from "../layoutSlice";
import { useAppStore } from "../store";
import { noteHandle } from "./handles";
import type { Command } from "./registry";

/** The top-level Block the cursor is in, in the Mind's editor the User last wrote in. */
function cursorBlockIn(mindId: string | null): string | null {
  const note = noteHandle.current;
  if (!mindId || note?.mindId !== mindId || note.editor.isDestroyed) return null;
  const { $from } = note.editor.state.selection;
  const id = $from.depth > 0 ? $from.node(1).attrs.id : null;
  return typeof id === "string" ? id : null;
}

export const comparisonCommands: readonly Command[] = [
  {
    // Compare Documents (#98): the sheet at the foot of the Mind in front, or of a new Mind.
    id: "comparison.start",
    label: "command.compareDocuments",
    keywords: ["command.compareDocuments.keywords"],
    group: "main",
    // Opening the sheet changes nothing; the table it makes is a change to accept.
    reversible: true,
    run: () => {
      const mindId = currentMindIdOf(useAppStore.getState().layout);
      return startCompareDocuments(mindId, cursorBlockIn(mindId));
    },
  },
];
