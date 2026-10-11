import type { Editor } from "@tiptap/core";
import { canInsertFootnote, insertFootnote } from "../editor/footnoteSchema";
import { noteHolding } from "../editor/noteKeymap";
import type { Command, CommandContext } from "./registry";

/** The note's keymap gives a command the Block at the cursor as its focus. */
function noteOf(ctx: CommandContext): Editor | null {
  const editor = ctx.focus?.kind === "block" ? noteHolding(ctx.focus.id) : null;
  return editor && !editor.isDestroyed && editor.isEditable ? editor : null;
}

/**
 * Insert footnote (#284), as in Word and Google Docs: ⌥⌘F (Ctrl+Alt+F) adds
 * a footnote after the selection and opens it to be written, or opens the
 * footnote selected. The "/" menu and the selection menu do the same.
 */
export const footnoteCommands: readonly Command[] = [
  {
    id: "footnote.insert",
    label: "editor.footnote.insert",
    shortcut: { keys: "CmdOrCtrl+Alt+F" },
    group: "main",
    appliesTo: ["block"],
    // The note's own undo (⌘Z) takes it back.
    reversible: true,
    when: (ctx) => noteOf(ctx) !== null,
    enabled(ctx) {
      const editor = noteOf(ctx);
      return editor && canInsertFootnote(editor.state)
        ? true
        : { reason: "editor.footnote.unavailable" };
    },
    run(ctx) {
      const editor = noteOf(ctx);
      if (editor) insertFootnote(editor);
    },
  },
];
