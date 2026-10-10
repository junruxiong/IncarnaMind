import { type Editor, Extension, isMacOS } from "@tiptap/core";
import { BLOCK_ID_ATTRIBUTE, QUESTION_BLOCK } from "../../../core/api";
import { useAnswers } from "../answers";
import { cursorBelowAnswer, whenAnswerShown } from "./composerAsk";

/** Focuses the composer, wherever the focus is (see `Composer`). Shown in hints and the guide. */
export const QUESTION_SHORTCUT_LABEL = isMacOS() ? "⌘J" : "Ctrl+J";

export interface QuestionCommandsOptions {
  /** Called with a Question's Block ID when Enter is pressed in it. */
  onAsk: (editor: Editor, questionId: string) => void;
}

/**
 * A Question line in the note, once asked from the composer, can be edited
 * like any text: Enter asks it again (Shift+Enter breaks the line).
 */
export const QuestionCommands = Extension.create<QuestionCommandsOptions>({
  name: "questionCommands",
  // Before Enter splits the Block.
  priority: 1000,

  addOptions() {
    return { onAsk: () => undefined };
  },

  addKeyboardShortcuts() {
    return {
      Enter: ({ editor }) => {
        const { $from } = editor.state.selection;
        if ($from.parent.type.name !== QUESTION_BLOCK) return false;
        const id = $from.parent.attrs[BLOCK_ID_ATTRIBUTE];
        if (typeof id === "string" && $from.parent.textContent.trim() !== "") {
          this.options.onAsk(editor, id);
        }
        return true;
      },
    };
  },
});

/**
 * Asks a Question of the Mind open in `editor` again, from the note. Once its
 * Answer is on its way, the cursor moves below it; if it can't be asked, the
 * Question shows why.
 */
export async function askInEditor(editor: Editor, mindId: string, questionId: string) {
  const result = await useAnswers.getState().ask(mindId, questionId);
  if (!result?.asked || !(await whenAnswerShown(editor, result.answerId))) return;
  if (!editor.isDestroyed && cursorBelowAnswer(editor, result.answerId)) {
    // Never left in the Question, where the next words typed would change it.
    editor.chain().focus().scrollIntoView().run();
  }
}
