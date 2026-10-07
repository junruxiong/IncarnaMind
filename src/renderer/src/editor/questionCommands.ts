import { type Editor, Extension, isMacOS } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";
import { BLOCK_ID_ATTRIBUTE, QUESTION_BLOCK } from "../../../core/api";
import { useAnswers } from "../answers";
import { findBlock } from "./blockCommands";

declare module "@tiptap/core" {
  interface Commands<ReturnType> {
    questions: {
      /**
       * Turns the paragraph or heading the cursor is in into a Question,
       * keeping its text; anywhere else, adds an empty Question after the Block.
       */
      setQuestion: () => ReturnType;
      /** Turns a Question back into a paragraph, or anything else into a Question (`setQuestion`). */
      toggleQuestion: () => ReturnType;
    };
  }
}

/** Turns the Block into a Question, or back. Shown in the slash menu and placeholders. */
export const QUESTION_SHORTCUT = "Mod-j";
export const QUESTION_SHORTCUT_LABEL = isMacOS() ? "⌘J" : "Ctrl+J";

const TURNS_INTO_QUESTION = new Set(["paragraph", "heading"]);

export interface QuestionCommandsOptions {
  /** Called with a Question's Block ID when Enter is pressed in it. */
  onAsk: (editor: Editor, questionId: string) => void;
}

/** Writing Questions: `setQuestion`, Mod-J to toggle one, and Enter to ask it (Shift+Enter breaks the line). */
export const QuestionCommands = Extension.create<QuestionCommandsOptions>({
  name: "questionCommands",
  // Before Enter splits the Block.
  priority: 1000,

  addOptions() {
    return { onAsk: () => undefined };
  },

  addCommands() {
    return {
      setQuestion:
        () =>
        ({ state, tr, dispatch, commands }) => {
          const { $from } = state.selection;
          if ($from.depth === 0) return false;
          const block = $from.node(1);
          if (block.type.name === QUESTION_BLOCK) return true;
          if ($from.depth === 1 && TURNS_INTO_QUESTION.has(block.type.name)) {
            return commands.setNode(QUESTION_BLOCK);
          }
          // In a code block, a list, an Answer…: a new Question after it.
          const type = state.schema.nodes[QUESTION_BLOCK];
          if (!type) return false;
          const after = $from.after(1);
          if (dispatch) {
            tr.insert(after, type.create());
            tr.setSelection(TextSelection.create(tr.doc, after + 1)).scrollIntoView();
          }
          return true;
        },

      toggleQuestion:
        () =>
        ({ state, commands }) => {
          const { $from } = state.selection;
          if ($from.depth === 1 && $from.parent.type.name === QUESTION_BLOCK) {
            return commands.setNode("paragraph");
          }
          return commands.setQuestion();
        },
    };
  },

  addKeyboardShortcuts() {
    return {
      [QUESTION_SHORTCUT]: () => this.editor.commands.toggleQuestion(),
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
 * Asks a Question of the Mind open in `editor`. Once its Answer is on its way,
 * the cursor moves below it; if it can't be asked, the Question shows why.
 */
export async function askInEditor(editor: Editor, mindId: string, questionId: string) {
  const result = await useAnswers.getState().ask(mindId, questionId);
  if (result?.asked) moveBelowAnswer(editor, result.answerId);
}

/** Puts the cursor in the Block after an Answer, to go on writing or ask the next Question. */
export function moveBelowAnswer(editor: Editor, answerId: string): void {
  if (editor.isDestroyed) return;
  const answer = findBlock(editor.state.doc, answerId);
  if (!answer) return;
  const after = answer.pos + answer.node.nodeSize;
  const next = editor.state.doc.nodeAt(after);
  if (next?.isTextblock) {
    editor
      .chain()
      .focus()
      .setTextSelection(after + 1)
      .scrollIntoView()
      .run();
  }
}
