import { type Editor, Extension } from "@tiptap/core";
import { Plugin, PluginKey, type Transaction } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { ANSWER_BLOCK, BLOCK_ID_ATTRIBUTE, QUESTION_BLOCK } from "../../../core/api";

type Folded = ReadonlySet<string>;

const questionFoldKey = new PluginKey<Folded>("questionFold");

/**
 * The Questions folded in each Mind (by Mind id): how it looks in this window,
 * not part of the Mind. Kept while the window is open, so switching tabs back
 * finds them as they were.
 */
const foldedByMind = new Map<string, Folded>();

/**
 * Folding a Question hides its Answer (DESIGN.md, Question). The Question gets
 * `data-folded` (and a decoration whose spec says so, see `isFolded`), its
 * Answer the class `answer-folded`; nothing is stored in the Mind.
 */
export function questionFoldPlugin(mindId: string): Plugin<Folded> {
  return new Plugin<Folded>({
    key: questionFoldKey,
    state: {
      init: () => foldedByMind.get(mindId) ?? new Set(),
      apply(tr, folded) {
        const toggled: unknown = tr.getMeta(questionFoldKey);
        if (typeof toggled !== "string") return folded;
        const next = new Set(folded);
        if (next.has(toggled)) next.delete(toggled);
        else next.add(toggled);
        foldedByMind.set(mindId, next);
        return next;
      },
    },
    props: {
      decorations(state) {
        const folded = questionFoldKey.getState(state);
        // Nothing folded, as almost always: no work at all while typing.
        if (!folded || folded.size === 0) return null;
        const decorations: Decoration[] = [];
        state.doc.forEach((node, pos) => {
          const end = pos + node.nodeSize;
          if (node.type.name === QUESTION_BLOCK && folded.has(node.attrs[BLOCK_ID_ATTRIBUTE])) {
            decorations.push(Decoration.node(pos, end, { "data-folded": "true" }, { folded }));
          } else if (node.type.name === ANSWER_BLOCK && folded.has(node.attrs.questionId)) {
            decorations.push(
              Decoration.node(pos, end, { class: "answer-folded" }, { hidden: true }),
            );
          }
        });
        return DecorationSet.create(state.doc, decorations);
      },
    },
  });
}

export interface QuestionFoldOptions {
  mindId: string;
}

/** Folding Questions in the Mind's editor: a Question's fold chevron toggles it (`toggleFold`). */
export const QuestionFold = Extension.create<QuestionFoldOptions>({
  name: "questionFold",

  addOptions() {
    return { mindId: "" };
  },

  addProseMirrorPlugins() {
    return [questionFoldPlugin(this.options.mindId)];
  },
});

/** Marks a transaction as folding the Question's Answer away, or showing it again. */
export const foldToggled = (tr: Transaction, questionId: string): Transaction =>
  tr.setMeta(questionFoldKey, questionId);

/** Folds the Question's Answer away, or shows it again. */
export function toggleFold(editor: Editor, questionId: string): void {
  editor.commands.command(({ tr }) => {
    foldToggled(tr, questionId);
    return true;
  });
}

/** Whether the Question is folded, from the decorations its node view is given. */
export const isFolded = (decorations: readonly { spec: unknown }[]): boolean =>
  decorations.some((decoration) => {
    const spec = decoration.spec as { folded?: unknown } | null;
    return spec?.folded !== undefined;
  });
