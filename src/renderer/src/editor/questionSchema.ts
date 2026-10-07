import { Extension, mergeAttributes, Node, type NodeViewRenderer } from "@tiptap/core";
import {
  ANSWER_BLOCK,
  type AnswerAttributes,
  BLOCK_ID_ATTRIBUTE,
  INCLUDE_IN_CONTEXT_ATTRIBUTE,
  NOTE_BLOCK_TYPES,
  QUESTION_BLOCK,
  type QuestionAttributes,
} from "../../../core/api";

/**
 * Questions and Answers are top-level Blocks only: they belong to this group,
 * which only the Mind itself takes, never a list, a quote or an Answer.
 */
const MIND_BLOCK_GROUP = "mindBlock";

/** A Mind: Notes, Questions and Answers, in the order the User placed them. Replaces StarterKit's document. */
export const MindDocument = Node.create({
  name: "doc",
  topNode: true,
  content: `(block | ${MIND_BLOCK_GROUP})+`,
});

/** Attributes stored in the Mind's Yjs document but not drawn in the HTML. */
const stored = (defaultValue: unknown = null) => ({ default: defaultValue, rendered: false });

export interface QuestionOptions {
  /** Draws the Question, e.g. with its model picker. */
  view: NodeViewRenderer | null;
  /** Called when the User presses Enter in a Question, with its Block ID. */
  onAsk: (questionId: string) => void;
}

/** A Question: the User's text, asked with Enter. Shift+Enter breaks the line. */
export const Question = Node.create<QuestionOptions>({
  name: QUESTION_BLOCK,
  group: MIND_BLOCK_GROUP,
  content: "inline*",
  defining: true,

  addOptions() {
    return { view: null, onAsk: () => undefined };
  },

  addAttributes() {
    const attributes: Record<Exclude<keyof QuestionAttributes, "id">, ReturnType<typeof stored>> = {
      providerId: stored(),
      modelId: stored(),
    };
    return attributes;
  },

  parseHTML() {
    return [{ tag: `div[data-type="${QUESTION_BLOCK}"]` }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["div", mergeAttributes(HTMLAttributes, { "data-type": QUESTION_BLOCK }), 0];
  },

  addNodeView() {
    return this.options.view;
  },

  addKeyboardShortcuts() {
    return {
      Enter: ({ editor }) => {
        const { $from } = editor.state.selection;
        if ($from.parent.type.name !== QUESTION_BLOCK) return false;
        const id = $from.parent.attrs[BLOCK_ID_ATTRIBUTE];
        if (typeof id === "string" && $from.parent.textContent.trim() !== "") {
          this.options.onAsk(id);
        }
        return true;
      },
    };
  },
});

export interface AnswerOptions {
  /** Draws the Answer: its status, stop and regenerate, and errors. */
  view: NodeViewRenderer | null;
}

/**
 * An Answer, written by the core (`askQuestion`) right after its Question.
 * Its content is ordinary rich text, which the User can edit.
 */
export const Answer = Node.create<AnswerOptions>({
  name: ANSWER_BLOCK,
  group: MIND_BLOCK_GROUP,
  content: "block+",
  defining: true,
  // Typing and deleting stay inside it: its text never merges with the Blocks around it.
  isolating: true,

  addOptions() {
    return { view: null };
  },

  addAttributes() {
    const attributes: Record<Exclude<keyof AnswerAttributes, "id">, object> = {
      questionId: stored(),
      providerId: stored(),
      modelId: stored(),
      status: {
        default: "done",
        parseHTML: (element: HTMLElement) => element.getAttribute("data-status") ?? "done",
        renderHTML: (attributes: Record<string, unknown>) => ({ "data-status": attributes.status }),
      },
      errorKind: stored(),
      errorMessage: stored(),
      generatedHash: stored(),
    };
    return attributes;
  },

  parseHTML() {
    return [{ tag: `div[data-type="${ANSWER_BLOCK}"]` }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["div", mergeAttributes(HTMLAttributes, { "data-type": ANSWER_BLOCK }), 0];
  },

  addNodeView() {
    return this.options.view;
  },
});

/**
 * Each Note's "include in Question context" flag: on by default; switched off,
 * the Note is left out of what Questions see and is drawn muted.
 */
export const QuestionContextFlag = Extension.create({
  name: "questionContextFlag",

  addGlobalAttributes() {
    return [
      {
        types: [...NOTE_BLOCK_TYPES],
        attributes: {
          [INCLUDE_IN_CONTEXT_ATTRIBUTE]: {
            default: null,
            parseHTML: (element: HTMLElement) =>
              element.getAttribute("data-context") === "off" ? false : null,
            renderHTML: (attributes: Record<string, unknown>) =>
              attributes[INCLUDE_IN_CONTEXT_ATTRIBUTE] === false ? { "data-context": "off" } : {},
          },
        },
      },
    ];
  },
});
