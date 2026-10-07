import { Extension, mergeAttributes, Node, type NodeViewRenderer } from "@tiptap/core";
import { Fragment, type NodeType, type Node as ProseMirrorNode, Slice } from "@tiptap/pm/model";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import {
  ANSWER_BLOCK,
  type AnswerAttributes,
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
}

/** A Question: the User's text. The editor asks it with Enter (see `QuestionCommands`). */
export const Question = Node.create<QuestionOptions>({
  name: QUESTION_BLOCK,
  group: MIND_BLOCK_GROUP,
  content: "inline*",
  defining: true,

  addOptions() {
    return { view: null };
  },

  addAttributes() {
    const attributes: Record<Exclude<keyof QuestionAttributes, "id">, object> = {
      providerId: stored(),
      modelId: stored(),
      forcedSkill: stored(),
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
      citationSupport: stored(),
      toolCalls: stored(),
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

  addProseMirrorPlugins() {
    const type = this.type;
    return [
      new Plugin({
        key: new PluginKey("answerPaste"),
        props: {
          // Text copied from inside an Answer pastes as Notes, Citations and all.
          transformPasted: (slice) => unwrapOpenAnswers(slice, type),
        },
      }),
    ];
  },
});

/**
 * A slice copied from inside an Answer carries the Answer around it, which
 * ProseMirror would recreate when the slice fills an empty line: pasting part
 * of an Answer into a Note would make another Answer. Answers the slice is
 * open into are unwrapped to their content; whole Answers (e.g. one dragged by
 * its handle) are left alone.
 */
export function unwrapOpenAnswers(slice: Slice, answer: NodeType): Slice {
  const { content, openStart, openEnd } = slice;
  const last = content.childCount - 1;
  const openFirst = content.firstChild?.type === answer && openStart > 0;
  const openLast = content.lastChild?.type === answer && openEnd > 0;
  if (!openFirst && !openLast) return slice;
  const nodes: ProseMirrorNode[] = [];
  content.forEach((node, _offset, index) => {
    const unwrap = (index === 0 && openFirst) || (index === last && openLast);
    if (unwrap) {
      node.content.forEach((child) => {
        nodes.push(child);
      });
    } else nodes.push(node);
  });
  return new Slice(
    Fragment.fromArray(nodes),
    openFirst ? openStart - 1 : openStart,
    openLast ? openEnd - 1 : openEnd,
  );
}

/** Whether a top-level Block is a Note the User switched out of Question context. */
export const isSwitchedOff = (node: ProseMirrorNode): boolean =>
  node.attrs[INCLUDE_IN_CONTEXT_ATTRIBUTE] === false &&
  (NOTE_BLOCK_TYPES as readonly string[]).includes(node.type.name);

/**
 * Each Note's "include in Question context" flag: on by default; switched off,
 * the Note is left out of what Questions see and is drawn muted (the
 * "context-off" class, which reaches Notes drawn by their own views too).
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

  addProseMirrorPlugins() {
    return [
      new Plugin({
        key: new PluginKey("questionContextFlag"),
        props: {
          decorations(state) {
            const decorations: Decoration[] = [];
            state.doc.forEach((node, pos) => {
              if (isSwitchedOff(node)) {
                decorations.push(
                  Decoration.node(pos, pos + node.nodeSize, { class: "context-off" }),
                );
              }
            });
            return decorations.length > 0 ? DecorationSet.create(state.doc, decorations) : null;
          },
        },
      }),
    ];
  },
});
