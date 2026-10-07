import type { Extensions, NodeViewRenderer } from "@tiptap/core";
import CodeBlockLowlight from "@tiptap/extension-code-block-lowlight";
import Highlight from "@tiptap/extension-highlight";
import { BlockMath, InlineMath } from "@tiptap/extension-mathematics";
import UniqueID from "@tiptap/extension-unique-id";
import StarterKit from "@tiptap/starter-kit";
import {
  ANSWER_BLOCK,
  BLOCK_ID_ATTRIBUTE,
  NOTE_BLOCK_TYPES,
  QUESTION_BLOCK,
} from "../../../core/api";
import { Citation } from "./citationSchema";
import { lowlight } from "./codeLanguages";
import { Answer, MindDocument, Question, QuestionContextFlag } from "./questionSchema";

/** The attribute each Block keeps its UUID in. It is stored in the Mind's Yjs document. */
export { BLOCK_ID_ATTRIBUTE };

/**
 * The node types a top-level Block can be: a Note's, a Question or an Answer.
 * UniqueID gives every node of these types a UUID as it is created, and keeps
 * it when the Block is edited, turned into another type or dragged. It can't
 * tell top-level nodes from nested ones, so paragraphs inside lists, quotes
 * and Answers get an ID too; only top-level IDs name Blocks.
 */
export const BLOCK_TYPES: readonly string[] = [...NOTE_BLOCK_TYPES, QUESTION_BLOCK, ANSWER_BLOCK];

/** The node types that hold a formula, written in LaTeX. */
export const MATH_TYPES: readonly string[] = ["blockMath", "inlineMath"];

export interface NoteSchemaOptions {
  /** Called with a formula's position when it is clicked, to edit it. */
  onEditMath?: (pos: number) => void;
  /** Draws code blocks, e.g. with a language picker. */
  codeBlockView?: NodeViewRenderer;
  /** Draws Questions, e.g. with a model picker. */
  questionView?: NodeViewRenderer;
  /** Draws Answers, e.g. with their status and a stop button. */
  answerView?: NodeViewRenderer;
  /** Draws Citations, e.g. as badges that open the cited page. */
  citationView?: NodeViewRenderer;
}

/**
 * What a Mind can hold, and how it is stored. Notes: headings, bold, italic,
 * strikethrough, highlight, lists, code blocks highlighted by lowlight, and
 * inline and block math rendered by KaTeX, each with an "include in Question
 * context" flag. Questions and Answers, top-level only. Citations, inline in
 * Answers and in whatever they are copied into. Every Block has a UUID. The
 * editor adds its interface on top (collaboration, menus, the drag handle,
 * smart typography); tests use this alone.
 */
export function noteExtensions({
  onEditMath,
  codeBlockView,
  questionView,
  answerView,
  citationView,
}: NoteSchemaOptions = {}): Extensions {
  const onClick = onEditMath && ((_node: unknown, pos: number) => onEditMath(pos));
  const CodeBlock = codeBlockView
    ? CodeBlockLowlight.extend({ addNodeView: () => codeBlockView })
    : CodeBlockLowlight;

  return [
    StarterKit.configure({
      // MindDocument takes Questions and Answers as well as Notes.
      document: false,
      codeBlock: false,
      // Collaboration brings its own undo, which only undoes this client's changes.
      undoRedo: false,
      // It would add a paragraph to Minds that end in another Block just by opening them.
      trailingNode: false,
      // Shows where a dragged Block will land.
      dropcursor: { color: "rgb(14 165 233)", width: 2 },
    }),
    MindDocument,
    // One colour: a `highlight` mark with no attributes. Mod-Shift-H, or `==text==` typed or pasted.
    Highlight,
    CodeBlock.configure({ lowlight }),
    BlockMath.configure({ katexOptions: { displayMode: true, throwOnError: false }, onClick }),
    InlineMath.configure({ katexOptions: { throwOnError: false }, onClick }),
    Question.configure({ view: questionView ?? null }),
    Answer.configure({ view: answerView ?? null }),
    Citation.configure({ view: citationView ?? null }),
    QuestionContextFlag,
    UniqueID.configure({ attributeName: BLOCK_ID_ATTRIBUTE, types: [...BLOCK_TYPES] }),
  ];
}
