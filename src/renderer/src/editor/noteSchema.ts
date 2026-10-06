import type { Extensions, NodeViewRenderer } from "@tiptap/core";
import CodeBlockLowlight from "@tiptap/extension-code-block-lowlight";
import { BlockMath, InlineMath } from "@tiptap/extension-mathematics";
import UniqueID from "@tiptap/extension-unique-id";
import StarterKit from "@tiptap/starter-kit";
import { lowlight } from "./codeLanguages";

/** The attribute each Block keeps its UUID in. It is stored in the Mind's Yjs document. */
export const BLOCK_ID_ATTRIBUTE = "id";

/**
 * The node types a top-level Block can be. UniqueID gives every node of these
 * types a UUID as it is created, and keeps it when the Block is edited, turned
 * into another type or dragged. It can't tell top-level nodes from nested ones,
 * so paragraphs inside lists and quotes get an ID too; only top-level IDs name Blocks.
 */
export const BLOCK_TYPES: readonly string[] = [
  "paragraph",
  "heading",
  "codeBlock",
  "blockMath",
  "bulletList",
  "orderedList",
  "blockquote",
  "horizontalRule",
];

/** The node types that hold a formula, written in LaTeX. */
export const MATH_TYPES: readonly string[] = ["blockMath", "inlineMath"];

export interface NoteSchemaOptions {
  /** Called with a formula's position when it is clicked, to edit it. */
  onEditMath?: (pos: number) => void;
  /** Draws code blocks, e.g. with a language picker. */
  codeBlockView?: NodeViewRenderer;
}

/**
 * What a Note can hold, and how it is stored: headings, bold, italic,
 * strikethrough, lists, code blocks highlighted by lowlight, and inline and
 * block math rendered by KaTeX, each Block with a UUID. The editor adds its
 * interface on top (collaboration, menus, the drag handle); tests use this alone.
 */
export function noteExtensions({ onEditMath, codeBlockView }: NoteSchemaOptions = {}): Extensions {
  const onClick = onEditMath && ((_node: unknown, pos: number) => onEditMath(pos));
  const CodeBlock = codeBlockView
    ? CodeBlockLowlight.extend({ addNodeView: () => codeBlockView })
    : CodeBlockLowlight;

  return [
    StarterKit.configure({
      codeBlock: false,
      // Collaboration brings its own undo, which only undoes this client's changes.
      undoRedo: false,
      // It would add a paragraph to Minds that end in another Block just by opening them.
      trailingNode: false,
      // Shows where a dragged Block will land.
      dropcursor: { color: "rgb(14 165 233)", width: 2 },
    }),
    CodeBlock.configure({ lowlight }),
    BlockMath.configure({ katexOptions: { displayMode: true, throwOnError: false }, onClick }),
    InlineMath.configure({ katexOptions: { throwOnError: false }, onClick }),
    UniqueID.configure({ attributeName: BLOCK_ID_ATTRIBUTE, types: [...BLOCK_TYPES] }),
  ];
}
