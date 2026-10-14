import { Extension, Mark, mergeAttributes, Node, type NodeViewRenderer } from "@tiptap/core";
import {
  IMAGE_NODE,
  WORD_ATOM_NODE,
  WORD_BLOCK_NODE,
  WORD_LOOK_ATTRIBUTE,
  WORD_RUN_MARK,
} from "../../../core/api";
import { readWordAtom, wordAtomText } from "../../../shared/wordFiles";

/**
 * An attribute kept in the HTML as `data-word-<name>`, so an atom copied and
 * pasted (into another paragraph, or another Mind) keeps where its XML is.
 */
function dataAttribute(name: string, defaultValue: string | null) {
  const key = `data-word-${name}`;
  return {
    default: defaultValue,
    parseHTML: (element: HTMLElement) => element.getAttribute(key) ?? defaultValue,
    renderHTML: (attributes: Record<string, unknown>) => {
      const value = attributes[name];
      return typeof value === "string" ? { [key]: value } : {};
    },
  };
}

const atomAttributes = () => ({
  kind: dataAttribute("kind", "other"),
  label: dataAttribute("label", ""),
  text: dataAttribute("text", ""),
  source: dataAttribute("source", null),
  info: {
    default: null,
    parseHTML: (element: HTMLElement) => {
      try {
        const value: unknown = JSON.parse(element.getAttribute("data-word-info") ?? "null");
        return typeof value === "object" ? value : null;
      } catch {
        return null;
      }
    },
    renderHTML: (attributes: Record<string, unknown>) =>
      attributes.info ? { "data-word-info": JSON.stringify(attributes.info) } : {},
  },
});

export interface WordAtomOptions {
  /** Draws the atom: its chip, mark or box, and what it says on hover. */
  view: NodeViewRenderer | null;
}

/**
 * What a Word import kept as it is, inside a paragraph's text (#104): a
 * field, a content control, a text box, a footnote's mark, another person's
 * tracked change… (see `WordAtomAttributes`). It can't be edited inside; it
 * moves with the text and goes when that text is deleted.
 */
export const WordAtom = Node.create<WordAtomOptions>({
  name: WORD_ATOM_NODE,
  group: "inline",
  inline: true,
  atom: true,
  selectable: true,
  draggable: false,

  addOptions() {
    return { view: null };
  },

  addAttributes() {
    return atomAttributes();
  },

  parseHTML() {
    return [{ tag: `span[data-type="${WORD_ATOM_NODE}"]` }];
  },

  renderHTML({ node, HTMLAttributes }) {
    return [
      "span",
      mergeAttributes(HTMLAttributes, { "data-type": WORD_ATOM_NODE }),
      wordAtomText(readWordAtom(node.attrs)),
    ];
  },

  renderText({ node }) {
    return wordAtomText(readWordAtom(node.attrs));
  },

  addNodeView() {
    return this.options.view;
  },
});

export interface WordBlockOptions {
  view: NodeViewRenderer | null;
}

/**
 * What a Word import kept as it is as a Block of its own (#104): a block
 * content control (a table of contents, a cover page), a nested table, a
 * paragraph inside a field over several, the header and the footer. Its
 * text is read for Questions; it can't be edited inside.
 */
export const WordBlock = Node.create<WordBlockOptions>({
  name: WORD_BLOCK_NODE,
  group: "block",
  atom: true,
  selectable: true,
  draggable: true,

  addOptions() {
    return { view: null };
  },

  addAttributes() {
    return atomAttributes();
  },

  parseHTML() {
    return [{ tag: `div[data-type="${WORD_BLOCK_NODE}"]` }];
  },

  renderHTML({ node, HTMLAttributes }) {
    return [
      "div",
      mergeAttributes(HTMLAttributes, { "data-type": WORD_BLOCK_NODE }),
      wordAtomText(readWordAtom(node.attrs)),
    ];
  },

  renderText({ node }) {
    return wordAtomText(readWordAtom(node.attrs));
  },

  addNodeView() {
    return this.options.view;
  },
});

/**
 * A picture in the text, as Tiptap names it (`IMAGE_NODE`): an imported Word
 * file's inline pictures (#104), kept as `data:` URLs (the app's
 * Content-Security-Policy allows no other kind), with the size Word drew them
 * at and where the original is in the package. It can be moved and deleted.
 */
export const InlineImage = Node.create({
  name: IMAGE_NODE,
  group: "inline",
  inline: true,
  atom: true,
  selectable: true,
  draggable: true,

  addAttributes() {
    const number = (name: string) => ({
      default: null,
      parseHTML: (element: HTMLElement) => {
        const value = Number(element.getAttribute(name));
        return Number.isFinite(value) && value > 0 ? value : null;
      },
    });
    return {
      src: { default: null },
      alt: { default: null },
      width: number("width"),
      height: number("height"),
      source: dataAttribute("source", null),
    };
  },

  parseHTML() {
    return [{ tag: 'img[src^="data:image/"]' }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["img", mergeAttributes(HTMLAttributes, { class: "mind-image", draggable: "true" })];
  },

  renderText({ node }) {
    return typeof node.attrs.alt === "string" ? node.attrs.alt : "";
  },
});

/** The node types a Word file's look is kept on (#746). */
export const WORD_LOOK_TYPES = ["paragraph", "heading", "table", "tableCell", "tableHeader"];

/**
 * A Word file's look on the Blocks of a Mind imported from it (#746): the key
 * of each paragraph's, heading's, table's and cell's style in the Mind's
 * `WordLook`. Enter keeps it, as Word gives a new paragraph the properties of
 * the one before.
 */
export const WordLookAttribute = Extension.create({
  name: "wordLookAttribute",

  addGlobalAttributes() {
    return [
      {
        types: WORD_LOOK_TYPES,
        attributes: {
          [WORD_LOOK_ATTRIBUTE]: {
            default: null,
            keepOnSplit: true,
            parseHTML: (element: HTMLElement) => element.getAttribute("data-word-look"),
            renderHTML: (attributes: Record<string, unknown>) => {
              const key = attributes[WORD_LOOK_ATTRIBUTE];
              return typeof key === "string" ? { "data-word-look": key } : {};
            },
          },
        },
      },
    ];
  },
});

/**
 * A run of text's look in a Mind imported from Word (#746): its style's key
 * in the Mind's `WordLook`. Inclusive, so what is typed at its end takes it,
 * as typing in Word takes the run's formatting.
 */
export const WordRun = Mark.create({
  name: WORD_RUN_MARK,
  inclusive: true,

  addAttributes() {
    return {
      k: {
        default: null,
        parseHTML: (element: HTMLElement) => element.getAttribute("data-word-run"),
        renderHTML: (attributes: Record<string, unknown>) =>
          typeof attributes.k === "string" ? { "data-word-run": attributes.k } : {},
      },
    };
  },

  parseHTML: () => [{ tag: "span[data-word-run]" }],
  renderHTML: ({ HTMLAttributes }) => ["span", HTMLAttributes, 0],
});
