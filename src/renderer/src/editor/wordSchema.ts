import { mergeAttributes, Node, type NodeViewRenderer } from "@tiptap/core";
import { isChangeOrigin } from "@tiptap/extension-collaboration";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { Plugin, PluginKey, type Transaction } from "@tiptap/pm/state";
import { IMAGE_NODE, WORD_ATOM_NODE, WORD_BLOCK_NODE } from "../../../core/api";
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
  /**
   * Called instead of a change that would delete a section break: `retry`
   * makes the change after all. Without it, such a change goes through.
   */
  onSectionBreakDelete: ((retry: () => void) => void) | null;
}

/** Set on a transaction to let it delete section breaks: the User was asked. */
const SECTION_BREAKS_ASKED = "wordSectionBreaksAsked";

const isSectionBreak = (node: ProseMirrorNode) =>
  node.type.name === WORD_ATOM_NODE && node.attrs.kind === "sectionBreak";

const sectionBreaksIn = (doc: ProseMirrorNode) => {
  let total = 0;
  doc.descendants((node) => {
    if (isSectionBreak(node)) total++;
  });
  return total;
};

/**
 * Whether a transaction removes a section break: only where it changes the
 * document is looked at, and only then is the whole counted, so typing stays fast.
 */
function deletesSectionBreak(tr: Transaction): boolean {
  let touched = false;
  tr.steps.forEach((step, index) => {
    const before = tr.docs[index];
    if (!before || touched) return;
    step.getMap().forEach((from, to) => {
      if (touched || to <= from) return;
      before.nodesBetween(from, to, (node) => {
        if (isSectionBreak(node)) touched = true;
        return !touched;
      });
    });
  });
  // One moved, not removed (a paragraph dragged elsewhere), keeps the count.
  return touched && sectionBreaksIn(tr.doc) < sectionBreaksIn(tr.before);
}

/**
 * What a Word import kept as it is, inside a paragraph's text (#104): a
 * field, a content control, a text box, a footnote's mark, another person's
 * tracked change… (see `WordAtomAttributes`). It can't be edited inside; it
 * moves with the text and goes when that text is deleted, except a section
 * break, whose deletion asks first (`onSectionBreakDelete`).
 */
export const WordAtom = Node.create<WordAtomOptions>({
  name: WORD_ATOM_NODE,
  group: "inline",
  inline: true,
  atom: true,
  selectable: true,
  draggable: false,

  addOptions() {
    return { view: null, onSectionBreakDelete: null };
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

  addProseMirrorPlugins() {
    const ask = this.options.onSectionBreakDelete;
    if (!ask) return [];
    return [
      new Plugin({
        key: new PluginKey("wordSectionBreaks"),
        filterTransaction: (tr) => {
          // Changes from Yjs (another device, or undo) are never held back.
          if (!tr.docChanged || tr.getMeta(SECTION_BREAKS_ASKED) || isChangeOrigin(tr)) return true;
          if (!deletesSectionBreak(tr)) return true;
          ask(() => {
            const view = this.editor.view;
            const retry = view.state.tr;
            for (const step of tr.steps) retry.maybeStep(step);
            view.dispatch(retry.setMeta(SECTION_BREAKS_ASKED, true));
          });
          return false;
        },
      }),
    ];
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
