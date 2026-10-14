import { mergeAttributes, Node, type NodeViewRenderer } from "@tiptap/core";
import { DOCUMENT_CHIP_NODE } from "../../../core/api";

export interface DocumentChipOptions {
  /** Draws the chip: the Document's name, which opens it. */
  view: NodeViewRenderer | null;
}

/**
 * A Document chip: an inline node the User writes with "@" in a Note, naming
 * a Document. It stores the Document's id, and its name as it was, for when
 * the Document is gone. It is atomic: the cursor passes over it, and Backspace
 * takes it out whole. Copied into another Mind it still names the Document.
 */
export const DocumentChip = Node.create<DocumentChipOptions>({
  name: DOCUMENT_CHIP_NODE,
  group: "inline",
  inline: true,
  atom: true,
  selectable: true,
  draggable: false,

  addOptions() {
    return { view: null };
  },

  addAttributes() {
    return {
      documentId: {
        default: "",
        parseHTML: (element: HTMLElement) => element.getAttribute("data-document-id") ?? "",
        renderHTML: (attributes: Record<string, unknown>) => ({
          "data-document-id": String(attributes.documentId ?? ""),
        }),
      },
      name: {
        default: "",
        parseHTML: (element: HTMLElement) => element.getAttribute("data-name") ?? "",
        renderHTML: (attributes: Record<string, unknown>) => ({
          "data-name": String(attributes.name ?? ""),
        }),
      },
    };
  },

  parseHTML() {
    return [{ tag: `span[data-type="${DOCUMENT_CHIP_NODE}"]` }];
  },

  renderHTML({ node, HTMLAttributes }) {
    return [
      "span",
      mergeAttributes(HTMLAttributes, { "data-type": DOCUMENT_CHIP_NODE }),
      String(node.attrs.name ?? ""),
    ];
  },

  renderText({ node }) {
    return String(node.attrs.name ?? "");
  },

  addNodeView() {
    return this.options.view;
  },
});
