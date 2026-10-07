import { mergeAttributes, Node, type NodeViewRenderer } from "@tiptap/core";
import { CITATION_NODE, type CitationAttributes } from "../../../core/api";
import { citationReference } from "../../../shared/citations";
import { parseLocation } from "../../../shared/locations";

export interface CitationOptions {
  /** Draws the Citation: its badge, and what clicking it does. */
  view: NodeViewRenderer | null;
}

/**
 * An attribute kept in the HTML as `data-<name>`, so a Citation copied and
 * pasted (into a Note, or another Mind) keeps everything it stores.
 */
function dataAttribute(name: string, kind: "text" | "page" = "text") {
  const key = `data-${name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}`;
  return {
    default: null,
    parseHTML: (element: HTMLElement) => {
      const value = element.getAttribute(key);
      if (value === null) return null;
      if (kind === "text") return value;
      const page = Number(value);
      return Number.isInteger(page) && page >= 1 ? page : null;
    },
    renderHTML: (attributes: Record<string, unknown>) => {
      const value = attributes[name];
      return value === null || value === undefined ? {} : { [key]: String(value) };
    },
  };
}

/**
 * A Citation: an inline node in the text of an Answer, which the core writes
 * for each marker the model cited (see `CitationAttributes`). It moves with
 * the text around it, goes when that text is deleted, and keeps its Passage,
 * quote and check result when copied, e.g. into a Note.
 */
export const Citation = Node.create<CitationOptions>({
  name: CITATION_NODE,
  group: "inline",
  inline: true,
  atom: true,
  selectable: true,
  draggable: false,

  addOptions() {
    return { view: null };
  },

  addAttributes() {
    const attributes: Record<keyof CitationAttributes, object> = {
      passageId: dataAttribute("passageId"),
      documentId: dataAttribute("documentId"),
      documentName: dataAttribute("documentName"),
      contentHash: dataAttribute("contentHash"),
      pageFrom: dataAttribute("pageFrom", "page"),
      pageTo: dataAttribute("pageTo", "page"),
      // Where it points (ADR-0011), as JSON in the HTML.
      location: {
        default: null,
        parseHTML: (element: HTMLElement) => parseLocation(element.getAttribute("data-location")),
        renderHTML: (attributes: Record<string, unknown>) => {
          const location = parseLocation(attributes.location);
          return location ? { "data-location": JSON.stringify(location) } : {};
        },
      },
      quote: dataAttribute("quote"),
      check: {
        default: "checking",
        parseHTML: (element: HTMLElement) => element.getAttribute("data-check") ?? "checking",
        renderHTML: (attributes: Record<string, unknown>) => ({ "data-check": attributes.check }),
      },
      checkReason: dataAttribute("checkReason"),
    };
    return attributes;
  },

  parseHTML() {
    return [{ tag: `span[data-type="${CITATION_NODE}"]` }];
  },

  renderHTML({ node, HTMLAttributes }) {
    return [
      "span",
      mergeAttributes(HTMLAttributes, { "data-type": CITATION_NODE }),
      citationReference(node.attrs as CitationAttributes),
    ];
  },

  renderText({ node }) {
    return citationReference(node.attrs as CitationAttributes);
  },

  addNodeView() {
    return this.options.view;
  },
});
