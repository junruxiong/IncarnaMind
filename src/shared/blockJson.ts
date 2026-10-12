/**
 * Blocks as ProseMirror (and Tiptap) write them in JSON, and the Citations in
 * an Answer's text: what the core writes into a Mind, and what the assistant
 * draws an Answer of a conversation with (ADR-0017), from the same Markdown
 * and Citations. Pure: the core and the window share it.
 */
import { CITATION_NODE, type Citation, type CitationAttributes } from "../core/api";

/** A node as ProseMirror (and Tiptap) write it in JSON. */
export interface NodeJSON {
  type: string;
  attrs?: Record<string, unknown>;
  content?: NodeJSON[];
  /** Text nodes only. */
  text?: string;
  /** Text nodes only. */
  marks?: MarkJSON[];
}

export interface MarkJSON {
  type: string;
  attrs?: Record<string, unknown>;
}

/** A Citation marker as the model writes it in the text: "[^1]". */
const CITATION_MARKER = /\[\^(\d{1,4})\]/g;

/**
 * A Citation node, every attribute set (null ones are left out, as y-tiptap
 * stores them); with no Citation, one still being checked.
 */
export function citationNode(citation: Citation | null): NodeJSON {
  const attrs: CitationAttributes = citation ?? {
    passageId: null,
    documentId: null,
    documentName: null,
    contentHash: null,
    pageFrom: null,
    pageTo: null,
    location: null,
    quote: null,
    check: "checking",
    checkReason: null,
  };
  return {
    type: CITATION_NODE,
    attrs: Object.fromEntries(Object.entries(attrs).filter(([, value]) => value !== null)),
  };
}

/**
 * Turns the markers in an Answer's text into Citation nodes: `node` gives a
 * marker's node, or null to remove the marker (with the space before it, so
 * "a claim [^3]." reads "a claim."). Text marked as code keeps its markers.
 */
export function withCitations(
  blocks: readonly NodeJSON[],
  node: (marker: number) => NodeJSON | null,
): NodeJSON[] {
  return blocks.map((block) => replaceMarkers(block, node));
}

function replaceMarkers(block: NodeJSON, node: (marker: number) => NodeJSON | null): NodeJSON {
  if (!block.content) return block;
  const content: NodeJSON[] = [];
  for (const child of block.content) {
    const isCode = child.marks?.some((mark) => mark.type === "code") ?? false;
    if (child.type !== "text" || !child.text || isCode) {
      content.push(replaceMarkers(child, node));
      continue;
    }
    let last = 0;
    for (const match of child.text.matchAll(CITATION_MARKER)) {
      let before = child.text.slice(last, match.index);
      const citation = node(Number(match[1]));
      // A removed marker takes the space before it, unless a word follows.
      if (!citation) {
        const after = child.text.slice(match.index + match[0].length);
        if (!/^\p{L}/u.test(after)) before = before.replace(/[ \t]+$/, "");
      }
      if (before) content.push({ ...child, text: before });
      if (citation) content.push(citation);
      last = match.index + match[0].length;
    }
    const rest = child.text.slice(last);
    if (rest) content.push(last === 0 ? child : { ...child, text: rest });
  }
  // Two text nodes with the same marks that a removed marker left side by side become one.
  const merged: NodeJSON[] = [];
  for (const child of content) {
    const previous = merged.at(-1);
    if (
      previous?.type === "text" &&
      child.type === "text" &&
      JSON.stringify(previous.marks ?? []) === JSON.stringify(child.marks ?? [])
    ) {
      merged[merged.length - 1] = {
        ...previous,
        text: `${previous.text ?? ""}${child.text ?? ""}`,
      };
    } else {
      merged.push(child);
    }
  }
  return { ...block, content: merged };
}
