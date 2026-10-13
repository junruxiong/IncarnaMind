/**
 * A Mind imported from a Word file, and the Blocks it was imported as, in one
 * shape (#105): JSON nodes, as the import made them, flattened into the units
 * that match the document's XML one for one. A paragraph, a heading or a list
 * item's paragraph is a `w:p`; a table, a `w:tbl`; a kept-as-is Block, the
 * element it names. Lists are flat in Word, so a list's items are units of
 * their own, with the list they are in.
 */
import * as Y from "yjs";
import type { MarkJSON, NodeJSON } from "../../../shared/blockJson";
import { readProposalAttribute } from "../../../shared/proposals";
import { readWordAtom } from "../../../shared/wordFiles";
import {
  ANSWER_BLOCK,
  BLOCK_ID_ATTRIBUTE,
  PROPOSAL_ATTRIBUTE,
  PROPOSAL_INSERT_MARK,
  WORD_BLOCK_NODE,
} from "../../api";

/** The formatting of text the export writes: the editor's marks, and its link. */
export interface TextMarks {
  bold: boolean;
  italic: boolean;
  underline: boolean;
  strike: boolean;
  highlight: boolean;
  code: boolean;
  link: string | null;
}

export const NO_MARKS: TextMarks = {
  bold: false,
  italic: false,
  underline: false,
  strike: false,
  highlight: false,
  code: false,
  link: null,
};

export function marksOf(marks: readonly MarkJSON[] | undefined): TextMarks {
  const has = (type: string) => marks?.some((mark) => mark.type === type) ?? false;
  const link = marks?.find((mark) => mark.type === "link")?.attrs?.href;
  return {
    bold: has("bold"),
    italic: has("italic"),
    underline: has("underline"),
    strike: has("strike"),
    highlight: has("highlight"),
    code: has("code"),
    link: typeof link === "string" && link !== "" ? link : null,
  };
}

export const sameMarks = (a: TextMarks, b: TextMarks): boolean =>
  a.bold === b.bold &&
  a.italic === b.italic &&
  a.underline === b.underline &&
  a.strike === b.strike &&
  a.highlight === b.highlight &&
  a.code === b.code &&
  a.link === b.link;

/**
 * A Mind's node as JSON, as the Mind reads it: what a Proposal adds, which
 * isn't the Mind's until it is accepted, is left out (ADR-0003). `elements`
 * maps each node back to its element, for what reads the Mind's own way
 * (Citations, footnotes).
 */
export function mindJson(
  element: Y.XmlElement,
  elements: Map<NodeJSON, Y.XmlElement>,
): NodeJSON | null {
  if (readProposalAttribute(element.getAttribute(PROPOSAL_ATTRIBUTE))?.kind === "insert") {
    return null;
  }
  const content: NodeJSON[] = [];
  for (const child of element.toArray()) {
    if (child instanceof Y.XmlText) {
      for (const run of child.toDelta() as {
        insert: unknown;
        attributes?: Record<string, unknown>;
      }[]) {
        const attributes = run.attributes ?? {};
        if (PROPOSAL_INSERT_MARK in attributes) continue;
        if (typeof run.insert !== "string" || run.insert === "") continue;
        const marks: MarkJSON[] = Object.entries(attributes).map(([type, value]) =>
          value && typeof value === "object"
            ? { type, attrs: value as Record<string, unknown> }
            : { type },
        );
        content.push({ type: "text", text: run.insert, ...(marks.length > 0 && { marks }) });
      }
    } else if (child instanceof Y.XmlElement) {
      const node = mindJson(child, elements);
      if (node) content.push(node);
    }
  }
  const node: NodeJSON = {
    type: element.nodeName,
    attrs: element.getAttributes() as Record<string, unknown>,
    content,
  };
  elements.set(node, element);
  return node;
}

/** A list a unit is an item of: the Mind's list node, whether it is numbered, and how deep. */
export interface ListPlace {
  list: NodeJSON;
  ordered: boolean;
  depth: number;
}

/** What a unit is, as the export writes it. */
export type UnitKind = "paragraph" | "table" | "kept" | "other";

/** One unit: a node and, for one imported from the document, the place of its XML. */
export interface Unit {
  node: NodeJSON;
  kind: UnitKind;
  /** The place of its XML ("w:p#12"); null for content the document didn't have. */
  key: string | null;
  list: ListPlace | null;
}

const LISTS = new Set(["bulletList", "orderedList", "taskList"]);
const PARAGRAPHS = new Set(["paragraph", "heading"]);

/** The id a node has as a Block. */
export const idOf = (node: NodeJSON): string | null => {
  const id = node.attrs?.[BLOCK_ID_ATTRIBUTE];
  return typeof id === "string" ? id : null;
};

/** Whether a kept-as-is Block is a header or a footer: its source names a part, not an element. */
export const isFurniture = (node: NodeJSON): boolean =>
  node.type === WORD_BLOCK_NODE && !readWordAtom(node.attrs ?? {}).source?.includes("#");

/**
 * The units in some Blocks, in order: list items one by one, an Answer's
 * Blocks as if they weren't in one. `keyOf` says where a node's XML is. A
 * header or footer isn't one: it is a part of the package of its own.
 */
export function unitsOf(
  blocks: readonly NodeJSON[],
  keyOf: (node: NodeJSON) => string | null,
  list: ListPlace | null = null,
  into: Unit[] = [],
): Unit[] {
  for (const node of blocks) {
    if (LISTS.has(node.type)) {
      const place = (depth: number): ListPlace => ({
        list: node,
        ordered: node.type === "orderedList",
        depth,
      });
      const depth = list ? list.depth + 1 : 0;
      for (const item of node.content ?? []) {
        unitsOf(item.content ?? [], keyOf, place(depth), into);
      }
    } else if (node.type === ANSWER_BLOCK) {
      unitsOf(node.content ?? [], keyOf, list, into);
    } else if (node.type === WORD_BLOCK_NODE) {
      if (isFurniture(node)) continue;
      const source = readWordAtom(node.attrs ?? {}).source;
      into.push({ node, kind: "kept", key: source, list });
    } else {
      const key = keyOf(node);
      const kind: UnitKind = PARAGRAPHS.has(node.type)
        ? "paragraph"
        : node.type === "table"
          ? "table"
          : "other";
      into.push({ node, kind, key: kind === "other" ? null : key, list });
    }
  }
  return into;
}

/** Every node under some nodes, those first, in document order. */
export function* nodesIn(nodes: readonly NodeJSON[]): Generator<NodeJSON> {
  for (const node of nodes) {
    yield node;
    yield* nodesIn(node.content ?? []);
  }
}
