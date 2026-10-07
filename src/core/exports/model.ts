/**
 * A Mind as it is exported: its Blocks read out of the Yjs document into plain
 * data, which the Markdown and .docx writers both render. Reading happens once,
 * so the two formats always agree on what is exported, and on the order (and
 * so the numbers) of the footnotes.
 *
 * Node types are those the editor stores (Tiptap's names, see `noteSchema`),
 * plus Tiptap's tables and images, which the export handles when a Mind has
 * them. A node type this version doesn't know is kept as its text.
 */
import * as Y from "yjs";
import { ANSWER_BLOCK, CITATION_NODE, QUESTION_BLOCK } from "../api";

export interface Marks {
  bold: boolean;
  italic: boolean;
  strike: boolean;
  underline: boolean;
  highlight: boolean;
  code: boolean;
  /** The link's target, or null. */
  link: string | null;
}

/** A Citation's footnote: one per Citation, in the order they appear. */
export interface Footnote {
  /** The Document's name and the Citation's Location, e.g. "Tides, p. 12–13" or "Deck, slide 4". */
  source: string;
  /** The quote wasn't found, or can't be checked: the footnote gets the "[unverified]" marker. */
  unverified: boolean;
}

export interface Image {
  src: string;
  alt: string;
}

export type Inline =
  | { kind: "text"; text: string; marks: Marks }
  | { kind: "math"; latex: string }
  | { kind: "break" }
  | { kind: "citation"; footnote: Footnote }
  | { kind: "image"; image: Image };

export interface TableCell {
  header: boolean;
  /** How many columns the cell spans, from 1. */
  colspan: number;
  content: Block[];
}

export type Block =
  | { kind: "paragraph"; content: Inline[] }
  | { kind: "heading"; level: number; content: Inline[] }
  | { kind: "code"; language: string; code: string }
  | { kind: "math"; latex: string }
  | { kind: "list"; ordered: boolean; start: number; items: Block[][] }
  | { kind: "quote"; content: Block[] }
  | { kind: "rule" }
  | { kind: "table"; rows: TableCell[][] }
  | { kind: "image"; image: Image }
  | { kind: "question"; content: Inline[] };

export interface ReadOptions {
  includeQuestions: boolean;
  /** The footnote for a Citation, from its node's attributes. */
  footnote(attributes: Record<string, unknown>): Footnote;
}

/** A stretch of text with one set of marks, as `Y.XmlText.toDelta()` gives it. */
interface Run {
  insert: unknown;
  attributes?: Record<string, unknown>;
}

/** The Mind's Blocks, in order. Answers are unwrapped into their Blocks. */
export function readBlocks(blocks: Y.XmlFragment, options: ReadOptions): Block[] {
  return readChildren(blocks.toArray(), options);
}

/** How many Question Blocks the Mind has. */
export function countQuestions(blocks: Y.XmlFragment): number {
  return blocks
    .toArray()
    .filter((child) => child instanceof Y.XmlElement && child.nodeName === QUESTION_BLOCK).length;
}

/** Every footnote in the Blocks, in order. */
export function footnotesIn(blocks: readonly Block[]): Footnote[] {
  const found: Footnote[] = [];
  const inInline = (content: readonly Inline[]) => {
    for (const inline of content) if (inline.kind === "citation") found.push(inline.footnote);
  };
  const visit = (block: Block) => {
    switch (block.kind) {
      case "paragraph":
      case "heading":
      case "question":
        inInline(block.content);
        break;
      case "list":
        for (const item of block.items) item.forEach(visit);
        break;
      case "quote":
        block.content.forEach(visit);
        break;
      case "table":
        for (const row of block.rows) for (const cell of row) cell.content.forEach(visit);
        break;
    }
  };
  blocks.forEach(visit);
  return found;
}

function readChildren(
  children: readonly (Y.XmlElement | Y.XmlText | Y.XmlHook)[],
  options: ReadOptions,
): Block[] {
  const blocks: Block[] = [];
  for (const child of children) {
    if (child instanceof Y.XmlElement) blocks.push(...readBlock(child, options));
    else if (child instanceof Y.XmlText) {
      // Text straight in a Block container: keep it as a paragraph.
      const content = readInline([child], options);
      if (content.length > 0) blocks.push({ kind: "paragraph", content });
    }
  }
  return blocks;
}

const INLINE_TYPES: ReadonlySet<string> = new Set([
  "inlineMath",
  "hardBreak",
  "image",
  CITATION_NODE,
]);

const hasBlockChildren = (element: Y.XmlElement) =>
  element
    .toArray()
    .some((child) => child instanceof Y.XmlElement && !INLINE_TYPES.has(child.nodeName));

function readBlock(element: Y.XmlElement, options: ReadOptions): Block[] {
  switch (element.nodeName) {
    case "paragraph":
      return [{ kind: "paragraph", content: readInline(element.toArray(), options) }];
    case "heading": {
      const level = Math.min(Math.max(Number(element.getAttribute("level")) || 1, 1), 6);
      return [{ kind: "heading", level, content: readInline(element.toArray(), options) }];
    }
    case "codeBlock":
      return [
        { kind: "code", language: stringAttribute(element, "language"), code: plainText(element) },
      ];
    case "blockMath":
      return [{ kind: "math", latex: stringAttribute(element, "latex") }];
    case "bulletList":
    case "orderedList": {
      const ordered = element.nodeName === "orderedList";
      const start = ordered
        ? Math.max(Math.trunc(Number(element.getAttribute("start"))) || 1, 0)
        : 1;
      const items = element
        .toArray()
        .filter((item): item is Y.XmlElement => item instanceof Y.XmlElement)
        .map((item) => readChildren(item.toArray(), options));
      return [{ kind: "list", ordered, start, items }];
    }
    case "blockquote":
      return [{ kind: "quote", content: readChildren(element.toArray(), options) }];
    case "horizontalRule":
      return [{ kind: "rule" }];
    case "table":
      return [{ kind: "table", rows: readTable(element, options) }];
    case "image":
      return [{ kind: "image", image: readImage(element) }];
    case QUESTION_BLOCK:
      return options.includeQuestions
        ? [{ kind: "question", content: readInline(element.toArray(), options) }]
        : [];
    case ANSWER_BLOCK:
      return readChildren(element.toArray(), options);
    default:
      // A node type this version doesn't know: its Blocks, or its text.
      if (hasBlockChildren(element)) return readChildren(element.toArray(), options);
      return [{ kind: "paragraph", content: readInline(element.toArray(), options) }];
  }
}

function readTable(table: Y.XmlElement, options: ReadOptions): TableCell[][] {
  const rows: TableCell[][] = [];
  for (const row of table.toArray()) {
    if (!(row instanceof Y.XmlElement)) continue;
    const cells: TableCell[] = [];
    for (const cell of row.toArray()) {
      if (!(cell instanceof Y.XmlElement)) continue;
      cells.push({
        header: cell.nodeName === "tableHeader",
        colspan: Math.max(Math.trunc(Number(cell.getAttribute("colspan"))) || 1, 1),
        content: readChildren(cell.toArray(), options),
      });
    }
    if (cells.length > 0) rows.push(cells);
  }
  return rows;
}

function readImage(element: Y.XmlElement): Image {
  return { src: stringAttribute(element, "src"), alt: stringAttribute(element, "alt") };
}

const NO_MARKS: Marks = {
  bold: false,
  italic: false,
  strike: false,
  underline: false,
  highlight: false,
  code: false,
  link: null,
};

function marksOf(attributes: Record<string, unknown> = {}): Marks {
  const link = attributes.link as { href?: unknown } | undefined;
  return {
    bold: Object.hasOwn(attributes, "bold"),
    italic: Object.hasOwn(attributes, "italic"),
    strike: Object.hasOwn(attributes, "strike"),
    underline: Object.hasOwn(attributes, "underline"),
    highlight: Object.hasOwn(attributes, "highlight"),
    code: Object.hasOwn(attributes, "code"),
    link: link && typeof link.href === "string" && link.href !== "" ? link.href : null,
  };
}

function readInline(
  children: readonly (Y.XmlElement | Y.XmlText | Y.XmlHook)[],
  options: ReadOptions,
): Inline[] {
  const content: Inline[] = [];
  for (const child of children) {
    if (child instanceof Y.XmlText) {
      for (const run of child.toDelta() as Run[]) {
        if (typeof run.insert === "string" && run.insert !== "") {
          content.push({ kind: "text", text: run.insert, marks: marksOf(run.attributes) });
        }
      }
    } else if (child instanceof Y.XmlElement) {
      switch (child.nodeName) {
        case "inlineMath":
          content.push({ kind: "math", latex: stringAttribute(child, "latex") });
          break;
        case "hardBreak":
          content.push({ kind: "break" });
          break;
        case "image":
          content.push({ kind: "image", image: readImage(child) });
          break;
        case CITATION_NODE:
          trimTrailingSpace(content);
          content.push({
            kind: "citation",
            footnote: options.footnote(child.getAttributes() as Record<string, unknown>),
          });
          break;
        default: {
          // Inline content this version doesn't know: its text.
          const text = plainText(child);
          if (text) content.push({ kind: "text", text, marks: NO_MARKS });
        }
      }
    }
  }
  return content;
}

/**
 * A footnote's number goes right after the word it follows, as in print: the
 * space the editor keeps before a Citation is dropped.
 */
function trimTrailingSpace(content: Inline[]): void {
  const last = content.at(-1);
  if (last?.kind !== "text") return;
  const text = last.text.replace(/[ \t ]+$/, "");
  if (text) content[content.length - 1] = { ...last, text };
  else content.pop();
}

function stringAttribute(element: Y.XmlElement, name: string): string {
  const value = element.getAttribute(name);
  return typeof value === "string" ? value : "";
}

function plainText(element: Y.XmlElement | Y.XmlText): string {
  if (element instanceof Y.XmlText) {
    return (element.toDelta() as Run[])
      .map((run) => (typeof run.insert === "string" ? run.insert : ""))
      .join("");
  }
  return element
    .toArray()
    .map((child) =>
      child instanceof Y.XmlElement || child instanceof Y.XmlText ? plainText(child) : "",
    )
    .join("");
}
