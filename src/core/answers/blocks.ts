/**
 * Reading and writing Blocks in a Mind's Yjs document from the core, in the
 * shape Tiptap's Yjs binding (y-tiptap) gives them, so the editor shows what
 * the core writes and leaves it alone:
 *
 * - a node is a `Y.XmlElement` named after its type, with each non-null
 *   attribute set, the schema's defaults included;
 * - consecutive text nodes are one `Y.XmlText`, each mark a formatting
 *   attribute named after the mark, holding its attributes;
 * - inline nodes that aren't text (a formula, a line break) are elements
 *   between the texts.
 *
 * The core knows no editor schema: it writes only the node types and marks of
 * Notes (see `NOTE_BLOCK_TYPES`), and the tests check what it writes against
 * the editor's schema.
 */
import { createHash, randomUUID } from "node:crypto";
import * as Y from "yjs";
import { BLOCK_ID_ATTRIBUTE, NOTE_BLOCK_TYPES } from "../api";

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

type FormattingAttributes = Record<string, unknown>;

/** A stretch of text with one set of marks, as `Y.XmlText.toDelta()` gives it. */
interface Run {
  insert: string;
  attributes?: FormattingAttributes;
}

type Child = { kind: "text"; runs: Run[] } | { kind: "element"; node: NodeJSON };

/** Node types that carry a Block ID. Tiptap's UniqueID gives them one; the core does for those it writes. */
const ID_TYPES: ReadonlySet<string> = new Set(NOTE_BLOCK_TYPES);

/** The top-level Blocks of a Mind, in order. */
export function topLevelBlocks(blocks: Y.XmlFragment): Y.XmlElement[] {
  return blocks.toArray().filter((child): child is Y.XmlElement => child instanceof Y.XmlElement);
}

/** The top-level Block of this type with this ID, and its index among the Mind's children. */
export function findBlock(
  blocks: Y.XmlFragment,
  type: string,
  id: string,
): { element: Y.XmlElement; index: number } | null {
  const children = blocks.toArray();
  for (let index = 0; index < children.length; index++) {
    const element = children[index];
    if (
      element instanceof Y.XmlElement &&
      element.nodeName === type &&
      element.getAttribute(BLOCK_ID_ATTRIBUTE) === id
    ) {
      return { element, index };
    }
  }
  return null;
}

/** An attribute of an element, or null when it isn't set. */
export function attribute(element: Y.XmlElement, name: string): unknown {
  return element.getAttribute(name) ?? null;
}

export function textAttribute(element: Y.XmlElement, name: string): string | null {
  const value = element.getAttribute(name);
  return typeof value === "string" ? value : null;
}

// ---------------------------------------------------------------------------
// Writing

function childrenOf(content: readonly NodeJSON[]): Child[] {
  const children: Child[] = [];
  let text: Run[] | null = null;
  for (const node of content) {
    if (node.type === "text") {
      if (!node.text) continue;
      if (!text) {
        text = [];
        children.push({ kind: "text", runs: text });
      }
      text.push({ insert: node.text, attributes: formattingOf(node.marks) });
    } else {
      text = null;
      children.push({ kind: "element", node });
    }
  }
  return children;
}

function formattingOf(marks: readonly MarkJSON[] | undefined): FormattingAttributes | undefined {
  if (!marks || marks.length === 0) return undefined;
  return Object.fromEntries(marks.map((mark) => [mark.type, mark.attrs ?? {}]));
}

/** A node's attributes that are set (y-tiptap stores no null attribute). */
function setAttributes(node: NodeJSON): Record<string, unknown> {
  const set: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(node.attrs ?? {})) {
    if (value !== null && value !== undefined) set[key] = value;
  }
  return set;
}

/** The attributes to store for a new node: those set, plus a new Block ID if its type takes one. */
function attributesToStore(node: NodeJSON): Record<string, unknown> {
  const stored = setAttributes(node);
  if (ID_TYPES.has(node.type) && stored[BLOCK_ID_ATTRIBUTE] === undefined) {
    stored[BLOCK_ID_ATTRIBUTE] = randomUUID();
  }
  return stored;
}

function createText(runs: readonly Run[]): Y.XmlText {
  const text = new Y.XmlText();
  text.applyDelta(runs.map((run) => ({ insert: run.insert, attributes: run.attributes ?? {} })));
  return text;
}

/** A new Yjs element for a node and everything in it. */
export function createElement(node: NodeJSON): Y.XmlElement {
  const element = new Y.XmlElement(node.type);
  for (const [key, value] of Object.entries(attributesToStore(node))) {
    // biome-ignore lint/suspicious/noExplicitAny: Yjs types attribute values loosely.
    element.setAttribute(key, value as any);
  }
  element.insert(0, childrenOf(node.content ?? []).map(createChild));
  return element;
}

function createChild(child: Child): Y.XmlElement | Y.XmlText {
  return child.kind === "text" ? createText(child.runs) : createElement(child.node);
}

/**
 * Makes an element's content match `content` with as small a change as
 * possible: what is already there stays, and text that grew gets only the new
 * characters. A streaming Answer then changes only at its end, so nothing
 * above the end is redrawn.
 */
export function syncContent(element: Y.XmlElement, content: readonly NodeJSON[]): void {
  const wanted = childrenOf(content);
  for (let index = 0; index < wanted.length; index++) {
    const want = wanted[index] as Child;
    const current = element.get(index);
    if (current instanceof Y.XmlText && want.kind === "text") {
      syncText(current, want.runs);
    } else if (
      current instanceof Y.XmlElement &&
      want.kind === "element" &&
      current.nodeName === want.node.type
    ) {
      syncAttributes(current, want.node);
      syncContent(current, want.node.content ?? []);
    } else {
      if (current !== undefined) element.delete(index, 1);
      element.insert(index, [createChild(want)]);
    }
  }
  if (element.length > wanted.length) element.delete(wanted.length, element.length - wanted.length);
}

function syncAttributes(element: Y.XmlElement, node: NodeJSON): void {
  const wanted = setAttributes(node);
  delete wanted[BLOCK_ID_ATTRIBUTE];
  const current = element.getAttributes() as Record<string, unknown>;
  for (const [key, value] of Object.entries(wanted)) {
    // biome-ignore lint/suspicious/noExplicitAny: Yjs types attribute values loosely.
    if (!sameValue(current[key], value)) element.setAttribute(key, value as any);
  }
  // The Block ID stays, whoever set it.
  for (const key of Object.keys(current)) {
    if (key !== BLOCK_ID_ATTRIBUTE && !(key in wanted)) element.removeAttribute(key);
  }
}

/** Keeps the longest common start of the text, then writes the rest. */
function syncText(text: Y.XmlText, wanted: readonly Run[]): void {
  const current = (text.toDelta() as Run[]).filter((run) => typeof run.insert === "string");
  let keep = commonStart(current, wanted);
  const joined = wanted.map((run) => run.insert).join("");
  // Never split a surrogate pair.
  if (keep > 0 && isHighSurrogate(joined.charCodeAt(keep - 1))) keep--;
  if (keep < text.length) text.delete(keep, text.length - keep);
  let offset = 0;
  let at = keep;
  for (const run of wanted) {
    const end = offset + run.insert.length;
    if (end > keep) {
      const piece = run.insert.slice(Math.max(0, keep - offset));
      text.insert(at, piece, run.attributes ?? {});
      at += piece.length;
    }
    offset = end;
  }
}

const isHighSurrogate = (code: number) => code >= 0xd800 && code <= 0xdbff;

/** How many characters two runs of text share from their start, marks included. */
function commonStart(a: readonly Run[], b: readonly Run[]): number {
  let length = 0;
  let i = 0;
  let j = 0;
  let offsetA = 0;
  let offsetB = 0;
  while (i < a.length && j < b.length) {
    const runA = a[i] as Run;
    const runB = b[j] as Run;
    if (!sameValue(runA.attributes ?? {}, runB.attributes ?? {})) break;
    while (
      offsetA < runA.insert.length &&
      offsetB < runB.insert.length &&
      runA.insert[offsetA] === runB.insert[offsetB]
    ) {
      offsetA++;
      offsetB++;
      length++;
    }
    if (offsetA < runA.insert.length && offsetB < runB.insert.length) break;
    if (offsetA === runA.insert.length) {
      i++;
      offsetA = 0;
    }
    if (offsetB === runB.insert.length) {
      j++;
      offsetB = 0;
    }
  }
  return length;
}

/** Deep equality for attribute values (JSON), ignoring key order and treating null keys as absent. */
function sameValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    return (
      Array.isArray(a) &&
      Array.isArray(b) &&
      a.length === b.length &&
      a.every((item, index) => sameValue(item, b[index]))
    );
  }
  const entriesOf = (value: object) =>
    Object.entries(value).filter(([, item]) => item !== null && item !== undefined);
  const left = entriesOf(a);
  const right = new Map(entriesOf(b));
  return (
    left.length === right.size &&
    left.every(([key, item]) => right.has(key) && sameValue(item, right.get(key)))
  );
}

// ---------------------------------------------------------------------------
// Reading, as Markdown

/** A Block (or any element) as Markdown, e.g. for Question context. */
export function toMarkdown(element: Y.XmlElement): string {
  return blockMarkdown(element).trim();
}

/** An element's content as Markdown: its child Blocks separated by blank lines. */
export function contentMarkdown(element: Y.XmlElement): string {
  return blocksMarkdown(element.toArray()).trim();
}

/** A fingerprint of an element's content, to tell whether it changed since. */
export function contentHash(element: Y.XmlElement): string {
  return createHash("sha256").update(contentMarkdown(element)).digest("base64url").slice(0, 22);
}

function blocksMarkdown(children: readonly (Y.XmlElement | Y.XmlText | Y.XmlHook)[]): string {
  return children
    .map((child) => (child instanceof Y.XmlElement ? blockMarkdown(child) : textOf(child)))
    .filter((text) => text.trim() !== "")
    .join("\n\n");
}

function blockMarkdown(element: Y.XmlElement): string {
  switch (element.nodeName) {
    case "heading": {
      const level = Number(element.getAttribute("level")) || 1;
      return `${"#".repeat(Math.min(Math.max(level, 1), 6))} ${inlineMarkdown(element)}`;
    }
    case "codeBlock": {
      const language = textAttribute(element, "language") ?? "";
      const code = plainText(element);
      const fence = code.includes("```") ? "````" : "```";
      return `${fence}${language}\n${code}\n${fence}`;
    }
    case "blockMath":
      return `$$\n${textAttribute(element, "latex") ?? ""}\n$$`;
    case "horizontalRule":
      return "---";
    case "bulletList":
    case "orderedList":
      return listMarkdown(element);
    case "blockquote":
      return blocksMarkdown(element.toArray())
        .split("\n")
        .map((line) => (line ? `> ${line}` : ">"))
        .join("\n");
    case "listItem":
    case "answer":
      return blocksMarkdown(element.toArray());
    default:
      // Paragraphs, Questions, and node types this version doesn't know: their inline content.
      if (element.toArray().some((child) => child instanceof Y.XmlElement && isBlock(child))) {
        return blocksMarkdown(element.toArray());
      }
      return inlineMarkdown(element);
  }
}

const INLINE_TYPES = new Set(["inlineMath", "hardBreak"]);
const isBlock = (element: Y.XmlElement) => !INLINE_TYPES.has(element.nodeName);

function listMarkdown(list: Y.XmlElement): string {
  const ordered = list.nodeName === "orderedList";
  let number = Number(list.getAttribute("start")) || 1;
  const items: string[] = [];
  for (const item of list.toArray()) {
    if (!(item instanceof Y.XmlElement)) continue;
    const marker = ordered ? `${number++}.` : "-";
    const indent = " ".repeat(marker.length + 1);
    const body = blocksMarkdown(item.toArray())
      .split("\n")
      .map((line, index) => (index === 0 ? `${marker} ${line}` : line ? `${indent}${line}` : ""))
      .join("\n");
    items.push(body);
  }
  return items.join("\n");
}

/** Inline content with marks as Markdown: **bold**, *italic*, ~~strike~~, `code`, [links](…), $math$. */
function inlineMarkdown(element: Y.XmlElement): string {
  let markdown = "";
  for (const child of element.toArray()) {
    if (child instanceof Y.XmlText) {
      for (const run of child.toDelta() as Run[]) {
        if (typeof run.insert === "string") markdown += markRun(run);
      }
    } else if (child instanceof Y.XmlElement) {
      if (child.nodeName === "inlineMath") markdown += `$${textAttribute(child, "latex") ?? ""}$`;
      else if (child.nodeName === "hardBreak") markdown += "\n";
      else markdown += blockMarkdown(child);
    }
  }
  return markdown;
}

function markRun({ insert, attributes = {} }: Run): string {
  if (Object.hasOwn(attributes, "code")) return `\`${insert}\``;
  let text = insert;
  if (Object.hasOwn(attributes, "bold")) text = `**${text}**`;
  if (Object.hasOwn(attributes, "italic")) text = `*${text}*`;
  if (Object.hasOwn(attributes, "strike")) text = `~~${text}~~`;
  const link = attributes.link as { href?: unknown } | undefined;
  if (link && typeof link.href === "string") text = `[${text}](${link.href})`;
  return text;
}

/** All the text in an element, without marks or structure. */
export function plainText(element: Y.XmlElement | Y.XmlText): string {
  if (element instanceof Y.XmlText) return textOf(element);
  return element
    .toArray()
    .map((child) =>
      child instanceof Y.XmlElement || child instanceof Y.XmlText ? plainText(child) : "",
    )
    .join("");
}

function textOf(child: Y.XmlText | Y.XmlHook): string {
  if (!(child instanceof Y.XmlText)) return "";
  return (child.toDelta() as Run[])
    .map((run) => (typeof run.insert === "string" ? run.insert : ""))
    .join("");
}
