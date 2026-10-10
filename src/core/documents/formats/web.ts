/**
 * A saved web page (#261) → Units. The snapshot is our own XHTML, written
 * when the page is saved (../web/snapshot): the page's title, address, site
 * and author in its head, and the readable article in its body, in a few
 * kinds of block (headings, paragraphs, list items, quotes, code, tables).
 *
 * A web page is cited by the section a quote sits under and its paragraph
 * ("§ History, ¶ 3"): one Unit per heading section, as for Markdown, a long
 * one split into parts at paragraph ends. Each paragraph of a section is
 * anchored "p1", "p2"…, counted from its heading across its parts, and the
 * label says so (`UnitLabel.paragraphs`).
 *
 * Pure, with no Node or DOM imports: the processing worker reads snapshots
 * with it, and the viewer reads them with the same code, so the blocks it
 * draws are the blocks that were indexed.
 */
import type { Anchor, TextUnit } from "../../../shared/units";
import { MAX_SECTION_CHARS } from "./docx";
import { ExtractionError } from "./errors";
import { elements, parseXml, textOf, type XmlElement, type XmlNode } from "./xml";

/** What a snapshot's head says of the page. */
export interface WebPageMeta {
  title: string | null;
  /** The address it was saved from. */
  address: string | null;
  siteName: string | null;
  author: string | null;
  /** When the page says it was published, as it gives it (ISO 8601 when it can be read). */
  published: string | null;
  /** The page's language, as its `lang` gives it. */
  language: string | null;
}

/** A block of the article: a heading, or a paragraph a Citation can point at. */
export interface WebBlock {
  element: XmlElement;
  /** A heading's level, 1 to 6; null for a paragraph. */
  level: number | null;
  /** The block's text, as it is indexed: a table's rows on lines, their cells separated by tabs. */
  text: string;
}

/** A Unit of a web page, with the blocks it holds (`first` to `last`, indexes into `blocks`). */
export interface WebUnit extends TextUnit {
  blocks: { first: number; last: number };
}

export interface WebPage {
  meta: WebPageMeta;
  /** The page's readable content; null if the snapshot has none. */
  article: XmlElement | null;
  blocks: WebBlock[];
  units: WebUnit[];
}

const HEADING = /^h([1-6])$/;

/** Elements whose content is block content: their children are read as blocks. */
const CONTAINERS = new Set([
  "article",
  "aside",
  "blockquote",
  "body",
  "details",
  "div",
  "dl",
  "figure",
  "footer",
  "header",
  "main",
  "ol",
  "section",
  "summary",
  "ul",
]);

/** Elements read as one paragraph each, unless they hold blocks of their own. */
const PARAGRAPHS = new Set(["p", "dt", "dd", "figcaption", "caption"]);

/** Every element that starts a block, for telling inline content from block content. */
export const BLOCK_ELEMENTS = new Set([
  ...CONTAINERS,
  ...PARAGRAPHS,
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "li",
  "pre",
  "table",
  "hr",
]);

const nameOf = (element: XmlElement) => element.name.toLowerCase();

const isBlock = (node: XmlNode): node is XmlElement =>
  typeof node !== "string" && BLOCK_ELEMENTS.has(nameOf(node));

/** Whitespace as HTML collapses it: runs of spaces, tabs and line breaks. */
const SPACES = /[ \t\n\r\f]+/g;

/** The text of inline content, as a browser shows it: spaces collapsed, a line break at each `br`. */
function inlineText(nodes: readonly XmlNode[]): string {
  const lines: string[] = [""];
  const walk = (list: readonly XmlNode[]) => {
    for (const node of list) {
      if (typeof node === "string") lines[lines.length - 1] += node;
      else if (nameOf(node) === "br") lines.push("");
      else if (nameOf(node) !== "img") walk(node.children);
    }
  };
  walk(nodes);
  return lines
    .map((line) => line.replace(SPACES, " ").trim())
    .join("\n")
    .trim();
}

/** A table's text: each row on a line, its cells separated by tabs. */
function tableText(table: XmlElement): string {
  const rows: string[] = [];
  const walk = (element: XmlElement) => {
    for (const child of elements(element)) {
      const name = nameOf(child);
      if (name === "tr") {
        const cells = elements(child)
          .filter((cell) => nameOf(cell) === "td" || nameOf(cell) === "th")
          .map((cell) => inlineText(cell.children).replace(/\s*\n\s*/g, " "));
        if (cells.some(Boolean)) rows.push(cells.join("\t"));
      } else if (name === "thead" || name === "tbody" || name === "tfoot") walk(child);
    }
  };
  walk(table);
  return rows.join("\n");
}

/** A `pre` block's text, as written: the line break that may open it isn't part of it. */
const preText = (pre: XmlElement) =>
  textOf(pre)
    .replace(/^\r?\n/, "")
    .trimEnd();

/** The article's blocks, in reading order. */
function collectBlocks(container: XmlElement, out: WebBlock[]): void {
  for (const element of elements(container)) {
    const name = nameOf(element);
    const heading = HEADING.exec(name);
    if (heading) {
      const text = inlineText(element.children);
      if (text) out.push({ element, level: Number(heading[1]), text });
    } else if (name === "pre") {
      const text = preText(element);
      if (text.trim()) out.push({ element, level: null, text });
    } else if (name === "table") {
      const text = tableText(element);
      if (text) out.push({ element, level: null, text });
    } else if (name === "li") {
      // Its own words, then any list or block inside it.
      const text = inlineText(element.children.filter((node) => !isBlock(node)));
      if (text) out.push({ element, level: null, text });
      collectBlocks(element, out);
    } else if (PARAGRAPHS.has(name) && !element.children.some(isBlock)) {
      const text = inlineText(element.children);
      if (text) out.push({ element, level: null, text });
    } else if (CONTAINERS.has(name) || PARAGRAPHS.has(name)) {
      collectBlocks(element, out);
    }
  }
}

function findElement(root: XmlElement, name: string): XmlElement | null {
  for (const element of elements(root)) {
    if (nameOf(element) === name) return element;
    const inside = findElement(element, name);
    if (inside) return inside;
  }
  return null;
}

function readMeta(root: XmlElement): WebPageMeta {
  const head = findElement(root, "head");
  const content = (key: string): string | null => {
    for (const meta of head ? elements(head) : []) {
      if (nameOf(meta) !== "meta") continue;
      if (meta.attrs.name !== key && meta.attrs.property !== key) continue;
      const value = meta.attrs.content?.trim();
      if (value) return value;
    }
    return null;
  };
  const title = head ? findElement(head, "title") : null;
  return {
    title: (title && inlineText(title.children)) || null,
    address: content("incarnamind:address"),
    siteName: content("og:site_name"),
    author: content("author"),
    published: content("article:published_time"),
    language: root.attrs.lang?.trim() || null,
  };
}

/** The Units of the blocks: one per heading section, split into parts past `MAX_SECTION_CHARS`. */
function unitsOf(blocks: readonly WebBlock[]): WebUnit[] {
  const units: WebUnit[] = [];
  const path: string[] = [];
  let current: { text: string; anchors: Anchor[]; first: number; part: number } | null = null;
  let paragraph = 0;

  const close = (last: number) => {
    if (!current?.text.trim()) return;
    const sectionPath = path.filter(Boolean);
    units.push({
      page: units.length + 1,
      kind: "section",
      label: {
        path: sectionPath,
        ...(current.part > 1 ? { part: current.part } : {}),
        paragraphs: true,
      },
      text: current.text,
      anchors: current.anchors,
      blocks: { first: current.first, last },
    });
  };

  blocks.forEach((block, index) => {
    if (block.level !== null) {
      close(index - 1);
      path.length = block.level - 1;
      path[block.level - 1] = block.text;
      paragraph = 0;
      current = { text: block.text, anchors: [], first: index, part: 1 };
      return;
    }
    if (current?.text && current.text.length + 1 + block.text.length > MAX_SECTION_CHARS) {
      close(index - 1);
      current = { text: "", anchors: [], first: index, part: current.part + 1 };
    }
    current ??= { text: "", anchors: [], first: index, part: 1 };
    if (current.text) current.text += "\n";
    const start = current.text.length;
    current.text += block.text;
    paragraph++;
    current.anchors.push({ start, end: current.text.length, target: `p${paragraph}` });
  });
  close(blocks.length - 1);
  return units;
}

/** Reads a snapshot. Throws `ExtractionError` if it isn't one. */
export function readWebPage(source: string): WebPage {
  const root = parseXml(source);
  if (nameOf(root) !== "html") {
    throw new ExtractionError("unreadable", "This isn't a saved web page.");
  }
  const body = findElement(root, "body");
  const article = body ? (findElement(body, "article") ?? body) : null;
  const blocks: WebBlock[] = [];
  if (article) collectBlocks(article, blocks);
  return { meta: readMeta(root), article, blocks, units: unitsOf(blocks) };
}

/** A web page's Units, as they are stored. */
export function webUnits(source: string): TextUnit[] {
  return readWebPage(source).units.map(({ blocks: _blocks, ...unit }) => unit);
}
