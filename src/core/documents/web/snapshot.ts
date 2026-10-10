/**
 * A web page's snapshot (#261): its readable article, found by Mozilla's
 * Readability (Apache-2.0) in a linkedom DOM, written as our own XHTML (read
 * by ../formats/web). Only a few kinds of block and inline element are
 * kept, and of attributes only a link's http(s) address, a picture's
 * description and a table cell's span: no script, style, frame, form, event
 * handler or picture address survives, so nothing in it runs or loads.
 */
import { Readability } from "@mozilla/readability";
import { parseHTML } from "linkedom";
import { CJK } from "../../../shared/text";
import { escapeXml } from "../../exports/wordml";
import { BLOCK_ELEMENTS, readWebPage } from "../formats/web";
import { WebPageError } from "./errors";

/** The DOM, as far as this module reads linkedom's. */
interface DomNode {
  readonly nodeType: number;
  readonly nodeValue: string | null;
  readonly childNodes: ArrayLike<DomNode>;
  readonly textContent: string | null;
}

interface DomElement extends DomNode {
  readonly localName: string;
  getAttribute(name: string): string | null;
  remove(): void;
  querySelectorAll(selector: string): ArrayLike<DomElement>;
}

interface DomDocument {
  readonly documentElement: DomElement | null;
  readonly title: string;
  querySelector(selector: string): DomElement | null;
  querySelectorAll(selector: string): ArrayLike<DomElement>;
}

export interface Snapshot {
  /** The snapshot file's text. */
  xhtml: string;
  title: string;
  siteName: string;
  author: string | null;
  published: string | null;
}

const ELEMENT = 1;
const TEXT = 3;

/** Left out with everything in them. */
const DROPPED = new Set([
  "area",
  "audio",
  "base",
  "button",
  "canvas",
  "dialog",
  "embed",
  "form",
  "frame",
  "frameset",
  "head",
  "iframe",
  "input",
  "link",
  "map",
  "meta",
  "nav",
  "noscript",
  "object",
  "option",
  "param",
  "script",
  "select",
  "source",
  "style",
  "svg",
  "template",
  "textarea",
  "title",
  "track",
  "video",
]);

const INLINE_TAGS: Readonly<Record<string, string>> = {
  b: "strong",
  strong: "strong",
  em: "em",
  i: "em",
  code: "code",
  kbd: "code",
  samp: "code",
  tt: "code",
  sub: "sub",
  sup: "sup",
  s: "s",
  del: "s",
  strike: "s",
  mark: "mark",
};

/** Containers kept as they are; any other container is unwrapped. */
const KEPT_CONTAINERS = new Set(["blockquote", "figure", "dl"]);
const LEAF_BLOCKS = new Set(["p", "dt", "dd", "figcaption", "caption", "address"]);

/** Letters and digits a page needs to be worth reading; a CJK character counts as three. */
const MIN_READABLE = 300;
/** A page with a password field and less than this is a sign-in page. */
const MIN_BEHIND_SIGN_IN = 1500;

const CJK_CHARACTER = new RegExp(`[${CJK}\\u3000-\\u303f\\uff00-\\uffef]`, "u");

const isElement = (node: DomNode): node is DomElement => node.nodeType === ELEMENT;

const nameOf = (element: DomElement) => element.localName.toLowerCase();

function hasBlocks(element: DomElement): boolean {
  for (const child of Array.from(element.childNodes)) {
    if (!isElement(child) || DROPPED.has(nameOf(child))) continue;
    if (BLOCK_ELEMENTS.has(nameOf(child)) || hasBlocks(child)) return true;
  }
  return false;
}

/** The first block with text in an element, if no text comes before it. */
function firstBlock(element: DomNode): DomElement | null {
  for (const child of Array.from(element.childNodes)) {
    if (child.nodeType === TEXT) {
      if ((child.nodeValue ?? "").trim()) return null;
      continue;
    }
    if (!isElement(child) || DROPPED.has(nameOf(child)) || !(child.textContent ?? "").trim()) {
      continue;
    }
    return hasBlocks(child) ? firstBlock(child) : child;
  }
  return null;
}

/** A footnote or reference mark ("[3]", "[citation needed]"): the quote a model gives leaves it out. */
function isReferenceMark(sup: DomElement): boolean {
  const text = (sup.textContent ?? "").trim();
  if (/^\[[^\]]{1,40}\]$/.test(text)) return true;
  const links = Array.from(sup.childNodes).filter(isElement);
  return (
    text.length <= 4 &&
    links.length === 1 &&
    /#/.test((links[0] as DomElement).getAttribute("href") ?? "")
  );
}

/** Inline content as XHTML, its whitespace collapsed as a browser shows it. */
class InlineWriter {
  private parts: string[] = [];
  private pendingSpace = false;
  private last = "";
  /** Where the tags opened since the last text start: a space owed goes before them, not inside. */
  private openedAt: number | null = null;
  /** Letters and digits written. */
  text = 0;
  images = 0;

  constructor(private readonly base: string) {}

  get xhtml(): string {
    return this.parts.join("");
  }

  private write(text: string): void {
    for (const piece of text.split(/([ \t\n\r\f\u00a0]+)/)) {
      if (!piece) continue;
      if (/^[ \t\n\r\f\u00a0]+$/.test(piece)) {
        this.pendingSpace = this.last !== "" && this.last !== "\n";
        continue;
      }
      if (
        this.pendingSpace &&
        !(CJK_CHARACTER.test(this.last) && CJK_CHARACTER.test(piece[0] ?? ""))
      ) {
        this.parts.splice(this.openedAt ?? this.parts.length, 0, " ");
      }
      this.pendingSpace = false;
      this.openedAt = null;
      this.parts.push(escapeXml(piece));
      this.last = piece.at(-1) ?? "";
      this.text += piece.match(/[\p{L}\p{N}]/gu)?.length ?? 0;
    }
  }

  private lineBreak(): void {
    if (this.last === "" || this.last === "\n") return;
    this.parts.push("<br/>");
    this.pendingSpace = false;
    this.last = "\n";
  }

  /** The address of a link, resolved, if it is a web page's. */
  private href(element: DomElement): string | null {
    const raw = element.getAttribute("href");
    if (!raw) return null;
    const url = URL.parse(raw, this.base);
    return url && (url.protocol === "https:" || url.protocol === "http:") ? url.href : null;
  }

  node(node: DomNode, flattenBlocks = false): void {
    if (node.nodeType === TEXT) {
      this.write(node.nodeValue ?? "");
      return;
    }
    if (!isElement(node)) return;
    const name = nameOf(node);
    if (DROPPED.has(name)) return;
    if (name === "br") {
      this.lineBreak();
      return;
    }
    if (name === "img") {
      const size =
        Number(node.getAttribute("width") ?? "2") * Number(node.getAttribute("height") ?? "2");
      if (size <= 1) return; // a tracking pixel
      const alt = (node.getAttribute("alt") ?? "").replace(/\s+/g, " ").trim();
      this.parts.push(`<img alt="${escapeXml(alt)}"/>`);
      this.images++;
      return;
    }
    if (name === "math") {
      this.write(node.getAttribute("alttext") ?? "");
      return;
    }
    if (name === "sup" && isReferenceMark(node)) return;
    const tag = name === "a" ? (this.href(node) ? "a" : null) : (INLINE_TAGS[name] ?? null);
    const block = flattenBlocks && BLOCK_ELEMENTS.has(name);
    if (block) this.lineBreak();
    if (tag) this.openedAt ??= this.parts.length;
    if (tag === "a") this.parts.push(`<a href="${escapeXml(this.href(node) ?? "")}">`);
    else if (tag) this.parts.push(`<${tag}>`);
    for (const child of Array.from(node.childNodes)) this.node(child, flattenBlocks);
    if (tag) {
      this.parts.push(`</${tag}>`);
      this.openedAt = null;
    }
    if (block) this.lineBreak();
  }
}

/** The article's blocks as XHTML, in the shapes ../formats/web reads. */
class BlockWriter {
  readonly parts: string[] = [];

  constructor(private readonly base: string) {}

  private inline(nodes: readonly DomNode[], flattenBlocks = false): InlineWriter {
    const writer = new InlineWriter(this.base);
    for (const node of nodes) writer.node(node, flattenBlocks);
    return writer;
  }

  private leaf(tag: string, nodes: readonly DomNode[]): void {
    const content = this.inline(nodes);
    if (content.text > 0 || content.images > 0) {
      this.parts.push(`<${tag}>${content.xhtml}</${tag}>\n`);
    }
  }

  /**
   * A container's children: blocks as blocks, and runs of inline content
   * between them as paragraphs. With `bareFirst`, the run before the first
   * block stays bare, as a list item's own words do.
   */
  content(parent: DomNode, bareFirst = false): void {
    let run: DomNode[] = [];
    let first = true;
    const flush = () => {
      if (run.length === 0) return;
      if (first && bareFirst) {
        const content = this.inline(run);
        if (content.text > 0 || content.images > 0) this.parts.push(content.xhtml);
      } else this.leaf("p", run);
      run = [];
    };
    for (const child of Array.from(parent.childNodes)) {
      if (
        isElement(child) &&
        !DROPPED.has(nameOf(child)) &&
        (BLOCK_ELEMENTS.has(nameOf(child)) || hasBlocks(child))
      ) {
        flush();
        first = false;
        this.block(child);
      } else run.push(child);
    }
    flush();
  }

  private list(list: DomElement, tag: "ul" | "ol"): void {
    const start = Number.parseInt(list.getAttribute("start") ?? "", 10);
    this.parts.push(
      tag === "ol" && Number.isInteger(start) && start !== 1
        ? `<ol start="${start}">\n`
        : `<${tag}>\n`,
    );
    for (const child of Array.from(list.childNodes)) {
      if (!isElement(child) || DROPPED.has(nameOf(child))) continue;
      this.parts.push("<li>");
      if (nameOf(child) === "li") this.content(child, true);
      else this.block(child);
      this.parts.push("</li>\n");
    }
    this.parts.push(`</${tag}>\n`);
  }

  private cell(cell: DomElement): string {
    const tag = nameOf(cell) === "th" ? "th" : "td";
    const spans = ["colspan", "rowspan"].flatMap((name) => {
      const value = Number.parseInt(cell.getAttribute(name) ?? "", 10);
      return Number.isInteger(value) && value > 1 && value < 1000 ? [` ${name}="${value}"`] : [];
    });
    return `<${tag}${spans.join("")}>${this.inline(Array.from(cell.childNodes), true).xhtml}</${tag}>`;
  }

  private table(table: DomElement): void {
    const rows: string[] = [];
    const walk = (element: DomElement) => {
      for (const child of Array.from(element.childNodes)) {
        if (!isElement(child)) continue;
        const name = nameOf(child);
        if (name === "tr") {
          const cells = Array.from(child.childNodes)
            .filter(isElement)
            .filter((cell) => nameOf(cell) === "td" || nameOf(cell) === "th");
          if (cells.length > 0)
            rows.push(`<tr>${cells.map((cell) => this.cell(cell)).join("")}</tr>`);
        } else if (name === "thead" || name === "tbody" || name === "tfoot") walk(child);
      }
    };
    walk(table);
    if (rows.length > 0) this.parts.push(`<table>\n${rows.join("\n")}\n</table>\n`);
  }

  block(element: DomElement): void {
    const name = nameOf(element);
    if (DROPPED.has(name)) return;
    const heading = /^h([1-6])$/.exec(name);
    if (heading) {
      const content = this.inline(Array.from(element.childNodes), true);
      if (content.text > 0) this.parts.push(`<${name}>${content.xhtml}</${name}>\n`);
    } else if (name === "pre") {
      const text = element.textContent ?? "";
      if (text.trim())
        this.parts.push(`<pre>${escapeXml(text.replace(/^\r?\n/, "").trimEnd())}</pre>\n`);
    } else if (name === "ul" || name === "ol") {
      this.list(element, name);
    } else if (name === "li") {
      this.parts.push("<ul>\n<li>");
      this.content(element, true);
      this.parts.push("</li>\n</ul>\n");
    } else if (name === "table") {
      // A table laid out to hold a page's parts is read as its content, not as data.
      if (element.querySelectorAll("table").length > 0) this.content(element);
      else this.table(element);
    } else if (name === "hr") {
      this.parts.push("<hr/>\n");
    } else if (LEAF_BLOCKS.has(name) && !hasBlocks(element)) {
      this.leaf(
        name === "address" || name === "caption" ? "p" : name,
        Array.from(element.childNodes),
      );
    } else if (KEPT_CONTAINERS.has(name)) {
      this.parts.push(`<${name}>\n`);
      this.content(element);
      this.parts.push(`</${name}>\n`);
    } else {
      this.content(element);
    }
  }
}

const meta = (document: DomDocument, key: string): string | null => {
  const element = document.querySelector(`meta[property="${key}"], meta[name="${key}"]`);
  return element?.getAttribute("content")?.trim() || null;
};

/** Letters and digits, a CJK character counted as three: what there is to read. */
function readable(text: string): number {
  let count = 0;
  for (const character of text.match(/[\p{L}\p{N}]/gu) ?? []) {
    count += CJK_CHARACTER.test(character) ? 3 : 1;
  }
  return count;
}

const clean = (text: string | null | undefined): string | null => {
  const cleaned = (text ?? "").replace(/\s+/g, " ").trim();
  return cleaned || null;
};

const SEPARATOR = "[-–—|_:·•»]";

/** The site's name a page's title gives beside the article's own: "Tea - Wikipedia". */
function siteInTitle(pageTitle: string | null, title: string): string | null {
  if (!pageTitle || pageTitle === title) return null;
  const rest = pageTitle.startsWith(title)
    ? new RegExp(`^\\s*${SEPARATOR}\\s*(.+)$`).exec(pageTitle.slice(title.length))
    : pageTitle.endsWith(title)
      ? new RegExp(`^(.+?)\\s*${SEPARATOR}\\s*$`).exec(pageTitle.slice(0, -title.length))
      : null;
  const site = rest?.[1]?.trim();
  return site && site.length <= 60 ? site : null;
}

/** A date as ISO 8601: as given when it is, its offset kept; else read as a date, if it reads. */
function publishedDate(value: string | null): string | null {
  if (!value || /^\d{4}-\d{2}/.test(value)) return value;
  const time = Date.parse(value);
  return Number.isNaN(time) ? null : new Date(time).toISOString();
}

/**
 * Makes a page's snapshot from its HTML. `url` is where the page was (after
 * redirects), for its links; `address` the one it is saved under. Throws
 * `WebPageError` for a page behind a sign-in, or one that is mostly not text.
 */
export function snapshotOf(html: string, url: string, address: string): Snapshot {
  const page = parseHTML(html).document;
  const document: DomDocument = page;
  const video = /^video\b/i.test(meta(document, "og:type") ?? "");
  const signIn = document.querySelector('input[type="password"]') !== null;
  const host = (URL.parse(url)?.hostname ?? "").replace(/^www\./, "");
  const fallbackTitle = clean(meta(document, "og:title")) ?? clean(document.title);
  // Edit links beside MediaWiki's headings.
  for (const element of Array.from(document.querySelectorAll(".mw-editsection"))) element.remove();

  const article = new Readability(page, { serializer: (node) => node }).parse();
  const refuse = (count: number): never => {
    if (video) throw new WebPageError("video", "The page is a video, with little text to read.");
    if (signIn && count < MIN_BEHIND_SIGN_IN) {
      throw new WebPageError("needs-sign-in", "The page asks for a password.");
    }
    throw new WebPageError("not-text", `The page has too little text to read (${count}).`);
  };
  const content: DomNode | null | undefined = article?.content;
  if (!article || !content) return refuse(0);

  let title = clean(article.title) ?? fallbackTitle ?? host ?? address;
  // A first heading that repeats the title is the title, without the site's name after it.
  const first = firstBlock(content);
  const heading = first && /^h[1-6]$/.test(nameOf(first)) ? clean(first.textContent) : null;
  if (first && heading && title.includes(heading)) {
    title = heading;
    first.remove();
  }
  const writer = new BlockWriter(url);
  writer.content(content);
  const siteName =
    clean(meta(document, "og:site_name")) ??
    siteInTitle(clean(document.title), title) ??
    clean(article.siteName) ??
    host;
  const author = clean(article.byline) ?? clean(meta(document, "author"));
  const published = publishedDate(clean(article.publishedTime));
  const language = clean(article.lang ?? document.documentElement?.getAttribute("lang"));
  const head = [
    '<meta charset="utf-8"/>',
    `<title>${escapeXml(title)}</title>`,
    `<meta name="incarnamind:address" content="${escapeXml(address)}"/>`,
    `<meta property="og:site_name" content="${escapeXml(siteName)}"/>`,
    ...(author ? [`<meta name="author" content="${escapeXml(author)}"/>`] : []),
    ...(published
      ? [`<meta property="article:published_time" content="${escapeXml(published)}"/>`]
      : []),
  ];
  const xhtml = [
    `<html xmlns="http://www.w3.org/1999/xhtml"${language ? ` lang="${escapeXml(language)}"` : ""}>`,
    `<head>\n${head.join("\n")}\n</head>`,
    `<body>\n<article>\n${writer.parts.join("")}</article>\n</body>`,
    "</html>\n",
  ].join("\n");

  const count = readable(
    readWebPage(xhtml)
      .blocks.map((block) => block.text)
      .join(" "),
  );
  if (count < MIN_READABLE || (video && count < MIN_BEHIND_SIGN_IN)) refuse(count);
  if (signIn && count < MIN_BEHIND_SIGN_IN) refuse(count);
  return { xhtml, title, siteName, author, published };
}
