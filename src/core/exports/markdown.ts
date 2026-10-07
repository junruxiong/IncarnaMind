/**
 * A Mind as Markdown (CommonMark, with GitHub's tables and footnotes), for an
 * archive: Notes and Answers as Markdown, Questions marked as Questions, math
 * as `$…$` and `$$…$$`, and each Citation as its own footnote.
 */
import type { Block, Footnote, Inline, Marks } from "./model";

export interface MarkdownLabels {
  /** Marks a Question, e.g. "Question:". */
  question: string;
  /** Follows an unverified Citation's source in its footnote, e.g. "[unverified]". */
  unverified: string;
}

interface Context {
  /** Inside a table cell, where "|" must be escaped and lines can't break. */
  table: boolean;
}

export function renderMarkdown(
  title: string,
  blocks: readonly Block[],
  labels: MarkdownLabels,
): string {
  const footnotes: Footnote[] = [];
  const writer = new MarkdownWriter(labels, (footnote) => {
    footnotes.push(footnote);
    return footnotes.length;
  });
  const parts: string[] = [];
  if (title) parts.push(`# ${escapeText(title, false)}`);
  const body = writer.blocks(blocks, { table: false });
  if (body) parts.push(body);
  if (footnotes.length > 0) {
    parts.push(
      footnotes
        .map((footnote, index) => {
          const marker = footnote.unverified ? ` ${labels.unverified}` : "";
          return `[^${index + 1}]: ${escapeText(footnote.source, false)}${marker}`;
        })
        .join("\n"),
    );
  }
  return `${parts.join("\n\n")}\n`;
}

class MarkdownWriter {
  constructor(
    private readonly labels: MarkdownLabels,
    /** Numbers a Citation's footnote: 1, 2, 3… in the order they are written. */
    private readonly number: (footnote: Footnote) => number,
  ) {}

  blocks(blocks: readonly Block[], context: Context): string {
    return blocks
      .map((block) => this.block(block, context))
      .filter((text) => text.trim() !== "")
      .join("\n\n");
  }

  private block(block: Block, context: Context): string {
    switch (block.kind) {
      case "paragraph":
        return this.inline(block.content, context).split("\n").map(escapeLineStart).join("\n");
      case "heading": {
        const text = this.inline(block.content, context).replace(/\\\n/g, " ");
        return text ? `${"#".repeat(block.level)} ${text}` : "";
      }
      case "question": {
        const text = this.inline(block.content, context);
        return `**${this.labels.question}**${text ? ` ${text}` : ""}`;
      }
      case "code": {
        const longest = Math.max(0, ...(block.code.match(/`+/g) ?? []).map((run) => run.length));
        const fence = "`".repeat(Math.max(3, longest + 1));
        return `${fence}${block.language.replace(/[`\s]/g, "")}\n${block.code}\n${fence}`;
      }
      case "math":
        return `$$\n${block.latex}\n$$`;
      case "list":
        return this.list(block, context);
      case "quote":
        return this.blocks(block.content, context)
          .split("\n")
          .map((line) => (line ? `> ${line}` : ">"))
          .join("\n");
      case "rule":
        return "---";
      case "table":
        return this.table(block.rows);
      case "image":
        return image(block.image.src, block.image.alt, context);
    }
  }

  private list(block: Extract<Block, { kind: "list" }>, context: Context): string {
    let number = block.start;
    return block.items
      .map((item) => {
        const marker = block.ordered ? `${number++}.` : "-";
        const indent = " ".repeat(marker.length + 1);
        // A list right under the item's text stays tight; other Blocks are a paragraph apart.
        let body = "";
        for (const child of item) {
          const text = this.block(child, context);
          if (!text.trim()) continue;
          if (body) body += child.kind === "list" ? "\n" : "\n\n";
          body += text;
        }
        if (!body) return marker;
        return body
          .split("\n")
          .map((line, index) => (index === 0 ? `${marker} ${line}` : line ? indent + line : ""))
          .join("\n");
      })
      .join("\n");
  }

  /** A GitHub table: its first row is the header row, as GitHub requires one. */
  private table(rows: Extract<Block, { kind: "table" }>["rows"]): string {
    const columns = Math.max(
      1,
      ...rows.map((row) => row.reduce((count, cell) => count + cell.colspan, 0)),
    );
    const lines = rows.map((row) => {
      const cells: string[] = [];
      for (const cell of row) {
        cells.push(this.blocks(cell.content, { table: true }).replace(/\n+/g, "<br>"));
        for (let more = 1; more < cell.colspan; more++) cells.push("");
      }
      while (cells.length < columns) cells.push("");
      return `| ${cells.join(" | ")} |`;
    });
    if (lines.length === 0) return "";
    lines.splice(1, 0, `|${" --- |".repeat(columns)}`);
    return lines.join("\n");
  }

  /**
   * Inline content with its marks: **bold**, *italic*, ~~strike~~, `code` and
   * [links](…), properly nested; $math$; footnote references; hard breaks.
   */
  inline(content: readonly Inline[], context: Context): string {
    let out = "";
    /** Marks open at this point, outermost first. */
    let open: OpenMark[] = [];

    /** Closes the open marks from `from` up; whitespace before them moves after them, as CommonMark needs. */
    const closeFrom = (from: number) => {
      if (from >= open.length) return;
      const trailing = /[ \t]*$/.exec(out)?.[0] ?? "";
      out = out.slice(0, out.length - trailing.length);
      for (const mark of open.slice(from).reverse()) out += closingDelimiter(mark);
      out += trailing;
      open = open.slice(0, from);
    };

    for (const item of content) {
      if (item.kind === "break") {
        closeFrom(0);
        out += context.table ? "<br>" : "\\\n";
        continue;
      }
      if (item.kind !== "text") {
        out += this.atom(item, context);
        continue;
      }
      const wanted = wantedMarks(item.marks);
      const keep = open.findIndex((mark) => !wanted.some((want) => sameMark(want, mark)));
      if (keep >= 0) closeFrom(keep);
      let text = item.text;
      const toOpen = wanted.filter((want) => !open.some((mark) => sameMark(want, mark)));
      if (toOpen.length > 0 && text.trim() !== "") {
        // Whitespace stays outside the delimiters.
        const leading = /^[ \t]*/.exec(text)?.[0] ?? "";
        out += leading;
        text = text.slice(leading.length);
        for (const mark of toOpen) out += openingDelimiter(mark);
        open = [...open, ...toOpen];
      }
      out += item.marks.code ? codeSpan(text, context) : escapeText(text, context.table);
    }
    closeFrom(0);
    return out;
  }

  private atom(item: Exclude<Inline, { kind: "text" | "break" }>, context: Context): string {
    switch (item.kind) {
      case "math":
        return `$${item.latex}$`;
      case "citation":
        return `[^${this.number(item.footnote)}]`;
      case "image":
        return image(item.image.src, item.image.alt, context);
    }
  }
}

type OpenMark = { kind: "link"; href: string } | { kind: "bold" | "italic" | "strike" };

/** The marks Markdown can write, outermost first. Underline has no Markdown. */
function wantedMarks(marks: Marks): OpenMark[] {
  const wanted: OpenMark[] = [];
  if (marks.link) wanted.push({ kind: "link", href: marks.link });
  if (marks.bold) wanted.push({ kind: "bold" });
  if (marks.italic) wanted.push({ kind: "italic" });
  if (marks.strike) wanted.push({ kind: "strike" });
  return wanted;
}

const sameMark = (a: OpenMark, b: OpenMark) =>
  a.kind === b.kind && (a.kind !== "link" || (b.kind === "link" && a.href === b.href));

function openingDelimiter(mark: OpenMark): string {
  switch (mark.kind) {
    case "link":
      return "[";
    case "bold":
      return "**";
    case "italic":
      return "*";
    case "strike":
      return "~~";
  }
}

function closingDelimiter(mark: OpenMark): string {
  return mark.kind === "link" ? `](${destination(mark.href)})` : openingDelimiter(mark);
}

/** A link or image destination: spaces and parentheses would end it early. */
const destination = (url: string) => url.replace(/[ ()<>]/g, (char) => encodeURIComponent(char));

function image(src: string, alt: string, context: Context): string {
  return src ? `![${escapeText(alt, context.table)}](${destination(src)})` : "";
}

/** Escapes what Markdown would read as formatting: emphasis, code, links, math, HTML, entities. */
function escapeText(text: string, table: boolean): string {
  const escaped = text.replace(/[\\`*_~[\]$<]/g, "\\$&").replace(/&(?=#?\w+;)/g, "\\&");
  return table ? escaped.replace(/\|/g, "\\|") : escaped;
}

function codeSpan(text: string, context: Context): string {
  const longest = Math.max(0, ...(text.match(/`+/g) ?? []).map((run) => run.length));
  const fence = "`".repeat(longest + 1);
  const pad = text.startsWith("`") || text.endsWith("`") ? " " : "";
  const code = context.table ? text.replace(/\|/g, "\\|") : text;
  return `${fence}${pad}${code}${pad}${fence}`;
}

/**
 * Escapes what would start a heading, a quote, a list or a thematic break at
 * the start of a line of a paragraph.
 */
function escapeLineStart(line: string): string {
  const indent = /^ {0,3}/.exec(line)?.[0] ?? "";
  const rest = line.slice(indent.length);
  // "#" headings, ">" quotes, "-" and "+" list items.
  if (/^(#{1,6}|[+-])([ \t]|$)/.test(rest) || rest.startsWith(">")) return `${indent}\\${rest}`;
  // A line of "-" or "=" under a paragraph's text makes it a heading, or a thematic break.
  if (/^(-+|=+)[ \t]*$/.test(rest)) return `${indent}\\${rest}`;
  // "1." and "1)" ordered list items.
  const ordered = /^(\d{1,9})[.)]([ \t]|$)/.exec(rest);
  if (ordered?.[1]) return `${indent}${ordered[1]}\\${rest.slice(ordered[1].length)}`;
  return line;
}
