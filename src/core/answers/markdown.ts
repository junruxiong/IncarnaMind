/**
 * Turns the Markdown a chat model writes into the node types of Notes:
 * paragraphs, headings 1-6, bullet and ordered lists (nested), block quotes,
 * horizontal rules, fenced code blocks with their language, and math as
 * KaTeX formulas (`$…$` or `\(…\)` inline, `$$…$$` or `\[…\]` as a block),
 * with bold, italic, strikethrough, inline code and links. Anything else
 * (tables, HTML) stays as text. A line break inside a paragraph is kept.
 *
 * While an Answer streams, the text is cut off at an arbitrary point. With
 * `streaming`, what would still change how it looks is held back (a half-typed
 * list marker or fence, an unclosed formula or link) and what is open is shown
 * closed (bold, inline code, a code block), so the Answer grows without
 * flickering between raw Markdown and formatting.
 */
import type { MarkJSON, NodeJSON } from "./blocks";

export interface MarkdownOptions {
  /** The text may continue: hold back, or close, what is unfinished at its end. */
  streaming?: boolean;
}

/** Markdown as Blocks; at least one (an empty paragraph). */
export function markdownToBlocks(markdown: string, options: MarkdownOptions = {}): NodeJSON[] {
  const streaming = options.streaming ?? false;
  let text = markdown.replace(/\r\n?/g, "\n");
  if (streaming) {
    // A last line that is only the start of a marker ("1", "-", "##", "``", "$") could still become anything.
    const lineStart = text.lastIndexOf("\n") + 1;
    if (PARTIAL_MARKER.test(text.slice(lineStart))) text = text.slice(0, lineStart);
  }
  const blocks = parseBlocks(text.split("\n"), streaming);
  return blocks.length > 0 ? blocks : [{ type: "paragraph" }];
}

const PARTIAL_MARKER = /^[ \t]*(?:[-*+_]{1,2}|\d{1,9}[.)]?|#{1,6}|>|[`~]{1,2}|\$|\\)$/;
const FENCE = /^( {0,3})(`{3,}|~{3,})[ \t]*([^\s`]*)[^`]*$/;
const MATH_BLOCK = /^ {0,3}(\$\$|\\\[)(.*)$/;
const HEADING = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
const RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const QUOTE = /^ {0,3}>/;
const LIST_ITEM = /^( {0,3})([-*+]|\d{1,9}[.)])(?:([ \t]+)(.*))?$/;
/** Holding back an unclosed formula or link stops after this many characters: it was plain text after all. */
const MAX_PENDING = 300;

const isBlank = (line: string) => line.trim() === "";

function indentOf(line: string): number {
  let width = 0;
  for (const char of line) {
    if (char === " ") width++;
    else if (char === "\t") width += 4 - (width % 4);
    else break;
  }
  return width;
}

/** Removes up to `width` columns of indentation. */
function dedent(line: string, width: number): string {
  let removed = 0;
  let index = 0;
  while (index < line.length && removed < width) {
    const char = line[index];
    if (char === " ") removed++;
    else if (char === "\t") removed += 4 - (removed % 4);
    else break;
    index++;
  }
  return line.slice(index);
}

function startsBlock(line: string): boolean {
  return (
    FENCE.test(line) ||
    MATH_BLOCK.test(line) ||
    HEADING.test(line) ||
    RULE.test(line) ||
    QUOTE.test(line) ||
    LIST_ITEM.test(line)
  );
}

/** `open`: the last line may continue, so what ends there is unfinished. */
function parseBlocks(lines: readonly string[], open: boolean): NodeJSON[] {
  const blocks: NodeJSON[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i] as string;
    if (isBlank(line)) {
      i++;
      continue;
    }
    const isLast = i === lines.length - 1;

    const fence = FENCE.exec(line);
    if (fence) {
      // The fence line itself may still be getting its language.
      if (open && isLast) break;
      const [, indent = "", marker = "```", language = ""] = fence;
      const closer = new RegExp(
        `^ {0,3}${marker[0] === "`" ? "`" : "~"}{${marker.length},}[ \\t]*$`,
      );
      const code: string[] = [];
      i++;
      while (i < lines.length && !closer.test(lines[i] as string)) {
        code.push(dedent(lines[i] as string, indent.length));
        i++;
      }
      i++; // The closing fence, if any: an unclosed block runs to the end.
      blocks.push(codeBlock(language, code.join("\n")));
      continue;
    }

    const math = MATH_BLOCK.exec(line);
    if (math) {
      const [, opener = "$$", rest = ""] = math;
      const closer = opener === "$$" ? "$$" : "\\]";
      const body: string[] = [];
      let closed = false;
      let after = "";
      let remaining = rest;
      for (;;) {
        const end = remaining.indexOf(closer);
        if (end >= 0) {
          body.push(remaining.slice(0, end));
          after = remaining.slice(end + closer.length);
          closed = true;
          i++;
          break;
        }
        body.push(remaining);
        i++;
        if (i >= lines.length) break;
        remaining = lines[i] as string;
      }
      // An unclosed formula would show as a KaTeX error until it's complete.
      if (!closed && open) break;
      const latex = body.join("\n").trim();
      if (latex) blocks.push({ type: "blockMath", attrs: { latex } });
      if (after.trim()) blocks.push(paragraph(after.trim(), open && i >= lines.length));
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      const [, hashes = "#", title = ""] = heading;
      blocks.push({
        type: "heading",
        attrs: { level: hashes.length },
        content: parseInline(title, open && isLast),
      });
      i++;
      continue;
    }

    if (RULE.test(line)) {
      blocks.push({ type: "horizontalRule" });
      i++;
      continue;
    }

    if (QUOTE.test(line)) {
      const quoted: string[] = [];
      while (i < lines.length) {
        const current = lines[i] as string;
        if (QUOTE.test(current)) quoted.push(current.replace(/^ {0,3}> ?/, ""));
        else if (!isBlank(current) && !startsBlock(current) && quoted.length > 0)
          quoted.push(current);
        else break;
        i++;
      }
      const content = parseBlocks(quoted, open && i >= lines.length);
      blocks.push({
        type: "blockquote",
        content: content.length ? content : [{ type: "paragraph" }],
      });
      continue;
    }

    if (LIST_ITEM.test(line)) {
      const list = parseList(lines, i, open);
      blocks.push(list.block);
      i = list.next;
      continue;
    }

    const text: string[] = [line];
    i++;
    while (i < lines.length && !isBlank(lines[i] as string) && !startsBlock(lines[i] as string)) {
      text.push(lines[i] as string);
      i++;
    }
    blocks.push(paragraph(text.join("\n"), open && i >= lines.length));
  }
  return blocks;
}

/** Fence names models use for languages that highlight.js knows by another name. */
const LANGUAGE_ALIASES: Readonly<Record<string, string>> = {
  js: "javascript",
  jsx: "javascript",
  mjs: "javascript",
  node: "javascript",
  ts: "typescript",
  tsx: "typescript",
  py: "python",
  python3: "python",
  sh: "bash",
  shell: "bash",
  zsh: "bash",
  yml: "yaml",
  md: "markdown",
  c: "cpp",
  "c++": "cpp",
  rs: "rust",
  golang: "go",
  html: "xml",
};

function codeBlock(language: string, code: string): NodeJSON {
  const lower = language.toLowerCase();
  const name = Object.hasOwn(LANGUAGE_ALIASES, lower) ? (LANGUAGE_ALIASES[lower] as string) : lower;
  return {
    type: "codeBlock",
    attrs: { language: name || null },
    content: code ? [{ type: "text", text: code }] : [],
  };
}

function paragraph(text: string, open: boolean): NodeJSON {
  return { type: "paragraph", content: parseInline(text.trim(), open) };
}

interface ListMarker {
  indent: number;
  marker: string;
  /** Where the item's content starts, in columns. */
  contentIndent: number;
  first: string;
}

function listMarker(line: string): ListMarker | null {
  if (RULE.test(line)) return null;
  const match = LIST_ITEM.exec(line);
  if (!match) return null;
  const [, indent = "", marker = "-", spacing = "", first = ""] = match;
  const gap = spacing.length === 0 || spacing.length > 4 ? 1 : spacing.length;
  return {
    indent: indent.length,
    marker,
    contentIndent: indent.length + marker.length + gap,
    first: spacing.length > 4 ? `${spacing.slice(1)}${first}` : first,
  };
}

const isOrdered = (marker: string) => /\d/.test(marker);
/** Items continue a list when they use the same kind of marker: the same bullet, or the same number delimiter. */
const sameList = (a: string, b: string) =>
  isOrdered(a) ? isOrdered(b) && a.slice(-1) === b.slice(-1) : a === b;

function parseList(
  lines: readonly string[],
  start: number,
  open: boolean,
): { block: NodeJSON; next: number } {
  const firstMarker = listMarker(lines[start] as string) as ListMarker;
  const ordered = isOrdered(firstMarker.marker);
  const items: NodeJSON[] = [];
  let i = start;

  while (i < lines.length) {
    const item = listMarker(lines[i] as string);
    if (!item || !sameList(firstMarker.marker, item.marker)) break;
    const itemLines = [item.first];
    i++;
    while (i < lines.length) {
      const line = lines[i] as string;
      if (isBlank(line)) {
        let next = i;
        while (next < lines.length && isBlank(lines[next] as string)) next++;
        if (next < lines.length && indentOf(lines[next] as string) >= item.contentIndent) {
          for (; i < next; i++) itemLines.push("");
          continue;
        }
        break;
      }
      if (indentOf(line) >= item.contentIndent) {
        itemLines.push(dedent(line, item.contentIndent));
      } else if (!startsBlock(line) && !isBlank(itemLines.at(-1) ?? "")) {
        // A lazy continuation of the item's paragraph.
        itemLines.push(line.trimStart());
      } else {
        break;
      }
      i++;
    }
    const content = parseBlocks(itemLines, open && i >= lines.length);
    if (content[0]?.type !== "paragraph") content.unshift({ type: "paragraph" });
    items.push({ type: "listItem", content });

    // Blank lines between items keep the list going.
    let next = i;
    while (next < lines.length && isBlank(lines[next] as string)) next++;
    const following = next < lines.length ? listMarker(lines[next] as string) : null;
    if (
      following &&
      following.indent <= firstMarker.indent + 3 &&
      sameList(firstMarker.marker, following.marker)
    ) {
      i = next;
    } else {
      break;
    }
  }

  const block: NodeJSON = ordered
    ? {
        type: "orderedList",
        attrs: { start: Number.parseInt(firstMarker.marker, 10) || 1 },
        content: items,
      }
    : { type: "bulletList", content: items };
  return { block, next: i };
}

// ---------------------------------------------------------------------------
// Inline content

type Token =
  | { kind: "text"; text: string; marks: MarkJSON[] }
  | { kind: "math"; latex: string }
  | { kind: "break" }
  | {
      kind: "delimiter";
      char: "*" | "_" | "~";
      count: number;
      canOpen: boolean;
      canClose: boolean;
      marks: MarkJSON[];
    };

const ASCII_PUNCTUATION = /[!-/:-@[-`{-~]/;
const isWhitespace = (char: string | undefined) => char === undefined || /\s/.test(char);
const isPunctuation = (char: string | undefined) =>
  char !== undefined && /[\p{P}\p{S}]/u.test(char);

const CODE: MarkJSON = { type: "code", attrs: {} };

/** The link mark with every attribute of the editor's Link, so the editor sees it unchanged. */
const linkMark = (href: string): MarkJSON => ({
  type: "link",
  attrs: {
    href,
    target: "_blank",
    rel: "noopener noreferrer nofollow",
    class: null,
    title: null,
  },
});

const isSafeHref = (href: string) => /^(https?:|mailto:)/i.test(href);

/** Inline Markdown as text with marks, formulas and line breaks. */
export function parseInline(source: string, open: boolean): NodeJSON[] {
  const tokens = tokenize(source, open);
  matchDelimiters(tokens, open);
  return toNodes(tokens);
}

function runLength(source: string, index: number, char: string): number {
  let end = index;
  while (source[end] === char) end++;
  return end - index;
}

/** The index of the next run of exactly `length` backticks, or -1. */
function closingBackticks(source: string, from: number, length: number): number {
  let index = source.indexOf("`", from);
  while (index >= 0) {
    const run = runLength(source, index, "`");
    if (run === length) return index;
    index = source.indexOf("`", index + run);
  }
  return -1;
}

/** A closing `$`: not after whitespace, and not followed by a digit (so "$5 and $10" stays text). */
function closingDollar(source: string, from: number): number {
  for (let index = from; index < source.length; index++) {
    const char = source[index];
    if (char === "\\") {
      index++;
      continue;
    }
    if (char === "\n" && source[index + 1] === "\n") return -1;
    if (
      char === "$" &&
      index > from &&
      !isWhitespace(source[index - 1]) &&
      !/\d/.test(source[index + 1] ?? "")
    ) {
      return index;
    }
  }
  return -1;
}

function closingBracket(source: string, from: number): number {
  let depth = 0;
  for (let index = from; index < source.length; index++) {
    const char = source[index];
    if (char === "\\") index++;
    else if (char === "[") depth++;
    else if (char === "]") {
      depth--;
      if (depth === 0) return index;
    }
  }
  return -1;
}

function tokenize(source: string, open: boolean): Token[] {
  const tokens: Token[] = [];
  let text = "";
  const flush = () => {
    if (text) tokens.push({ kind: "text", text, marks: [] });
    text = "";
  };
  /** The rest is unfinished: show nothing of it until it is. */
  const holdBack = (from: number) => open && source.length - from <= MAX_PENDING;

  let i = 0;
  while (i < source.length) {
    const char = source[i] as string;
    const next = source[i + 1];

    if (char === "\\") {
      if (next === "(") {
        const end = source.indexOf("\\)", i + 2);
        if (end >= 0) {
          flush();
          const latex = source.slice(i + 2, end).trim();
          if (latex) tokens.push({ kind: "math", latex });
          i = end + 2;
          continue;
        }
        if (holdBack(i)) break;
      }
      if (next === "\n") {
        flush();
        tokens.push({ kind: "break" });
        i += 2;
        continue;
      }
      if (next !== undefined && ASCII_PUNCTUATION.test(next)) {
        text += next;
        i += 2;
        continue;
      }
      if (next === undefined && open) break;
      text += char;
      i++;
      continue;
    }

    if (char === "\n") {
      text = text.replace(/[ \t]+$/, "");
      flush();
      tokens.push({ kind: "break" });
      i++;
      while (source[i] === " " || source[i] === "\t") i++;
      continue;
    }

    if (char === "`") {
      const run = runLength(source, i, "`");
      const close = closingBackticks(source, i + run, run);
      if (close >= 0 || open) {
        flush();
        let code = source.slice(i + run, close >= 0 ? close : undefined).replace(/\n/g, " ");
        if (code.length > 2 && code.startsWith(" ") && code.endsWith(" ")) code = code.slice(1, -1);
        if (code) tokens.push({ kind: "text", text: code, marks: [CODE] });
        if (close < 0) break; // Still being written: shown as code already.
        i = close + run;
        continue;
      }
      text += source.slice(i, i + run);
      i += run;
      continue;
    }

    if (char === "$") {
      const run = runLength(source, i, "$");
      if (run === 2) {
        const close = source.indexOf("$$", i + 2);
        if (close > i + 2) {
          flush();
          tokens.push({ kind: "math", latex: source.slice(i + 2, close).trim() });
          i = close + 2;
          continue;
        }
        if (close < 0 && holdBack(i)) break;
      } else if (run === 1 && !isWhitespace(next)) {
        const close = closingDollar(source, i + 1);
        if (close >= 0) {
          flush();
          tokens.push({ kind: "math", latex: source.slice(i + 1, close) });
          i = close + 1;
          continue;
        }
        if (!/\d/.test(next ?? "") && holdBack(i)) break;
      } else if (run === 1 && next === undefined && open) {
        break;
      }
      text += source.slice(i, i + run);
      i += run;
      continue;
    }

    if (char === "!" && next === "[") {
      // An image: kept as a link to it.
      i++;
      continue;
    }

    if (char === "[") {
      const close = closingBracket(source, i);
      if (close < 0) {
        if (holdBack(i) && !source.includes("\n", i)) break;
      } else if (source[close + 1] === "(") {
        const end = source.indexOf(")", close + 2);
        if (end >= 0) {
          flush();
          const href = (
            source
              .slice(close + 2, end)
              .trim()
              .split(/\s+/)[0] ?? ""
          ).replace(/^<|>$/g, "");
          const label = parseInlineTokens(source.slice(i + 1, close));
          if (isSafeHref(href)) for (const token of label) addMark(token, linkMark(href));
          tokens.push(...label);
          i = end + 1;
          continue;
        }
        if (holdBack(i)) break;
      } else if (close === source.length - 1 && open) {
        break; // "[label]" may still get its "(url)".
      }
      text += char;
      i++;
      continue;
    }

    if (char === "<") {
      const autolink = /^<((?:https?:\/\/|mailto:)[^\s<>]+)>/i.exec(source.slice(i));
      if (autolink?.[1]) {
        flush();
        tokens.push({ kind: "text", text: autolink[1], marks: [linkMark(autolink[1])] });
        i += autolink[0].length;
        continue;
      }
    }

    if (char === "*" || char === "_" || char === "~") {
      const run = runLength(source, i, char);
      if (open && i + run === source.length) break;
      if (char === "~" && run !== 2) {
        text += source.slice(i, i + run);
        i += run;
        continue;
      }
      const before = i > 0 ? source[i - 1] : undefined;
      const after = source[i + run];
      const leftFlanking =
        !isWhitespace(after) &&
        (!isPunctuation(after) || isWhitespace(before) || isPunctuation(before));
      const rightFlanking =
        !isWhitespace(before) &&
        (!isPunctuation(before) || isWhitespace(after) || isPunctuation(after));
      flush();
      tokens.push({
        kind: "delimiter",
        char,
        count: run,
        // "_" doesn't emphasise inside words (snake_case).
        canOpen:
          char === "_" ? leftFlanking && (!rightFlanking || isPunctuation(before)) : leftFlanking,
        canClose:
          char === "_" ? rightFlanking && (!leftFlanking || isPunctuation(after)) : rightFlanking,
        marks: [],
      });
      i += run;
      continue;
    }

    text += char;
    i++;
  }
  flush();
  return tokens;
}

/** A closed piece of inline Markdown (e.g. a link's label), with its emphasis matched. */
function parseInlineTokens(source: string): Token[] {
  const tokens = tokenize(source, false);
  matchDelimiters(tokens, false);
  return tokens;
}

function addMark(token: Token, mark: MarkJSON): void {
  if (token.kind !== "text" && token.kind !== "delimiter") return;
  // Code excludes every other mark.
  if (token.marks.some((each) => each.type === "code")) return;
  if (token.marks.some((each) => each.type === mark.type)) return;
  token.marks.push(mark);
}

const markFor = (char: string, count: number): MarkJSON => ({
  type: char === "~" ? "strike" : count >= 2 ? "bold" : "italic",
  attrs: {},
});

/** Pairs emphasis delimiters into bold, italic and strikethrough. With `open`, unclosed ones close at the end. */
function matchDelimiters(tokens: Token[], open: boolean): void {
  for (let closerAt = 0; closerAt < tokens.length; closerAt++) {
    const closer = tokens[closerAt] as Token;
    if (closer.kind !== "delimiter" || !closer.canClose || closer.count === 0) continue;
    for (let openerAt = closerAt - 1; openerAt >= 0; openerAt--) {
      const opener = tokens[openerAt] as Token;
      if (
        opener.kind !== "delimiter" ||
        opener.char !== closer.char ||
        !opener.canOpen ||
        opener.count === 0
      ) {
        continue;
      }
      if (closer.char === "~" && (opener.count < 2 || closer.count < 2)) continue;
      const used = closer.char === "~" || (opener.count >= 2 && closer.count >= 2) ? 2 : 1;
      const mark = markFor(closer.char, used);
      for (let k = openerAt + 1; k < closerAt; k++) addMark(tokens[k] as Token, mark);
      opener.count -= used;
      closer.count -= used;
      if (closer.count > 0) closerAt--; // Match what is left of the closer too.
      break;
    }
  }
  if (!open) return;
  for (let at = 0; at < tokens.length; at++) {
    const opener = tokens[at] as Token;
    if (opener.kind !== "delimiter" || !opener.canOpen || opener.count === 0) continue;
    if (opener.char === "~" && opener.count < 2) continue;
    const mark = markFor(opener.char, opener.count);
    for (let k = at + 1; k < tokens.length; k++) addMark(tokens[k] as Token, mark);
    opener.count = 0;
  }
}

function sameMarks(a: readonly MarkJSON[], b: readonly MarkJSON[]): boolean {
  if (a.length !== b.length) return false;
  const types = new Set(a.map((mark) => JSON.stringify(mark)));
  return b.every((mark) => types.has(JSON.stringify(mark)));
}

function toNodes(tokens: readonly Token[]): NodeJSON[] {
  const nodes: NodeJSON[] = [];
  const pushText = (text: string, marks: readonly MarkJSON[]) => {
    if (!text) return;
    const last = nodes.at(-1);
    if (last?.type === "text" && sameMarks(last.marks ?? [], marks)) {
      last.text = `${last.text ?? ""}${text}`;
      return;
    }
    nodes.push(
      marks.length > 0 ? { type: "text", text, marks: [...marks] } : { type: "text", text },
    );
  };
  for (const token of tokens) {
    if (token.kind === "text") pushText(token.text, token.marks);
    else if (token.kind === "delimiter") pushText(token.char.repeat(token.count), token.marks);
    else if (token.kind === "math") {
      if (token.latex) nodes.push({ type: "inlineMath", attrs: { latex: token.latex } });
    } else nodes.push({ type: "hardBreak" });
  }
  while (nodes[0]?.type === "hardBreak") nodes.shift();
  while (nodes.at(-1)?.type === "hardBreak") nodes.pop();
  return nodes;
}
