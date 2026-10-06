/**
 * A small Markdown reader for the Document viewer. It understands the common
 * constructs (headings, paragraphs, lists, quotes, code, emphasis, links) and
 * shows everything else as text. It never produces HTML: the view renders this
 * tree with React, which escapes all text, so raw HTML in a file stays text.
 *
 * Every piece of text points back into the source (`start`/`end`), so a quote
 * found in the source can be highlighted where it is rendered.
 */

export type Inline =
  | { kind: "text"; start: number; end: number }
  | { kind: "code"; start: number; end: number }
  | { kind: "strong" | "em"; children: Inline[] }
  /** `href` is null for links the viewer won't follow (only http, https and mailto are). */
  | { kind: "link"; href: string | null; children: Inline[] };

export type Block =
  | { kind: "heading"; level: 1 | 2 | 3 | 4 | 5 | 6; inlines: Inline[] }
  | { kind: "paragraph"; inlines: Inline[] }
  | { kind: "quote"; inlines: Inline[] }
  | { kind: "list"; ordered: boolean; items: Inline[][] }
  /** Shown verbatim in a monospaced font: fenced code, and tables. */
  | { kind: "verbatim"; start: number; end: number }
  | { kind: "rule" };

interface Line {
  start: number;
  /** Excludes the line break. */
  end: number;
  text: string;
}

const FENCE = /^ {0,3}(`{3,}|~{3,})/;
const HEADING = /^ {0,3}(#{1,6})(?:[ \t]+|$)/;
const RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const QUOTE = /^ {0,3}> ?/;
const LIST_ITEM = /^ {0,3}(?:([-*+])|(\d{1,9})[.)])[ \t]+/;
const TABLE_ROW = /^ {0,3}\|/;
const BLANK = /^[ \t]*$/;
const CONTINUATION = /^(?: {2,}|\t)\S/;
const ESCAPABLE = /[!-/:-@[-`{-~]/;
const SAFE_LINK = /^(?:https?:|mailto:)/i;

function splitLines(source: string): Line[] {
  const lines: Line[] = [];
  let start = 0;
  while (start <= source.length) {
    const newline = source.indexOf("\n", start);
    const end = newline < 0 ? source.length : newline;
    lines.push({ start, end, text: source.slice(start, end) });
    if (newline < 0) break;
    start = newline + 1;
  }
  return lines;
}

const startsBlock = (text: string) =>
  FENCE.test(text) ||
  HEADING.test(text) ||
  RULE.test(text) ||
  QUOTE.test(text) ||
  LIST_ITEM.test(text) ||
  TABLE_ROW.test(text);

/** The offset where a line's content starts, after leading whitespace. */
const contentStart = (line: Line) => line.start + (line.text.length - line.text.trimStart().length);

/** Inlines for several source ranges that read as one run of text, joined by their line breaks. */
function joinLines(source: string, ranges: readonly { start: number; end: number }[]): Inline[] {
  const inlines: Inline[] = [];
  ranges.forEach((range, index) => {
    if (index > 0) {
      // The line break before this line: a real character of the source, shown as a space.
      const previous = ranges[index - 1] as { end: number };
      inlines.push({ kind: "text", start: previous.end, end: previous.end + 1 });
    }
    inlines.push(...parseInlines(source, range.start, range.end));
  });
  return inlines;
}

export function parseMarkdown(source: string): Block[] {
  const lines = splitLines(source);
  const blocks: Block[] = [];
  let index = 0;
  const lineAt = (at: number) => lines[at] as Line;

  while (index < lines.length) {
    const line = lineAt(index);
    const { text } = line;

    if (BLANK.test(text)) {
      index++;
      continue;
    }

    const fence = FENCE.exec(text);
    if (fence) {
      const marker = fence[1] as string;
      let close = index + 1;
      while (close < lines.length) {
        const candidate = lineAt(close).text.trim();
        if (candidate.startsWith(marker) && /^[`~]+$/.test(candidate)) break;
        close++;
      }
      const first = lines[index + 1];
      const last = lines[Math.min(close, lines.length) - 1];
      if (first && last && close > index + 1) {
        blocks.push({ kind: "verbatim", start: first.start, end: last.end });
      }
      index = close + 1;
      continue;
    }

    const heading = HEADING.exec(text);
    if (heading) {
      const level = (heading[1] as string).length as 1 | 2 | 3 | 4 | 5 | 6;
      let end = line.end;
      // A closing run of #s isn't part of the heading.
      const closing = /[ \t]+#+[ \t]*$|[ \t]+$/.exec(text.slice(heading[0].length));
      if (closing) end = line.start + heading[0].length + closing.index;
      blocks.push({
        kind: "heading",
        level,
        inlines: parseInlines(source, line.start + heading[0].length, end),
      });
      index++;
      continue;
    }

    if (RULE.test(text)) {
      blocks.push({ kind: "rule" });
      index++;
      continue;
    }

    if (QUOTE.test(text)) {
      const ranges: { start: number; end: number }[] = [];
      while (index < lines.length && QUOTE.test(lineAt(index).text)) {
        const quoted = lineAt(index);
        const prefix = (QUOTE.exec(quoted.text) as RegExpExecArray)[0].length;
        ranges.push({ start: quoted.start + prefix, end: quoted.end });
        index++;
      }
      blocks.push({ kind: "quote", inlines: joinLines(source, ranges) });
      continue;
    }

    const item = LIST_ITEM.exec(text);
    if (item) {
      const ordered = item[1] === undefined;
      const items: Inline[][] = [];
      while (index < lines.length) {
        const marker = LIST_ITEM.exec(lineAt(index).text);
        if (!marker || (marker[1] === undefined) !== ordered) break;
        const first = lineAt(index);
        const ranges = [{ start: first.start + marker[0].length, end: first.end }];
        index++;
        while (index < lines.length && CONTINUATION.test(lineAt(index).text)) {
          if (LIST_ITEM.test(lineAt(index).text.trimStart())) break;
          ranges.push({ start: contentStart(lineAt(index)), end: lineAt(index).end });
          index++;
        }
        items.push(joinLines(source, ranges));
      }
      blocks.push({ kind: "list", ordered, items });
      continue;
    }

    if (TABLE_ROW.test(text)) {
      const first = line;
      let last = line;
      while (index < lines.length && TABLE_ROW.test(lineAt(index).text)) {
        last = lineAt(index);
        index++;
      }
      blocks.push({ kind: "verbatim", start: first.start, end: last.end });
      continue;
    }

    const ranges = [{ start: contentStart(line), end: line.end }];
    index++;
    while (index < lines.length) {
      const next = lineAt(index);
      if (BLANK.test(next.text) || startsBlock(next.text)) break;
      ranges.push({ start: contentStart(next), end: next.end });
      index++;
    }
    blocks.push({ kind: "paragraph", inlines: joinLines(source, ranges) });
  }
  return blocks;
}

const isWordCharacter = (char: string | undefined) => !!char && /[\p{L}\p{N}]/u.test(char);
const isSpace = (char: string | undefined) => !char || /\s/u.test(char);

/** The closing delimiter for emphasis opened at `from`, or -1. */
function closingDelimiter(source: string, delimiter: string, from: number, end: number): number {
  if (isSpace(source[from])) return -1;
  let at = source.indexOf(delimiter, from);
  while (at >= 0 && at + delimiter.length <= end) {
    const single = delimiter.length === 1;
    const doubled = single && (source[at + 1] === delimiter || source[at - 1] === delimiter);
    const underscoreInsideWord =
      delimiter[0] === "_" && isWordCharacter(source[at + delimiter.length]);
    if (at > from && !isSpace(source[at - 1]) && !doubled && !underscoreInsideWord) return at;
    at = source.indexOf(delimiter, at + delimiter.length);
  }
  return -1;
}

/** Parses the inline Markdown in `source` from `start` to `end`. */
export function parseInlines(source: string, start: number, end: number): Inline[] {
  const inlines: Inline[] = [];
  let textStart = start;
  let at = start;
  const flush = (upTo: number) => {
    if (textStart < upTo) inlines.push({ kind: "text", start: textStart, end: upTo });
  };

  while (at < end) {
    const char = source[at] as string;

    if (char === "\\" && at + 1 < end && ESCAPABLE.test(source[at + 1] as string)) {
      flush(at);
      textStart = at + 1; // the escaped character starts the next run of text
      at += 2;
      continue;
    }

    if (char === "`") {
      let run = 1;
      while (source[at + run] === "`") run++;
      const fence = "`".repeat(run);
      let close = source.indexOf(fence, at + run);
      while (close >= 0 && source[close + run] === "`")
        close = source.indexOf(fence, close + run + 1);
      if (close >= 0 && close + run <= end) {
        flush(at);
        inlines.push({ kind: "code", start: at + run, end: close });
        at = close + run;
        textStart = at;
      } else {
        at += run;
      }
      continue;
    }

    if (char === "*" || char === "_") {
      const leftOk = char === "*" || !isWordCharacter(source[at - 1]);
      const strong = source[at + 1] === char;
      const delimiter = strong ? char + char : char;
      const close = leftOk ? closingDelimiter(source, delimiter, at + delimiter.length, end) : -1;
      if (close >= 0) {
        flush(at);
        inlines.push({
          kind: strong ? "strong" : "em",
          children: parseInlines(source, at + delimiter.length, close),
        });
        at = close + delimiter.length;
        textStart = at;
      } else {
        at += delimiter.length;
      }
      continue;
    }

    if (char === "[" || (char === "!" && source[at + 1] === "[")) {
      const image = char === "!";
      const open = image ? at + 1 : at;
      const closeBracket = source.indexOf("]", open + 1);
      const closeParen = closeBracket >= 0 ? source.indexOf(")", closeBracket) : -1;
      if (
        closeBracket >= 0 &&
        closeBracket < end &&
        source[closeBracket + 1] === "(" &&
        closeParen >= 0 &&
        closeParen < end
      ) {
        flush(at);
        const children = parseInlines(source, open + 1, closeBracket);
        // Images show their description: the viewer loads nothing from a Document.
        if (image) inlines.push(...children);
        else {
          const target =
            source
              .slice(closeBracket + 2, closeParen)
              .trim()
              .split(/\s+/)[0] ?? "";
          inlines.push({ kind: "link", href: SAFE_LINK.test(target) ? target : null, children });
        }
        at = closeParen + 1;
        textStart = at;
        continue;
      }
    }

    at++;
  }
  flush(end);
  return inlines;
}
