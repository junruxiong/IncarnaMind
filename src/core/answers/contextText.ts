/**
 * A Mind's Blocks, and a conversation's Answers, as Markdown for the model:
 * Question context, and the fingerprint of an Answer's text. The Blocks are
 * read by the one reader (../mindText); this only writes them.
 *
 * It shows bold, italic, strikethrough, inline code and links, math as `$…$`
 * and `$$…$$`, tables as pipe tables, Citations as plain references like
 * "[Attention Is All You Need, p. 3]", not as markers the model could reuse,
 * and the User's footnotes where their markers are, as Pandoc's inline notes:
 * "^[…]". Underline and highlight add nothing, and an image adds nothing (the
 * model can't see it). In a Mind imported from a Word file, someone's changes
 * still waiting are read as changes, with who made them and when:
 * "[deleted by Priya Nair, 2026-10-02: …]" (#734).
 */

import type * as Y from "yjs";
import { chosenSeries } from "../../shared/charts";
import { withFigureText } from "../../shared/figureMarkers";
import { citationLocation, englishLocation } from "../../shared/locations";
import { wordFileOf } from "../../shared/wordFiles";
import type { MarkedCitation, MarkedFigure } from "../api";
import {
  type Block,
  type Footnote,
  type Inline,
  type Marks,
  type ReadOptions,
  readBlock,
  readBlocks,
  type TableCell,
  tableLayout,
  theirChangesReading,
} from "../mindText";
import { changedContent, changeSpans, type ShownChange } from "../shownChanges";

/**
 * A Citation as a plain reference to its source, e.g. "[Attention Is All You
 * Need, p. 3]": how Question context shows the model Citations in Notes and
 * earlier Answers. One without a Document's name is not shown.
 */
function citationSource(attributes: Record<string, unknown>): Footnote {
  const name = attributes.documentName;
  if (typeof name !== "string" || name === "") return { source: "", unverified: false };
  const from = attributes.pageFrom;
  const to = attributes.pageTo;
  const location = citationLocation({
    location: attributes.location,
    pageFrom: typeof from === "number" ? from : null,
    pageTo: typeof to === "number" ? to : null,
  });
  return { source: location ? `${name}, ${englishLocation(location)}` : name, unverified: false };
}

/**
 * An Answer of a conversation as Markdown for the model, as an Answer in a Mind
 * reads: each Citation marker a plain reference to its source, each figure
 * marker its calculated figure's shown text, a marker with neither left out.
 */
export function answerMarkdown(
  text: string,
  citations: readonly MarkedCitation[],
  figures: readonly MarkedFigure[] = [],
): string {
  const byMarker = new Map(citations.map((citation) => [citation.marker, citation]));
  const figureByMarker = new Map(figures.map((figure) => [figure.marker, figure]));
  return withFigureText(text, (marker) => figureByMarker.get(marker) ?? null)
    .replace(/([ \t]?)\[\^(\d{1,4})\]/g, (_, space: string, marker: string) => {
      const citation = byMarker.get(Number(marker));
      const { source } = citationSource({
        documentName: citation?.documentName,
        pageFrom: citation?.pageFrom,
        pageTo: citation?.pageTo,
        location: citation?.location,
      });
      return source ? `${space}[${source}]` : "";
    })
    .trim();
}

/** Questions are read too, as the User's words; the space before a Citation stays. */
const READ: ReadOptions = {
  includeQuestions: true,
  footnote: citationSource,
  trimBeforeCitation: false,
};

/** How an element is read: in a Mind imported from a Word file, with someone's changes as changes. */
function readingOf(element: Y.XmlElement): ReadOptions {
  const { doc } = element;
  return doc && wordFileOf(doc) ? { ...READ, review: theirChangesReading(doc) } : READ;
}

/** A Block (or any element) as Markdown, e.g. for Question context. */
export function elementMarkdown(element: Y.XmlElement): string {
  return blocksMarkdown(readBlock(element, readingOf(element))).trim();
}

/** An element's content as Markdown: its child Blocks separated by blank lines. */
export function contentMarkdownOf(element: Y.XmlElement): string {
  return blocksMarkdown(readBlocks(element, readingOf(element))).trim();
}

function blocksMarkdown(blocks: readonly Block[]): string {
  return blocks
    .map(blockMarkdown)
    .filter((text) => text.trim() !== "")
    .join("\n\n");
}

function blockMarkdown(block: Block): string {
  switch (block.kind) {
    case "paragraph":
    case "question":
      return inlineMarkdown(changedContent(block));
    case "heading":
      return `${"#".repeat(block.level)} ${inlineMarkdown(changedContent(block))}`;
    case "code": {
      const fence = block.code.includes("```") ? "````" : "```";
      return `${fence}${block.language}\n${block.code}\n${fence}`;
    }
    case "math":
      return `$$\n${block.latex}\n$$`;
    case "rule":
      return "---";
    case "list":
      return listMarkdown(block);
    case "quote":
      return blocksMarkdown(block.content)
        .split("\n")
        .map((line) => (line ? `> ${line}` : ">"))
        .join("\n");
    case "table":
      return tableMarkdown(block.rows);
    case "chart":
      return chartMarkdown(block);
    case "image":
    // Only an export writes a Bibliography; a Mind's own Blocks never hold one.
    case "bibliography":
      return "";
  }
}

/**
 * A table as a pipe table, as models write them: its first row as the heading
 * row, a cell's lines joined by "<br>", and a cell over several columns or
 * rows in the first of them.
 */
function tableMarkdown(rows: readonly TableCell[][]): string {
  const layout = tableLayout(rows);
  const lines = layout.rows.map((row) => {
    const cells: string[] = [];
    for (const { cell, continued } of row) {
      const text = cell && !continued ? blocksMarkdown(cell.content) : "";
      // A backslash before a pipe is escaped first, or it would escape the pipe's own escape.
      cells.push(
        text
          .replace(/\\(?=\|)/g, "\\\\")
          .replace(/\|/g, "\\|")
          .replace(/\n+/g, "<br>"),
      );
      for (let more = 1; more < (cell?.colspan ?? 1); more++) cells.push("");
    }
    while (cells.length < layout.columns) cells.push("");
    return `| ${cells.join(" | ")} |`;
  });
  if (lines.length === 0) return "";
  lines.splice(1, 0, `|${" --- |".repeat(layout.columns)}`);
  return lines.join("\n");
}

function chartMarkdown(block: Extract<Block, { kind: "chart" }>): string {
  const { chart } = block;
  const drawn = chosenSeries(chart);
  const marks: Marks = {
    bold: false,
    italic: false,
    strike: false,
    underline: false,
    highlight: false,
    code: false,
    link: null,
  };
  const cell = (content: Inline[], header = false): TableCell => ({
    header,
    colspan: 1,
    rowspan: 1,
    align: null,
    content: [{ kind: "paragraph", content }],
  });
  const words = (text: string): Inline[] => (text ? [{ kind: "text", text, marks }] : []);
  const rows: TableCell[][] = [
    [
      cell(words(chart.data.labelHeading), true),
      ...drawn.map((series) => cell(words(chart.data.series[series] ?? ""), true)),
    ],
    ...chart.data.rows.map((row) => [
      cell(words(row.label)),
      ...drawn.map((series) => {
        const value = row.values[series];
        if (!value) return cell([]);
        const source = value.citation
          ? citationSource(value.citation as unknown as Record<string, unknown>)
          : null;
        return cell([
          ...words(value.shown),
          ...(source?.source
            ? ([
                { kind: "text", text: " ", marks },
                { kind: "citation", footnote: source, work: blankWork },
              ] as Inline[])
            : []),
        ]);
      }),
    ]),
  ];
  const cut = chart.data.cutFrom
    ? ` (its first ${chart.data.rows.length} rows of ${chart.data.cutFrom})`
    : "";
  return `Chart, ${chart.kind}${chart.title ? `: ${chart.title}` : ""}${cut}\n\n${tableMarkdown(rows)}`;
}

/** Question context writes a Citation's source, never its work. */
const blankWork = { documentId: null, documentName: null, location: null };

function listMarkdown(list: Extract<Block, { kind: "list" }>): string {
  let number = list.start;
  return list.items
    .map((item) => {
      const marker = list.ordered ? `${number++}.` : "-";
      const indent = " ".repeat(marker.length + 1);
      return blocksMarkdown(item)
        .split("\n")
        .map((line, index) => (index === 0 ? `${marker} ${line}` : line ? `${indent}${line}` : ""))
        .join("\n");
    })
    .join("\n");
}

function inlineMarkdown(content: readonly Inline[]): string {
  return changeSpans(content)
    .map(({ change, content: part }) => {
      const text = part.map(inlineText).join("");
      return change ? `[${changeWords(change)}: ${text}]` : text;
    })
    .join("");
}

/** Who made a change and when, for the model: "deleted by Priya Nair, 2026-10-02". */
function changeWords({ kind, pending }: ShownChange): string {
  const day = /^\d{4}-\d{2}-\d{2}/.exec(pending.at ?? "")?.[0];
  return `${kind === "insert" ? "inserted" : "deleted"} by ${pending.by || "someone"}${day ? `, ${day}` : ""}`;
}

function inlineText(inline: Inline): string {
  switch (inline.kind) {
    case "text":
      return markedText(inline.text, inline.marks);
    case "math":
      return `$${inline.latex}$`;
    case "break":
      return "\n";
    case "citation":
      return inline.footnote.source ? `[${inline.footnote.source}]` : "";
    case "footnote":
      return `^[${inlineMarkdown(inline.content)}]`;
    case "cite":
      return inline.runs.map((run) => run.text).join("");
    case "image":
      return "";
  }
}

function markedText(text: string, marks: Marks): string {
  if (marks.code) return `\`${text}\``;
  let marked = text;
  if (marks.bold) marked = `**${marked}**`;
  if (marks.italic) marked = `*${marked}*`;
  if (marks.strike) marked = `~~${marked}~~`;
  if (marks.link) marked = `[${marked}](${marks.link})`;
  return marked;
}
