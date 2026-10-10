/**
 * A Mind's Blocks as Markdown for the model: Question context, and the
 * fingerprint of an Answer's text. The Blocks are read by the one reader
 * (../mindText); this only writes them.
 *
 * It shows bold, italic, strikethrough, inline code and links, math as `$…$`
 * and `$$…$$`, and Citations as plain references like "[Attention Is All You
 * Need, p. 3]", not as markers the model could reuse. Underline and highlight
 * add nothing, and an image adds nothing (the model can't see it).
 */

import type * as Y from "yjs";
import { citationLocation, englishLocation } from "../../shared/locations";
import {
  type Block,
  type Footnote,
  type Inline,
  type Marks,
  type ReadOptions,
  readBlock,
  readBlocks,
} from "../mindText";

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

/** Questions are read too, as the User's words; the space before a Citation stays. */
const READ: ReadOptions = {
  includeQuestions: true,
  footnote: citationSource,
  trimBeforeCitation: false,
};

/** A Block (or any element) as Markdown, e.g. for Question context. */
export function elementMarkdown(element: Y.XmlElement): string {
  return blocksMarkdown(readBlock(element, READ)).trim();
}

/** An element's content as Markdown: its child Blocks separated by blank lines. */
export function contentMarkdownOf(element: Y.XmlElement): string {
  return blocksMarkdown(readBlocks(element, READ)).trim();
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
      return inlineMarkdown(block.content);
    case "heading":
      return `${"#".repeat(block.level)} ${inlineMarkdown(block.content)}`;
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
      return blocksMarkdown(block.rows.flatMap((row) => row.flatMap((cell) => cell.content)));
    case "image":
      return "";
  }
}

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
  return content.map(inlineText).join("");
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
