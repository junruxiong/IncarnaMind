/**
 * Writing Proposals into a Mind's Yjs document (see `PROPOSAL_INSERT_MARK` in
 * ../api), in the shape the editor's binding reads (see ../answers/blocks):
 *
 * - `readSelection` reads a stretch of a Mind's text, from one paragraph or
 *   heading to another (`TextPlace`), as the model will read it: Citations
 *   and the User's footnotes as numbered markers ("[1]", "[^1]"), formulas as
 *   $…$, paragraphs apart.
 * - `proposeRewrite` compares that stretch with the model's new text, word by
 *   word (./diff), and marks each place that changed as a Proposal of its
 *   own: the words it removes stay, under `proposalDelete`, and the words it
 *   adds go after them, under `proposalInsert`. Paragraphs the new text adds
 *   or drops are Proposals on whole Blocks.
 * - `proposeAfter` adds new paragraphs under the stretch, each a Proposal.
 * - `proposeBlocks` writes new content for a whole element (an Answer
 *   written again) over what it holds: Blocks kept as they are, paragraphs
 *   compared word by word, and other Blocks added or removed whole.
 * - `takeBackForRegenerate` rejects a Proposal and says where the text it
 *   covered is now, to make it again from that text.
 *
 * Nothing in the Mind is changed but by marks, attributes and the new words,
 * which are AI text (`GENERATED_MARK`) and stay so once accepted: the User's
 * text stays where and what it was until they accept. Each Proposal records
 * the text it changes (`ProposalRef.target`), so that it goes out of date
 * once that text changes.
 */
import * as Y from "yjs";
import { INLINE_CODE_MARK } from "../../shared/aiText";
import {
  INLINE_NODES,
  NODE_CHARACTER,
  readProposalAttribute,
  readProposalRef,
  resolveProposals,
  targetHash,
  targetText,
} from "../../shared/proposals";
import { createElement, formattingOf, type NodeJSON } from "../answers/blocks";
import {
  BLOCK_ID_ATTRIBUTE,
  CITATION_NODE,
  FOOTNOTE_NODE,
  GENERATED_MARK,
  PROPOSAL_ATTRIBUTE,
  PROPOSAL_DELETE_MARK,
  PROPOSAL_INSERT_MARK,
  type ProposalAttribute,
  type ProposalRef,
  type TextPlace,
} from "../api";
import { diffSteps, hunksOf, type Step, splitWords } from "./diff";

/** The Blocks whose text can be edited: Notes' and Answers' paragraphs and headings. */
const TEXT_BLOCKS: ReadonlySet<string> = new Set(["paragraph", "heading"]);
/** Elements whose text isn't edited, and whose Blocks aren't looked into. */
const NOT_EDITED: ReadonlySet<string> = new Set(["question", "codeBlock", "blockMath"]);
/** Containers a removed paragraph takes with it when it is all they hold. */
const WRAPPERS: ReadonlySet<string> = new Set([
  "listItem",
  "bulletList",
  "orderedList",
  "blockquote",
]);

const MATH = "inlineMath";
const BREAK = "hardBreak";

type Attributes = Record<string, unknown>;

interface Run {
  insert: unknown;
  attributes?: Attributes;
}

/** A token of the selected text: where it is in its Block (in `TextPlace` offsets) and what it is. */
interface OldToken {
  /** What the diff compares (see `keyOf`). */
  key: string;
  from: number;
  to: number;
  /** How a Proposal that removes it records it (`targetText`). */
  target: string;
  /** Its text, or for an inline node its name and attributes. */
  text?: string;
  /** An inline node as the model reads it: "[1]", "$x^2$". */
  shown?: string;
  attributes?: Attributes;
  node?: InlineNode;
}

/** An inline node of the selection, or one the new text adds. */
interface InlineNode {
  name: string;
  attributes: Attributes;
  /** What it holds, copied where it moves to: a footnote's text. */
  source?: Y.XmlElement;
}

/** The part of one paragraph or heading the selection covers. */
interface Segment {
  element: Y.XmlElement;
  from: number;
  to: number;
  /** The selection covers all of its text. */
  whole: boolean;
  /**
   * Its text is a paragraph of `Selection.text`. An empty one isn't: the
   * model's paragraphs pair with only those it was shown.
   */
  shown: boolean;
  /** Its tokens in [from, to), leading and trailing spaces left out: they stay as they are. */
  tokens: OldToken[];
  /** Where those tokens start and end (`from`..`to` with the spaces left out). */
  start: number;
  end: number;
  /**
   * The kind of Block the model was shown it as (`blockMarker`): its own,
   * when the selection starts at the Block's start; a plain paragraph's
   * otherwise, as a part of a Block has no marker.
   */
  kind: BlockKind;
}

/**
 * What kind of Block a paragraph of the selection is, as the model reads it
 * and writes it back: a heading of a level ("heading:2"), a bullet, a
 * numbered item, a quote, or a plain paragraph.
 */
type BlockKind = "paragraph" | "bullet" | "numbered" | "quote" | `heading:${number}`;

/** A paragraph's or heading's kind of Block, from where it is. */
function kindOf(element: Y.XmlElement): BlockKind {
  if (element.nodeName === "heading") {
    return `heading:${Math.min(6, Math.max(1, Number(element.getAttribute("level")) || 1))}`;
  }
  const parent = element.parent;
  if (parent instanceof Y.XmlElement && parent.nodeName === "listItem") {
    const list = parent.parent;
    return list instanceof Y.XmlElement && list.nodeName === "orderedList" ? "numbered" : "bullet";
  }
  if (parent instanceof Y.XmlElement && parent.nodeName === "blockquote") return "quote";
  return "paragraph";
}

/** How the model reads a paragraph's kind: a Markdown-like marker at its start; none for a plain one. */
function blockMarker(kind: BlockKind): string {
  if (kind.startsWith("heading:")) return `${"#".repeat(Number(kind.slice("heading:".length)))} `;
  return kind === "bullet" ? "- " : kind === "numbered" ? "1. " : kind === "quote" ? "> " : "";
}

/** A Block's marker at a line's start, as the model writes it. */
const BLOCK_MARKER = String.raw`(?:#{1,6}|[-*•]|\d+[.)]|>)[ \t]`;

/**
 * Lines whose text starts like a Block's marker (a typed "1. Background", a
 * pasted "• Revenue grew"), after any backslashes that escape one: from the
 * paragraph's start, or only after its first line.
 */
const LOOKS_MARKED = {
  start: new RegExp(String.raw`(^|\n)(?=\\*${BLOCK_MARKER})`, "g"),
  next: new RegExp(String.raw`(\n)(?=\\*${BLOCK_MARKER})`, "g"),
};
const ESCAPED_MARKER = {
  start: new RegExp(String.raw`(^|\n)\\(?=\\*${BLOCK_MARKER})`, "g"),
  next: new RegExp(String.raw`(\n)\\(?=\\*${BLOCK_MARKER})`, "g"),
};

/**
 * A paragraph's text with each line that would read as a Block's marker
 * escaped the way Markdown does ("\1. Background"), so that it stays text
 * (#703). The first line has nothing to escape after a marker of its own.
 */
const escapeMarkers = (text: string, kind: BlockKind) =>
  text.replace(LOOKS_MARKED[kind === "paragraph" ? "start" : "next"], "$1\\");

/** One of the model's paragraphs with those escapes taken off: one backslash each. */
const unescapeMarkers = (paragraph: string, marked: boolean) =>
  paragraph.replace(ESCAPED_MARKER[marked ? "next" : "start"], "$1");

/** A marker at the start of one of the model's paragraphs: its kind, and how long the marker is. */
function readBlockMarker(paragraph: string): { kind: BlockKind; length: number } | null {
  const marker = new RegExp(`^${BLOCK_MARKER}+`).exec(paragraph)?.[0];
  if (marker === undefined) return null;
  const hashes = /^#+/.exec(marker)?.[0].length;
  const kind: BlockKind = hashes
    ? `heading:${hashes}`
    : marker.startsWith(">")
      ? "quote"
      : /^\d/.test(marker)
        ? "numbered"
        : "bullet";
  return { kind, length: marker.length };
}

/** A stretch of a Mind's text, read for Edit with AI. */
export interface Selection {
  segments: Segment[];
  /** The Citations in it, in order: marker n is `citations[n - 1]`. */
  citations: Attributes[];
  /** The User's footnotes in it, in order: marker "[^n]" is `footnotes[n - 1]`. */
  footnotes: Y.XmlElement[];
  /** The selected text as the editor reads it: an inline node as U+FFFC, a line per paragraph. */
  plain: string;
  /**
   * The selected text as the model reads it: markers for Citations,
   * paragraphs apart, each Block that isn't a plain paragraph with its kind's
   * marker (`blockMarker`), and then text that looks like one escaped
   * (`escapeMarkers`).
   */
  text: string;
  /** Some paragraph of `text` has a kind's marker: the model's paragraphs are read with theirs. */
  marked: boolean;
  /** The whole paragraphs it is in, the same way. */
  context: string;
  /**
   * The words just before and just after it in its paragraphs, when it is
   * part of one: its new text must fit between them, and not repeat them.
   * "" at a paragraph's edge.
   */
  before: string;
  after: string;
  /** How many characters it holds, for the model's window. */
  characters: number;
}

export type ReadResult =
  | { ok: true; selection: Selection }
  | { ok: false; reason: "changed" | "has-proposals" | "busy" };

const stringOf = (value: unknown) => (typeof value === "string" ? value : "");

/** The paragraphs and headings of the Mind, in order, at any depth, with whether an Answer being written holds them. */
export function textBlocks(blocks: Y.XmlFragment): { element: Y.XmlElement; busy: boolean }[] {
  const found: { element: Y.XmlElement; busy: boolean }[] = [];
  const visit = (parent: Y.XmlFragment | Y.XmlElement, busy: boolean) => {
    for (const child of parent.toArray()) {
      if (!(child instanceof Y.XmlElement) || NOT_EDITED.has(child.nodeName)) continue;
      if (TEXT_BLOCKS.has(child.nodeName)) {
        found.push({ element: child, busy });
        continue;
      }
      visit(
        child,
        busy || (child.nodeName === "answer" && child.getAttribute("status") === "streaming"),
      );
    }
  };
  visit(blocks, false);
  return found;
}

/** Whether a Block, or one it is in, is a Proposal's whole Block. */
function inProposal(element: Y.XmlElement): boolean {
  for (let at: unknown = element; at instanceof Y.XmlElement; at = at.parent) {
    if (at.getAttribute(PROPOSAL_ATTRIBUTE) !== undefined) return true;
  }
  return false;
}

/** An element's children with where each starts in its text, an inline node counting as one. */
export function layout(element: Y.XmlElement) {
  let at = 0;
  return element.toArray().flatMap((node) => {
    if (!(node instanceof Y.XmlText || node instanceof Y.XmlElement)) return [];
    const size = node instanceof Y.XmlText ? node.length : 1;
    const child = { node, start: at, size };
    at += size;
    return [child];
  });
}

const sizeOf = (element: Y.XmlElement) =>
  layout(element).reduce((sum, child) => sum + child.size, 0);

const hasProposal = (attributes: Attributes | undefined) =>
  !!attributes && (PROPOSAL_INSERT_MARK in attributes || PROPOSAL_DELETE_MARK in attributes);

/**
 * How a token is compared: words as they are; an inline node by what it is,
 * after a character no text holds (`n`: a Citation's or footnote's marker).
 */
const keyOf = {
  text: (text: string) => text,
  citation: (n: number) => `\u0000c${n}`,
  footnote: (n: number) => `\u0000f${n}`,
  math: (latex: string) => `\u0000m${latex}`,
  break: () => "\u0000b",
  node: (name: string, n: number) => `\u0000n${name}${n}`,
};

/**
 * Reads the text from `from` to `to`, or says why it can't be edited: a place
 * isn't in the Mind (it changed), the text holds Proposals still waiting, or
 * an Answer being written holds it.
 */
export function readSelection(blocks: Y.XmlFragment, from: TextPlace, to: TextPlace): ReadResult {
  const all = textBlocks(blocks);
  const first = all.findIndex(
    ({ element }) => element.getAttribute(BLOCK_ID_ATTRIBUTE) === from.blockId,
  );
  const last = all.findIndex(
    ({ element }) => element.getAttribute(BLOCK_ID_ATTRIBUTE) === to.blockId,
  );
  if (first < 0 || last < first) return { ok: false, reason: "changed" };
  const chosen = all.slice(first, last + 1);
  if (chosen.some((each) => each.busy)) return { ok: false, reason: "busy" };
  if (chosen.some(({ element }) => inProposal(element)))
    return { ok: false, reason: "has-proposals" };

  const citations: Attributes[] = [];
  const footnotes: Y.XmlElement[] = [];
  const segments: Segment[] = [];
  const plain: string[] = [];
  const text: string[] = [];
  const context: string[] = [];
  let characters = 0;
  /** An inline node of the selection, as the model reads it: a numbered marker, a formula. */
  const nodeToken = (node: Y.XmlElement, attributes: Attributes, index: number): NodeKey => {
    const name = node.nodeName;
    if (name === CITATION_NODE) {
      citations.push(attributes);
      return { key: keyOf.citation(citations.length), shown: `[${citations.length}]` };
    }
    if (name === FOOTNOTE_NODE) {
      footnotes.push(node);
      return {
        key: keyOf.footnote(footnotes.length),
        shown: `[^${footnotes.length}]`,
        source: node,
      };
    }
    if (name === MATH) {
      const latex = stringOf(attributes.latex);
      return { key: keyOf.math(latex), shown: `$${latex}$` };
    }
    if (name === BREAK) return { key: keyOf.break(), shown: "\n" };
    // Nothing the model can keep: it comes back without it, for the User to review.
    return { key: keyOf.node(name, index), shown: "" };
  };
  for (const [index, { element }] of chosen.entries()) {
    const size = sizeOf(element);
    const start = index === 0 ? from.offset : 0;
    const end = index === chosen.length - 1 ? to.offset : size;
    if (start < 0 || end > size || start > end) return { ok: false, reason: "changed" };
    const tokens = tokensIn(element, start, end, nodeToken);
    if (!tokens) return { ok: false, reason: "has-proposals" };
    // The spaces at either end stay as they are.
    let lo = 0;
    let hi = tokens.length;
    while (lo < hi && /^\s+$/.test(tokens[lo]?.text ?? "x")) lo++;
    while (hi > lo && /^\s+$/.test(tokens[hi - 1]?.text ?? "x")) hi--;
    const kept = tokens.slice(lo, hi);
    const shownText = tokens
      .map((token) => token.text ?? token.shown ?? "")
      .join("")
      .trim();
    const kind = start === 0 ? kindOf(element) : "paragraph";
    segments.push({
      element,
      from: start,
      to: end,
      whole: start === 0 && end === size,
      shown: shownText !== "",
      tokens: kept,
      start: kept[0]?.from ?? start,
      end: kept.at(-1)?.to ?? start,
      kind,
    });
    plain.push(tokens.map((token) => token.text ?? NODE_CHARACTER).join(""));
    text.push(shownText);
    context.push(blockText(element));
    characters += shownText.length;
  }
  const firstBlock = chosen[0]?.element;
  const lastBlock = chosen.at(-1)?.element;
  const marked = segments.some((segment) => segment.shown && segment.kind !== "paragraph");
  return {
    ok: true,
    selection: {
      segments,
      citations,
      footnotes,
      plain: plain.join("\n"),
      text: text
        .map((shown, index) => {
          const kind = segments[index]?.kind ?? "paragraph";
          return shown === "" || !marked
            ? shown
            : `${blockMarker(kind)}${escapeMarkers(shown, kind)}`;
        })
        .filter((each) => each !== "")
        .join("\n\n"),
      marked,
      context: context.join("\n\n"),
      before: firstBlock ? nearWords(blockText(firstBlock, 0, from.offset), "before") : "",
      after: lastBlock ? nearWords(blockText(lastBlock, to.offset), "after") : "",
      characters,
    },
  };
}

/** About this many characters of the words around a selection are shown to the model. */
const NEAR_CHARACTERS = 80;

/**
 * The end of `text` nearest the selection, `NEAR_CHARACTERS` of it at most,
 * without a word cut in two at the far end.
 */
function nearWords(text: string, side: "before" | "after"): string {
  if (text.length <= NEAR_CHARACTERS) return text.trim();
  const cut = side === "before" ? text.slice(-NEAR_CHARACTERS) : text.slice(0, NEAR_CHARACTERS);
  const whole = side === "before" ? cut.replace(/^\S*\s+/u, "") : cut.replace(/\s+\S*$/u, "");
  return (whole === "" ? cut : whole).trim();
}

/** How an inline node is compared, and read by the model; a footnote moved takes its text along (`source`). */
interface NodeKey {
  key: string;
  shown?: string;
  source?: Y.XmlElement;
}

/**
 * The tokens of a paragraph's or heading's text from `start` to `end`
 * (`TextPlace` offsets): its words with their marks, and its inline nodes,
 * compared as `keyNode` says. Null when a Proposal waits there.
 */
function tokensIn(
  element: Y.XmlElement,
  start: number,
  end: number,
  keyNode: (node: Y.XmlElement, attributes: Attributes, index: number) => NodeKey,
): OldToken[] | null {
  const tokens: OldToken[] = [];
  for (const child of layout(element)) {
    if (child.node instanceof Y.XmlText) {
      let at = child.start;
      for (const run of child.node.toDelta() as Run[]) {
        if (typeof run.insert !== "string") continue;
        const runFrom = at;
        at += run.insert.length;
        const cut = run.insert.slice(Math.max(0, start - runFrom), Math.max(0, end - runFrom));
        if (at <= start || runFrom >= end || cut === "") continue;
        if (hasProposal(run.attributes)) return null;
        let tokenFrom = Math.max(start, runFrom);
        for (const word of splitWords(cut)) {
          tokens.push({
            key: keyOf.text(word),
            from: tokenFrom,
            to: tokenFrom + word.length,
            target: word,
            text: word,
            attributes: run.attributes ?? {},
          });
          tokenFrom += word.length;
        }
      }
      continue;
    }
    if (child.start < start || child.start >= end) continue;
    const attributes = child.node.getAttributes() as Attributes;
    if (PROPOSAL_ATTRIBUTE in attributes) return null;
    const { key, shown, source } = keyNode(child.node, attributes, tokens.length);
    tokens.push({
      key,
      from: child.start,
      to: child.start + 1,
      target: targetText(child.node),
      ...(shown !== undefined && { shown }),
      node: { name: child.node.nodeName, attributes, ...(source && { source }) },
    });
  }
  return tokens;
}

/**
 * A paragraph's text as context for the model, all of it or from `start` to
 * `end` (`TextPlace` offsets), as the Mind reads it: what Proposals still
 * waiting add left out, inline nodes as they read, Citations as "[·]".
 */
function blockText(element: Y.XmlElement, start = 0, end = Number.POSITIVE_INFINITY): string {
  let text = "";
  for (const { node, start: at, size } of layout(element)) {
    if (at + size <= start || at >= end) continue;
    if (node instanceof Y.XmlText) {
      let offset = at;
      for (const run of node.toDelta() as Run[]) {
        const words = stringOf(run.insert);
        const from = offset;
        offset += typeof run.insert === "string" ? words.length : 1;
        if (run.attributes && PROPOSAL_INSERT_MARK in run.attributes) continue;
        text += words.slice(Math.max(0, start - from), Math.max(0, end - from));
      }
      continue;
    }
    if (readProposalAttribute(node.getAttribute(PROPOSAL_ATTRIBUTE))?.kind === "insert") continue;
    if (node.nodeName === MATH) text += `$${stringOf(node.getAttribute("latex"))}$`;
    else if (node.nodeName === BREAK) text += "\n";
    else if (node.nodeName === CITATION_NODE) text += "[·]";
  }
  return text.trim();
}

// ---------------------------------------------------------------------------
// The model's text

/** A token of the new text: words, or an inline node of the selection kept (or another). */
interface NewToken {
  key: string;
  text?: string;
  /** The words' own marks (an Answer's), instead of those of the words they replace. */
  marks?: Attributes;
  node?: InlineNode;
}

const MARKER = /\[\^(\d+)\]|\[(\d+)\]|\$([^$\n]+)\$|\n/g;

/**
 * One of the new text's paragraphs: its tokens, and the kind of Block it is
 * by its marker. Null kind: the model wrote no markers at all, though it was
 * shown some, so the paragraph's kind is unknown and it may take any Block's.
 */
interface NewParagraph {
  kind: BlockKind | null;
  tokens: NewToken[];
}

/**
 * The new text's paragraphs as tokens. A marker "[n]" is Citation n of the
 * selection, with `fresh` attributes when its check came out differently
 * now, and "[^n]" is its footnote n; a marker for none it had is left out.
 * A formula the selection had is kept; another stays text. When the
 * selection was shown with its Blocks' markers, each paragraph's marker is
 * read off as its kind, and the escapes of text that looks like one are
 * taken off.
 */
function newParagraphs(
  text: string,
  selection: Selection,
  fresh: ReadonlyMap<number, Attributes>,
): NewParagraph[] {
  const formulas = new Set(
    selection.segments.flatMap((segment) =>
      segment.tokens.flatMap((token) =>
        token.node?.name === MATH ? [stringOf(token.node.attributes.latex)] : [],
      ),
    ),
  );
  const citationNode = (n: number): NewToken | null => {
    const attributes = fresh.get(n) ?? selection.citations[n - 1];
    if (!attributes) return null;
    return {
      key: `${keyOf.citation(n)}${fresh.has(n) ? "*" : ""}`,
      node: { name: CITATION_NODE, attributes },
    };
  };
  const footnoteNode = (n: number): NewToken | null => {
    const source = selection.footnotes[n - 1];
    if (!source) return null;
    return {
      key: keyOf.footnote(n),
      node: { name: FOOTNOTE_NODE, attributes: source.getAttributes() as Attributes, source },
    };
  };
  const tokensOf = (paragraph: string): NewToken[] => {
    const tokens: NewToken[] = [];
    const words = (part: string) => {
      for (const word of splitWords(part)) tokens.push({ key: keyOf.text(word), text: word });
    };
    let at = 0;
    for (const match of paragraph.matchAll(MARKER)) {
      words(paragraph.slice(at, match.index));
      at = match.index + match[0].length;
      const [, footnote, citation, formula] = match;
      if (match[0] === "\n") {
        tokens.push({ key: keyOf.break(), node: { name: BREAK, attributes: {} } });
      } else if (footnote !== undefined || citation !== undefined) {
        const n = Number(footnote ?? citation);
        const node = footnote !== undefined ? footnoteNode(n) : citationNode(n);
        if (node) tokens.push(node);
        // Spaces left where an unknown marker was are one.
        else if (/\s$/.test(paragraph.slice(0, match.index)) && /^\s/.test(paragraph.slice(at)))
          at++;
      } else if (formula !== undefined && formulas.has(formula)) {
        tokens.push({
          key: keyOf.math(formula),
          node: { name: MATH, attributes: { latex: formula } },
        });
      } else {
        words(match[0]);
      }
    }
    words(paragraph.slice(at));
    return tokens;
  };
  // A line that starts with a Block's mark starts a paragraph, even after a single line break.
  const apart = selection.marked
    ? new RegExp(String.raw`\n[ \t]*\n+|\n(?=${BLOCK_MARKER})`)
    : /\n[ \t]*\n+/;
  const paragraphs = text
    .split(apart)
    .map((paragraph) => paragraph.trim())
    .filter((paragraph) => paragraph !== "")
    .map((written) => {
      if (!selection.marked) return { marker: null, paragraph: written };
      const marker = readBlockMarker(written);
      return {
        marker,
        paragraph: unescapeMarkers(
          marker ? written.slice(marker.length) : written,
          marker !== null,
        ),
      };
    });
  const anyMarker = paragraphs.some(({ marker }) => marker !== null);
  return paragraphs.map(({ marker, paragraph }) => ({
    kind: marker?.kind ?? (selection.marked && !anyMarker ? null : "paragraph"),
    tokens: tokensOf(paragraph),
  }));
}

// ---------------------------------------------------------------------------
// Writing

export interface Written {
  /** The Proposals written, in order. */
  proposals: ProposalRef[];
  /** Citations the Proposals would remove and add back nowhere. */
  droppedCitations: number;
}

/** What a Proposal's marks and attributes hold: which it is, and the text it changes. */
const refOf = ({ id, request, target }: ProposalRef): ProposalRef =>
  target === undefined ? { id, request } : { id, request, target };

/** The marks of words a Proposal adds: AI text, unless they are inline code, which holds no other mark. */
const asAiText = (marks: Attributes): Attributes =>
  INLINE_CODE_MARK in marks ? marks : { ...marks, [GENERATED_MARK]: {} };

/** Text a Proposal adds, and its marks, with no other Proposal's. */
function withoutProposals(attributes: Attributes | undefined): Attributes {
  const kept: Attributes = {};
  for (const [key, value] of Object.entries(attributes ?? {})) {
    if (key !== PROPOSAL_INSERT_MARK && key !== PROPOSAL_DELETE_MARK) kept[key] = value;
  }
  return kept;
}

/** The marks of the text next to where words are added: no link or code, which they aren't part of. */
function besideMarks(segment: Segment, at: number): Attributes {
  const near =
    [...segment.tokens].reverse().find((token) => token.text !== undefined && token.to <= at) ??
    segment.tokens.find((token) => token.text !== undefined && token.from >= at);
  const { link: _link, code: _code, ...marks } = withoutProposals(near?.attributes);
  return marks;
}

/** Marks `from`..`to` of a Block's text as removed by the Proposal. */
function markRemoved(element: Y.XmlElement, from: number, to: number, ref: ProposalRef): void {
  for (const child of layout(element)) {
    const start = Math.max(from, child.start);
    const end = Math.min(to, child.start + child.size);
    if (start >= end) continue;
    if (child.node instanceof Y.XmlText) {
      child.node.format(start - child.start, end - start, { [PROPOSAL_DELETE_MARK]: refOf(ref) });
    } else {
      setProposal(child.node, { ...refOf(ref), kind: "delete" });
    }
  }
}

/** Sets a node's `proposal` attribute. Yjs types attributes as text, but keeps any JSON value. */
export function setProposal(element: Y.XmlElement, value: ProposalAttribute): void {
  (element as unknown as Y.XmlElement<Record<string, ProposalAttribute>>).setAttribute(
    PROPOSAL_ATTRIBUTE,
    value,
  );
}

/** Puts `node` into a Block's text at `at`, splitting the text there if it must. */
function insertNode(element: Y.XmlElement, at: number, node: Y.XmlElement): void {
  const children = layout(element);
  const inside = children.find(
    (child) => child.node instanceof Y.XmlText && child.start < at && at < child.start + child.size,
  );
  if (inside && inside.node instanceof Y.XmlText) {
    const local = at - inside.start;
    const tail: Run[] = [];
    let offset = 0;
    for (const run of inside.node.toDelta() as Run[]) {
      if (typeof run.insert !== "string") continue;
      const end = offset + run.insert.length;
      if (end > local) {
        tail.push({
          insert: run.insert.slice(Math.max(0, local - offset)),
          attributes: run.attributes ?? {},
        });
      }
      offset = end;
    }
    inside.node.delete(local, inside.size - local);
    const rest = new Y.XmlText();
    rest.applyDelta(tail.map((run) => ({ insert: run.insert, attributes: run.attributes ?? {} })));
    const index = children.indexOf(inside);
    element.insert(index + 1, [node, rest]);
    return;
  }
  element.insert(children.filter((child) => child.start + child.size <= at).length, [node]);
}

/** Puts `text` into a Block's text at `at`, with exactly these marks. */
function insertText(element: Y.XmlElement, at: number, text: string, marks: Attributes): void {
  const children = layout(element);
  const target =
    children.find(
      (child) =>
        child.node instanceof Y.XmlText && child.start < at && at <= child.start + child.size,
    ) ?? children.find((child) => child.node instanceof Y.XmlText && child.start === at);
  if (target && target.node instanceof Y.XmlText) {
    target.node.insert(at - target.start, text, { ...marks });
    return;
  }
  const fresh = new Y.XmlText();
  fresh.insert(0, text, { ...marks });
  element.insert(children.filter((child) => child.start + child.size <= at).length, [fresh]);
}

/**
 * An inline node for new text: a Citation, formula or line break, or a
 * footnote moved, copied with its text under an id of its own, since the
 * one it was copied from stays until the User accepts.
 */
function elementOf(node: InlineNode, newId: () => string): Y.XmlElement {
  if (node.source) {
    const copy = node.source.clone();
    copy.removeAttribute(PROPOSAL_ATTRIBUTE);
    copy.setAttribute(BLOCK_ID_ATTRIBUTE, newId());
    return copy;
  }
  return createElement({ type: node.name, attrs: withoutProposalAttribute(node.attributes) });
}

/** An inline node the Proposal adds. */
function nodeElement(node: InlineNode, ref: ProposalRef, newId: () => string): Y.XmlElement {
  const element = elementOf(node, newId);
  setProposal(element, { ...refOf(ref), kind: "insert" });
  return element;
}

function withoutProposalAttribute(attributes: Attributes): Attributes {
  const { [PROPOSAL_ATTRIBUTE]: _proposal, ...rest } = attributes;
  return rest;
}

const sameMarks = (a: Attributes, b: Attributes) => JSON.stringify(a) === JSON.stringify(b);

/**
 * Adds the new tokens at `at` in a Block's text, as the Proposal's words:
 * with their own marks (an Answer's), or else with `marks`.
 */
function insertTokens(
  element: Y.XmlElement,
  at: number,
  tokens: readonly NewToken[],
  marks: Attributes,
  ref: ProposalRef,
  newId: () => string,
): void {
  const inserted = refOf(ref);
  let position = at;
  let words = "";
  let wordMarks = marks;
  const flush = () => {
    if (words === "") return;
    insertText(element, position, words, {
      ...asAiText(wordMarks),
      [PROPOSAL_INSERT_MARK]: inserted,
    });
    position += words.length;
    words = "";
  };
  for (const token of tokens) {
    if (token.node) {
      flush();
      insertNode(element, position, nodeElement(token.node, ref, newId));
      position += 1;
      continue;
    }
    const own = token.marks ?? marks;
    if (!sameMarks(own, wordMarks)) flush();
    wordMarks = own;
    words += token.text ?? "";
  }
  flush();
}

/** A new paragraph or heading holding these tokens, with these attributes. */
function textBlockOf(
  tokens: readonly NewToken[],
  newId: () => string,
  type = "paragraph",
  attrs: Attributes = {},
): Y.XmlElement {
  const block = createElement({ type, attrs });
  const children: (Y.XmlElement | Y.XmlText)[] = [];
  let runs: Run[] = [];
  const flush = () => {
    if (runs.length === 0) return;
    const text = new Y.XmlText();
    text.applyDelta(runs.map((run) => ({ insert: run.insert, attributes: run.attributes ?? {} })));
    children.push(text);
    runs = [];
  };
  for (const token of tokens) {
    if (token.node) {
      flush();
      children.push(elementOf(token.node, newId));
      continue;
    }
    const marks = asAiText(token.marks ?? {});
    const last = runs.at(-1);
    if (last && sameMarks(last.attributes ?? {}, marks)) last.insert += token.text ?? "";
    else runs.push({ insert: token.text ?? "", attributes: marks });
  }
  flush();
  block.insert(0, children);
  return block;
}

/** A new paragraph, as the Proposal's whole Block, holding these tokens. */
function paragraphOf(
  tokens: readonly NewToken[],
  ref: ProposalRef,
  newId: () => string,
): Y.XmlElement {
  return textBlockOf(tokens, newId, "paragraph", {
    [PROPOSAL_ATTRIBUTE]: { ...refOf(ref), kind: "insert" } satisfies ProposalAttribute,
  });
}

/** The element a removed paragraph takes with it: the list item, list or quote that holds only it. */
function removedWith(element: Y.XmlElement): Y.XmlElement {
  let current = element;
  let parent = current.parent;
  while (parent instanceof Y.XmlElement && WRAPPERS.has(parent.nodeName) && parent.length === 1) {
    current = parent;
    parent = current.parent;
  }
  return current;
}

/**
 * Blocks to remove, with each list or quote they empty removed whole in their
 * place, so that accepting the Proposals leaves no empty list behind, which
 * the editor would drop beyond Undo's reach. In the order they come.
 */
function wholeWrappers(removed: readonly Y.XmlElement[]): Y.XmlElement[] {
  let units = [...removed];
  for (let changed = true; changed; ) {
    changed = false;
    for (const unit of units) {
      const parent = unit.parent;
      if (!(parent instanceof Y.XmlElement) || !WRAPPERS.has(parent.nodeName)) continue;
      const children = parent.toArray().filter((child) => child instanceof Y.XmlElement);
      if (!children.every((child) => units.includes(child))) continue;
      units = [...units.filter((each) => !children.includes(each)), parent];
      changed = true;
      break;
    }
  }
  const order = (unit: Y.XmlElement) => removed.findIndex((each) => isInside(each, unit));
  return units.sort((a, b) => order(a) - order(b));
}

/** Whether `element` is `container` or inside it. */
function isInside(element: Y.XmlElement, container: Y.XmlElement): boolean {
  for (let at: unknown = element; at instanceof Y.XmlElement; at = at.parent) {
    if (at === container) return true;
  }
  return false;
}

/** Adds `elements` after `anchor`, in its parent. */
function addAfter(anchor: Y.XmlElement, elements: Y.XmlElement[]): void {
  addBeside(anchor, elements, "after");
}

/** Adds `elements` just after or just before `anchor`, in its parent. */
function addBeside(anchor: Y.XmlElement, elements: Y.XmlElement[], side: "after" | "before"): void {
  const parent = anchor.parent;
  if (!(parent instanceof Y.XmlElement || parent instanceof Y.XmlFragment)) return;
  const index = parent.toArray().indexOf(anchor);
  parent.insert(side === "after" ? index + 1 : index, elements);
}

/** The list item, list or quote a Block sits in, as far out as they go. */
function outermost(element: Y.XmlElement): Y.XmlElement {
  let current = element;
  while (current.parent instanceof Y.XmlElement && WRAPPERS.has(current.parent.nodeName)) {
    current = current.parent;
  }
  return current;
}

/** Where a new Block goes: next to a paragraph or heading, which is of this kind. */
interface Place {
  element: Y.XmlElement;
  kind: BlockKind;
  side: "after" | "before";
}

/**
 * Adds a new Block of `kind` holding `tokens`, as a Proposal of its own, at
 * `place`: a list item into the list the place is in, when it is of that
 * list's kind, or else in a list of its own; a plain paragraph beside a plain
 * one; anything else beside the list or quote the place is in. Returns its
 * paragraph or heading, where the next new Block goes.
 */
function addBlock(
  place: Place,
  kind: BlockKind,
  tokens: readonly NewToken[],
  ref: ProposalRef,
  newId: () => string,
): Y.XmlElement {
  const proposal: ProposalAttribute = { ...refOf(ref), kind: "insert" };
  const heading = kind.startsWith("heading:");
  const text = textBlockOf(
    tokens,
    newId,
    heading ? "heading" : "paragraph",
    heading ? { level: Number(kind.slice("heading:".length)) } : {},
  );
  const listType = kind === "bullet" ? "bulletList" : kind === "numbered" ? "orderedList" : null;
  const item = place.element.parent;
  if (
    listType &&
    item instanceof Y.XmlElement &&
    item.nodeName === "listItem" &&
    item.parent instanceof Y.XmlElement &&
    item.parent.nodeName === listType
  ) {
    const added = createElement({ type: "listItem" });
    setProposal(added, proposal);
    added.insert(0, [text]);
    addBeside(item, [added], place.side);
    return text;
  }
  if (kind === "paragraph" && place.kind === "paragraph") {
    setProposal(text, proposal);
    addBeside(place.element, [text], place.side);
    return text;
  }
  let block = text;
  if (listType) {
    const added = createElement({ type: "listItem" });
    added.insert(0, [text]);
    block = createElement({ type: listType });
    block.insert(0, [added]);
  } else if (kind === "quote") {
    block = createElement({ type: "blockquote" });
    block.insert(0, [text]);
  }
  setProposal(block, proposal);
  addBeside(outermost(place.element), [block], place.side);
  return text;
}

/** The Citation a token is, by its marker (or by what it cites, in an Answer); null for any other token. */
function markerOf(token: { key: string }): string | null {
  const prefix = keyOf.citation(0).slice(0, -1);
  if (!token.key.startsWith(prefix)) return null;
  return token.key.slice(prefix.length).replace(/\*$/, "");
}

/** A run of kept tokens between two changes that joins them: only spaces and a few signs or letters. */
function joinsAcross(tokens: readonly OldToken[]) {
  return (from: number, to: number) => {
    const kept = tokens.slice(from, to);
    if (kept.some((token) => token.node)) return false;
    return (
      kept
        .map((token) => token.text ?? "")
        .join("")
        .replace(/\s+/g, "").length <= 3
    );
  };
}

/** What a Proposal that removes these tokens records as the text it changes. */
const targetOf = (tokens: readonly OldToken[]) => tokens.map((token) => token.target).join("");

/** The Citations Proposals remove and add, by marker, to count those removed and added back nowhere. */
interface CitationNotes {
  removed: Set<string>;
  added: Set<string>;
}

const noteCitation = (into: Set<string>, token: { key: string }) => {
  const marker = markerOf(token);
  if (marker) into.add(marker);
};

const droppedOf = (notes: CitationNotes) =>
  [...notes.removed].filter((marker) => !notes.added.has(marker)).length;

/**
 * Less than half the words kept: a paragraph written anew, better read as one
 * change than as a change at every other word.
 */
function mostlyNew(steps: readonly Step[], old: readonly OldToken[], next: readonly NewToken[]) {
  const isWord = (text: string | undefined) => text === undefined || !/^\s+$/.test(text);
  let kept = 0;
  let at = 0;
  for (const step of steps) {
    if (step === "insert") continue;
    if (step === "equal" && isWord(old[at]?.text)) kept++;
    at++;
  }
  const words = Math.max(
    old.filter((token) => isWord(token.text)).length,
    next.filter((token) => isWord(token.text)).length,
  );
  return kept * 2 < words;
}

/**
 * Writes the changes from a segment's tokens to `tokens`, each place that
 * changed a Proposal of its own (`next`, given the text it changes), from the
 * end, so the places read before each are still right. With `whole`, a
 * paragraph mostly written anew is one change.
 */
function writeHunks(
  segment: Segment,
  tokens: readonly NewToken[],
  next: (target: string) => ProposalRef,
  newId: () => string,
  notes: CitationNotes,
  whole = false,
): void {
  const steps = diffSteps(
    segment.tokens.map((token) => token.key),
    tokens.map((token) => token.key),
  );
  let hunks = hunksOf(steps, joinsAcross(segment.tokens));
  if (whole && hunks.length > 1 && mostlyNew(steps, segment.tokens, tokens)) {
    hunks = [{ oldFrom: 0, oldTo: segment.tokens.length, newFrom: 0, newTo: tokens.length }];
  }
  const refs = hunks.map((hunk) => next(targetOf(segment.tokens.slice(hunk.oldFrom, hunk.oldTo))));
  for (let at = hunks.length - 1; at >= 0; at--) {
    const hunk = hunks[at] as (typeof hunks)[number];
    const ref = refs[at] as ProposalRef;
    const removed = segment.tokens.slice(hunk.oldFrom, hunk.oldTo);
    const added = tokens.slice(hunk.newFrom, hunk.newTo);
    const end = removed.at(-1)?.to ?? segment.tokens[hunk.oldFrom]?.from ?? segment.end;
    const firstWords = removed.find((token) => token.text !== undefined);
    const marks = firstWords ? withoutProposals(firstWords.attributes) : besideMarks(segment, end);
    insertTokens(segment.element, end, added, marks, ref, newId);
    if (removed.length > 0) markRemoved(segment.element, removed[0]?.from ?? end, end, ref);
    for (const token of removed) noteCitation(notes.removed, token);
    for (const token of added) noteCitation(notes.added, token);
  }
}

/** A Proposal maker: each new Proposal of `request`, recording the text it changes, in order. */
function proposalsOf(request: string, newId: () => string) {
  const proposals: ProposalRef[] = [];
  const next = (target: string): ProposalRef => {
    const ref = { id: newId(), request, target: targetHash(target) };
    proposals.push(ref);
    return ref;
  };
  return { proposals, next };
}

/** The words of a run, for telling how alike two runs are: no spaces or punctuation, any case. */
const wordsOf = (tokens: readonly { key: string; text?: string }[]) =>
  tokens.flatMap((token) =>
    token.text === undefined
      ? [token.key.replace(/\*$/, "")]
      : /[\p{L}\p{N}]/u.test(token.text)
        ? [token.text.toLowerCase()]
        : [],
  );

/** How alike two runs of words are, from 0 to 1: the share of their words they have in common. */
function likeness(old: readonly OldToken[], next: readonly NewToken[]): number {
  const a = wordsOf(old);
  const b = wordsOf(next);
  if (a.length + b.length === 0) return 1;
  const counts = new Map<string, number>();
  for (const word of a) counts.set(word, (counts.get(word) ?? 0) + 1);
  let shared = 0;
  for (const word of b) {
    const left = counts.get(word) ?? 0;
    if (left === 0) continue;
    shared++;
    counts.set(word, left - 1);
  }
  return (2 * shared) / (a.length + b.length);
}

/** How alike a paragraph of unknown kind must be to one of the selection's to pair with it. */
const UNKNOWN_KIND_LIKENESS = 0.3;
/** How alike a new paragraph of unknown kind must be to one of the selection's to take its kind. */
const MOVED_LIKENESS = 0.5;

/**
 * The kind of a new paragraph the model wrote no mark for: that of the
 * selection's paragraph it is most like, when it is that paragraph moved;
 * else a plain paragraph's.
 */
function kindByLikeness(segments: readonly Segment[], paragraph: NewParagraph): BlockKind {
  let found: BlockKind = "paragraph";
  let best = MOVED_LIKENESS;
  for (const segment of segments) {
    const alike = likeness(segment.tokens, paragraph.tokens);
    if (alike >= best) {
      best = alike;
      found = segment.kind;
    }
  }
  return found;
}

/**
 * Which of the selection's paragraphs each of the model's is, in order: the
 * pairing with the most pairs of the same kind of Block, the more alike the
 * better, so that a paragraph the new text moves is added where it now goes
 * rather than written into whatever paragraph held that place. One of
 * unknown kind pairs with any. Null: a new paragraph, paired with none.
 */
function pairUp(
  segments: readonly Segment[],
  paragraphs: readonly NewParagraph[],
): (number | null)[] {
  const weight = (i: number, j: number): number | null => {
    const segment = segments[i] as Segment;
    const paragraph = paragraphs[j] as NewParagraph;
    if (paragraph.kind !== null && paragraph.kind !== segment.kind) return null;
    const alike = likeness(segment.tokens, paragraph.tokens);
    // Of unknown kind, only a paragraph like it can take a Block's place.
    if (paragraph.kind === null && alike < UNKNOWN_KIND_LIKENESS) return null;
    return 1 + 2 * alike;
  };
  const rows = segments.length;
  const columns = paragraphs.length;
  // best[i][j]: the most the paragraphs from i and from j can pair for.
  const best = Array.from({ length: rows + 1 }, () => new Array<number>(columns + 1).fill(0));
  const at = (i: number, j: number) => best[i]?.[j] ?? 0;
  for (let i = rows - 1; i >= 0; i--) {
    for (let j = columns - 1; j >= 0; j--) {
      const pair = weight(i, j);
      (best[i] as number[])[j] = Math.max(
        at(i + 1, j),
        at(i, j + 1),
        pair === null ? Number.NEGATIVE_INFINITY : pair + at(i + 1, j + 1),
      );
    }
  }
  // From the start, pairing whenever that is as good as anything else: a tie pairs in order.
  const paired: (number | null)[] = new Array<number | null>(columns).fill(null);
  const TIE = 1e-9;
  let i = 0;
  let j = 0;
  while (i < rows && j < columns) {
    const pair = weight(i, j);
    if (pair !== null && pair + at(i + 1, j + 1) >= at(i, j) - TIE) {
      paired[j] = i;
      i++;
      j++;
    } else if (at(i + 1, j) >= at(i, j) - TIE) i++;
    else j++;
  }
  return paired;
}

/**
 * Writes the model's `text` for the selection as Proposals of `request`.
 * Each of its paragraphs pairs with the selection's paragraph it is (see
 * `pairUp`) and is compared with it word by word, so a heading stays a
 * heading and a list item a list item; a paragraph that pairs with none is a
 * new Block of its kind where it goes in the new text; and a paragraph of the
 * selection with none goes (its whole Block, if all of it was selected).
 * `fresh`: Citations whose check came out differently now, by marker, with
 * their new attributes: one the text keeps is replaced by the checked copy.
 * Call it in a Yjs transaction.
 */
export function proposeRewrite(
  selection: Selection,
  text: string,
  request: string,
  newId: () => string,
  fresh: ReadonlyMap<number, Attributes> = new Map(),
): Written {
  const rewrite = createRewriteStream(selection, request, newId, () => fresh);
  rewrite.write(text, true);
  rewrite.finish();
  return rewrite.written();
}

/** How many of the selection's paragraphs a new one may pass over to pair with one much like it: those it passes go. */
const LOOKAHEAD = 2;

/** Where the next of the new text's paragraphs goes: anew over this paragraph of the selection, or after this one. */
export type RewritePlace = { over: Y.XmlElement } | { after: Y.XmlElement };

/**
 * `proposeRewrite`, as the model's text comes in (#202): each paragraph is
 * written once it is finished and its place is known, and nothing written
 * is written again, so nothing moves under the User. A paragraph much like
 * one of the selection's a little further on pairs with it at once, and those
 * it passes over go. Any other waits for the next that is (or the end), and
 * those between are paired as `pairUp` pairs them, so a section added in the
 * middle is added, not written over the one after it, and one moved up is
 * added where it now goes. When two are like none of the selection's before
 * any is much like one (a translation), each pairs in order as it comes.
 * `finish` takes out the selection's paragraphs the text never reached; a
 * request stopped before leaves them. Call each in a Yjs transaction on the
 * document `selection` was read from.
 */
export function createRewriteStream(
  selection: Selection,
  request: string,
  newId: () => string,
  fresh: (text: string) => ReadonlyMap<number, Attributes>,
) {
  const { proposals, next } = proposalsOf(request, newId);
  const notes: CitationNotes = { removed: new Set(), added: new Set() };
  // Only the paragraphs the model was shown: an empty one between two isn't in its text.
  const segments = selection.segments.filter((segment) => segment.shown);
  const first = segments[0];
  let place: Place | null = first
    ? { element: first.element, kind: first.kind, side: "before" }
    : null;
  /** The selection's next paragraph not yet paired or gone. */
  let at = 0;
  /** How many of the new text's paragraphs are taken: written, or held. */
  let taken = 0;
  /** Paragraphs like none of the selection's, waiting for their place. */
  const held: NewParagraph[] = [];
  /** A paragraph much like one of the selection's was paired: the text is mostly kept. */
  let anchored = false;
  /** Paragraphs like none of the selection's at all, before any was much like one. */
  let unlike = 0;
  /** Two were like none: the text is written anew, and each pairs in order. */
  let anew = false;

  /** The selection's paragraphs before `to` not paired go, each whole one with its Block. */
  const removeUpTo = (to: number) => {
    const gone: Y.XmlElement[] = [];
    for (const segment of segments.slice(at, to)) {
      if (segment.tokens.length === 0) continue;
      if (segment.whole) gone.push(removedWith(segment.element));
      else markRemoved(segment.element, segment.start, segment.end, next(targetOf(segment.tokens)));
      for (const token of segment.tokens) noteCitation(notes.removed, token);
    }
    for (const removed of wholeWrappers(gone)) {
      setProposal(removed, { ...refOf(next(targetText(removed))), kind: "delete" });
    }
    at = Math.max(at, to);
  };

  const pairWith = (paragraph: NewParagraph, index: number) => {
    const segment = segments[index];
    if (!segment) return;
    removeUpTo(index);
    writeHunks(segment, paragraph.tokens, next, newId, notes);
    place = { element: segment.element, kind: segment.kind, side: "after" };
    at = index + 1;
  };

  const addNew = (paragraph: NewParagraph) => {
    if (!place) return;
    const kind = paragraph.kind ?? kindByLikeness(segments, paragraph);
    for (const token of paragraph.tokens) noteCitation(notes.added, token);
    const added = addBlock(place, kind, paragraph.tokens, next(""), newId);
    place = { element: added, kind, side: "after" };
  };

  /** The paragraphs held, written: paired with the selection's from `at` to `to` as `pairUp` pairs them. */
  const writeHeld = (to: number) => {
    const from = at;
    const paired = pairUp(segments.slice(from, to), held);
    for (const [index, paragraph] of held.entries()) {
      const pair = paired[index];
      if (pair === null || pair === undefined) addNew(paragraph);
      else pairWith(paragraph, from + pair);
    }
    held.length = 0;
  };

  /** The first of the selection's paragraphs from `at` much like `paragraph`, or null. */
  const muchLike = (paragraph: NewParagraph): number | null => {
    for (let index = at; index < segments.length; index++) {
      const segment = segments[index] as Segment;
      if (paragraph.kind !== null && paragraph.kind !== segment.kind) continue;
      if (likeness(segment.tokens, paragraph.tokens) >= MOVED_LIKENESS) return index;
    }
    return null;
  };

  const take = (paragraph: NewParagraph) => {
    const like = muchLike(paragraph);
    if (like !== null && like - at <= LOOKAHEAD + held.length) {
      writeHeld(like);
      pairWith(paragraph, like);
      anchored = true;
      return;
    }
    held.push(paragraph);
    if (like === null && !anchored) unlike++;
    if (unlike >= 2) anew = true;
    if (at >= segments.length) writeHeld(at);
    else if (anew) writeHeld(Math.min(segments.length, at + held.length + LOOKAHEAD));
  };

  /** Runs `write`, and says whether it wrote any Proposal. */
  const wrote = (write: () => void): boolean => {
    const before = proposals.length;
    write();
    return proposals.length > before;
  };

  return {
    /**
     * The model's text so far: the paragraphs it has finished (all but the
     * last, or all of them once `done`) are written, or held until their
     * place is known. All at once, they pair as `pairUp` pairs them.
     * Returns whether any Proposal was written.
     */
    write: (text: string, done = false): boolean =>
      wrote(() => {
        const paragraphs = newParagraphs(text, selection, fresh(text));
        if (done && taken === 0) {
          held.push(...paragraphs);
          taken = paragraphs.length;
        }
        const finished = done ? paragraphs.length : paragraphs.length - 1;
        for (; taken < finished; taken++) {
          const paragraph = paragraphs[taken];
          if (paragraph) take(paragraph);
        }
        if (done) writeHeld(segments.length);
      }),

    /**
     * Stopped: the paragraphs held are written, each into the selection's
     * next one when much like it, else as a new Block where the text has got
     * to. Nothing the model never reached goes, so none of the selection's
     * paragraphs is removed.
     */
    stop: (): boolean =>
      wrote(() => {
        for (const paragraph of held) {
          const segment = segments[at];
          const alike =
            segment !== undefined &&
            (paragraph.kind === null || paragraph.kind === segment.kind) &&
            likeness(segment.tokens, paragraph.tokens) >= MOVED_LIKENESS;
          if (!alike) {
            addNew(paragraph);
            continue;
          }
          writeHunks(segment, paragraph.tokens, next, newId, notes);
          place = { element: segment.element, kind: segment.kind, side: "after" };
          at += 1;
        }
        held.length = 0;
      }),

    /** The model is done: the selection's paragraphs it never reached go. */
    finish: (): boolean =>
      wrote(() => {
        writeHeld(segments.length);
        removeUpTo(segments.length);
      }),

    /** The Proposals written so far. */
    written: (): Written => ({ proposals: [...proposals], droppedCitations: droppedOf(notes) }),

    /** Where the next paragraph goes; null with no place for any. */
    writingAt(): RewritePlace | null {
      const ahead = segments[at];
      if (ahead) return { over: ahead.element };
      return place ? { after: place.element } : null;
    },
  };
}

export type RewriteStream = ReturnType<typeof createRewriteStream>;

/**
 * Adds `text` under the selection, as new paragraphs after the top-level
 * Block it ends in, each a Proposal of `request`. Markers in it are dropped:
 * nothing it says was checked. Call it in a Yjs transaction.
 */
export function proposeAfter(
  blocks: Y.XmlFragment,
  selection: Selection,
  text: string,
  request: string,
  newId: () => string,
): Written {
  const last = selection.segments.at(-1)?.element;
  let top: Y.XmlElement | null = last ?? null;
  while (top && top.parent instanceof Y.XmlElement) top = top.parent;
  if (!top || top.parent !== blocks) return { proposals: [], droppedCitations: 0 };
  const { proposals, next } = proposalsOf(request, newId);
  const paragraphs = text
    .replace(/\s*\[\^?\d+\]/g, "")
    .split(/\n[ \t]*\n+/)
    .map((paragraph) => paragraph.trim())
    .filter((paragraph) => paragraph !== "");
  const added = paragraphs.map((paragraph) =>
    paragraphOf(
      splitWords(paragraph).map((word) => ({ key: keyOf.text(word), text: word })),
      next(""),
      newId,
    ),
  );
  addAfter(top, added);
  return { proposals, droppedCitations: 0 };
}

// ---------------------------------------------------------------------------
// An Answer written again, over the User's version

/** What tells one Citation from another in an Answer: what it cites, not its number. */
function citedKey(attributes: Attributes): string {
  const location = attributes.location;
  return [
    attributes.documentId,
    attributes.contentHash,
    attributes.quote,
    attributes.pageFrom,
    attributes.pageTo,
    location === undefined || location === null ? "" : JSON.stringify(location),
  ]
    .map((value) => (value === undefined || value === null ? "" : String(value)))
    .join("\u0001");
}

/** How an inline node of an Answer is compared: a Citation by what it cites, a formula by its LaTeX. */
function answerNodeKey(name: string, attributes: Attributes): string {
  if (name === CITATION_NODE) return `${keyOf.citation(0).slice(0, -1)}${citedKey(attributes)}`;
  if (name === MATH) return keyOf.math(stringOf(attributes.latex));
  if (name === BREAK) return keyOf.break();
  return `\u0000n${name}${JSON.stringify(withoutProposalAttribute(attributes))}`;
}

/** The attributes that make two Blocks of a type differ, beyond their text. */
const shapeOf = (name: string, attributes: Attributes) =>
  `${name}:${[attributes.level, attributes.start, attributes.language]
    .map((value) => (value === undefined || value === null ? "" : String(value)))
    .join(",")}`;

/** How a Block of the Mind is compared with one of the new Answer: its type, shape and text. */
function elementKey(element: Y.XmlElement): string {
  const attributes = element.getAttributes() as Attributes;
  if (INLINE_NODES.has(element.nodeName)) return answerNodeKey(element.nodeName, attributes);
  const inside = element
    .toArray()
    .map((child) => {
      if (child instanceof Y.XmlText) {
        return (child.toDelta() as Run[]).map((run) => stringOf(run.insert)).join("");
      }
      return child instanceof Y.XmlElement ? elementKey(child) : "";
    })
    .join("");
  return `\u0003${shapeOf(element.nodeName, attributes)}{${inside}}`;
}

/** The same, for a Block of the new Answer. */
function jsonKey(node: NodeJSON): string {
  if (node.type === "text") return node.text ?? "";
  const attributes = node.attrs ?? {};
  if (INLINE_NODES.has(node.type)) return answerNodeKey(node.type, attributes);
  const inside = (node.content ?? []).map(jsonKey).join("");
  return `\u0003${shapeOf(node.type, attributes)}{${inside}}`;
}

/** The new Answer's inline content as tokens, its words with their own marks. */
function jsonTokens(content: readonly NodeJSON[]): NewToken[] {
  const tokens: NewToken[] = [];
  for (const node of content) {
    if (node.type === "text") {
      const marks = formattingOf(node.marks) ?? {};
      for (const word of splitWords(node.text ?? "")) {
        tokens.push({ key: keyOf.text(word), text: word, marks });
      }
      continue;
    }
    const attributes = node.attrs ?? {};
    tokens.push({
      key: answerNodeKey(node.type, attributes),
      node: { name: node.type, attributes },
    });
  }
  return tokens;
}

/** Every Citation in an element, or in the new Answer's Block, by what it cites. */
function citationsIn(node: Y.XmlElement | NodeJSON): string[] {
  if (node instanceof Y.XmlElement) {
    if (node.nodeName === CITATION_NODE) return [citedKey(node.getAttributes() as Attributes)];
    return node
      .toArray()
      .flatMap((child) => (child instanceof Y.XmlElement ? citationsIn(child) : []));
  }
  if (node.type === CITATION_NODE) return [citedKey(node.attrs ?? {})];
  return (node.content ?? []).flatMap(citationsIn);
}

/** Two Blocks whose words are compared one by one: paragraphs, or headings of a level. */
const comparable = (element: Y.XmlElement, node: NodeJSON) =>
  TEXT_BLOCKS.has(element.nodeName) &&
  element.nodeName === node.type &&
  shapeOf(element.nodeName, element.getAttributes() as Attributes) ===
    shapeOf(node.type, node.attrs ?? {});

/**
 * Writes `content` (an Answer written again) over what `container` holds
 * now, as Proposals of `request`: Blocks that are the same stay as they are;
 * where they differ, a paragraph or heading paired with one of the same kind
 * is compared word by word (one change if it is mostly new), and other Blocks
 * are removed and added whole, the new after the old. What the User wrote
 * stays until they accept a change to it. `container` must hold no Proposals.
 * Call it in a Yjs transaction.
 */
export function proposeBlocks(
  container: Y.XmlElement,
  content: readonly NodeJSON[],
  request: string,
  newId: () => string,
): Written {
  const { proposals, next } = proposalsOf(request, newId);
  const notes: CitationNotes = { removed: new Set(), added: new Set() };
  const old = container.toArray().filter((child) => child instanceof Y.XmlElement);
  const hunks = hunksOf(diffSteps(old.map(elementKey), content.map(jsonKey)), () => false);

  /** A Block of the new Answer, added whole. */
  const addedBlock = (node: NodeJSON, ref: ProposalRef) => {
    for (const key of citationsIn(node)) notes.added.add(key);
    const element = createElement(node);
    setProposal(element, { ...refOf(ref), kind: "insert" });
    return element;
  };
  /** A Block of the Mind, removed whole. */
  const removeBlock = (element: Y.XmlElement, ref: ProposalRef) => {
    for (const key of citationsIn(element)) notes.removed.add(key);
    setProposal(element, { ...refOf(ref), kind: "delete" });
  };

  // Planned in order, so the Proposals are numbered as they read; written from the end.
  const plans = hunks.map((hunk) => {
    const removed = old.slice(hunk.oldFrom, hunk.oldTo);
    const added = content.slice(hunk.newFrom, hunk.newTo);
    const pairs = removed.slice(0, added.length).map((element, index) => {
      const node = added[index] as NodeJSON;
      const tokens = comparable(element, node)
        ? tokensIn(element, 0, sizeOf(element), (child, attributes) => ({
            key: answerNodeKey(child.nodeName, attributes),
          }))
        : null;
      if (tokens) {
        const segment: Segment = {
          element,
          from: 0,
          to: sizeOf(element),
          whole: true,
          shown: true,
          tokens,
          start: 0,
          end: sizeOf(element),
          kind: kindOf(element),
        };
        return { kind: "words" as const, segment, tokens: jsonTokens(node.content ?? []) };
      }
      return { kind: "whole" as const, element, node, ref: next(targetText(element)) };
    });
    // Words are compared when written, each change made then: their numbers come after.
    const gone = removed.slice(added.length).map((element) => ({
      element,
      ref: next(targetText(element)),
    }));
    const extra = added.slice(removed.length).map((node) => ({ node, ref: next("") }));
    return { hunk, pairs, gone, extra };
  });

  for (const { hunk, pairs, gone, extra } of plans.reverse()) {
    // New Blocks go after the last Block this place removes, else after the one before it.
    const anchor = old[hunk.oldTo - 1] ?? null;
    if (extra.length > 0) {
      const elements = extra.map(({ node, ref }) => addedBlock(node, ref));
      if (anchor) addAfter(anchor, elements);
      else container.insert(0, elements);
    }
    for (const { element, ref } of gone) removeBlock(element, ref);
    for (const pair of [...pairs].reverse()) {
      if (pair.kind === "words") {
        writeHunks(pair.segment, pair.tokens, next, newId, notes, true);
      } else {
        addAfter(pair.element, [addedBlock(pair.node, pair.ref)]);
        removeBlock(pair.element, pair.ref);
      }
    }
  }
  return { proposals, droppedCitations: droppedOf(notes) };
}

// ---------------------------------------------------------------------------
// Making a change again

/** The Block-wide Proposal an element is in: its own `proposal` attribute, or one of a Block around it. */
function wholeProposal(element: Y.XmlElement): ProposalAttribute | null {
  for (let at: unknown = element; at instanceof Y.XmlElement; at = at.parent) {
    const attribute = readProposalAttribute(at.getAttribute(PROPOSAL_ATTRIBUTE));
    if (attribute) return attribute;
  }
  return null;
}

/**
 * Rejects the Proposal `id`, in the caller's Yjs transaction, and says where
 * the text it covered is now: from where it starts to the end of the last
 * word it removes, the User's words among them included, for the request to
 * be made again from that text. Null when no paragraph or heading holds
 * text of it (it only added a Block), and nothing is changed then.
 */
export function takeBackForRegenerate(
  blocks: Y.XmlFragment,
  id: string,
): { from: TextPlace; to: TextPlace } | null {
  let from: TextPlace | null = null;
  let to: TextPlace | null = null;
  const isIt = (value: unknown) => readProposalRef(value)?.id === id;
  for (const { element } of textBlocks(blocks)) {
    const blockId = stringOf(element.getAttribute(BLOCK_ID_ATTRIBUTE));
    const whole = wholeProposal(element);
    if (whole?.id === id) {
      if (whole.kind === "delete") {
        from ??= { blockId, offset: 0 };
        to = { blockId, offset: sizeOf(element) };
      }
      continue;
    }
    /** Where each part will be once the Proposal is rejected: what it adds won't be there. */
    let offset = 0;
    for (const child of layout(element)) {
      if (child.node instanceof Y.XmlText) {
        for (const run of child.node.toDelta() as Run[]) {
          const length = typeof run.insert === "string" ? run.insert.length : 1;
          const adds = isIt(run.attributes?.[PROPOSAL_INSERT_MARK]);
          const removes = isIt(run.attributes?.[PROPOSAL_DELETE_MARK]);
          if (adds || removes) from ??= { blockId, offset };
          if (adds) continue;
          offset += length;
          if (removes) to = { blockId, offset };
        }
        continue;
      }
      const attribute = readProposalAttribute(child.node.getAttribute(PROPOSAL_ATTRIBUTE));
      const mine = attribute?.id === id;
      if (mine) from ??= { blockId, offset };
      if (mine && attribute?.kind === "insert") continue;
      offset += 1;
      if (mine) to = { blockId, offset };
    }
  }
  if (!from) return null;
  resolveProposals(blocks, (ref) => ref.id === id, "reject");
  return { from, to: to ?? from };
}
