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
  /** The selected text as the model reads it: markers for Citations, paragraphs apart. */
  text: string;
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
    segments.push({
      element,
      from: start,
      to: end,
      whole: start === 0 && end === size,
      shown: shownText !== "",
      tokens: kept,
      start: kept[0]?.from ?? start,
      end: kept.at(-1)?.to ?? start,
    });
    plain.push(tokens.map((token) => token.text ?? NODE_CHARACTER).join(""));
    text.push(shownText);
    context.push(blockText(element));
    characters += shownText.length;
  }
  const firstBlock = chosen[0]?.element;
  const lastBlock = chosen.at(-1)?.element;
  return {
    ok: true,
    selection: {
      segments,
      citations,
      footnotes,
      plain: plain.join("\n"),
      text: text.filter((each) => each !== "").join("\n\n"),
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
 * The new text's paragraphs as tokens. A marker "[n]" is Citation n of the
 * selection, with `fresh` attributes when its check came out differently
 * now, and "[^n]" is its footnote n; a marker for none it had is left out.
 * A formula the selection had is kept; another stays text.
 */
function newParagraphs(
  text: string,
  selection: Selection,
  fresh: ReadonlyMap<number, Attributes>,
): NewToken[][] {
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
  return text
    .split(/\n[ \t]*\n+/)
    .map((paragraph) => paragraph.trim())
    .filter((paragraph) => paragraph !== "")
    .map((paragraph) => {
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
    });
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
function setProposal(element: Y.XmlElement, value: ProposalAttribute): void {
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

/** A new paragraph, as the Proposal's whole Block, holding these tokens. */
function paragraphOf(
  tokens: readonly NewToken[],
  ref: ProposalRef,
  newId: () => string,
): Y.XmlElement {
  const paragraph = createElement({
    type: "paragraph",
    attrs: { [PROPOSAL_ATTRIBUTE]: { ...refOf(ref), kind: "insert" } satisfies ProposalAttribute },
  });
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
  paragraph.insert(0, children);
  return paragraph;
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

/** Adds `elements` after `anchor`, in its parent. */
function addAfter(anchor: Y.XmlElement, elements: Y.XmlElement[]): void {
  const parent = anchor.parent;
  if (!(parent instanceof Y.XmlElement || parent instanceof Y.XmlFragment)) return;
  const index = parent.toArray().indexOf(anchor);
  parent.insert(index + 1, elements);
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

/**
 * Writes the model's `text` for the selection as Proposals of `request`:
 * each paragraph against the selection's in order, word by word; a
 * paragraph more is a new Block after the last, and one fewer removes the
 * selection's (its whole Block, if all of it was selected). `fresh`:
 * Citations whose check came out differently now, by marker, with their new
 * attributes: one the text keeps is replaced by the checked copy. Call it
 * in a Yjs transaction.
 */
export function proposeRewrite(
  selection: Selection,
  text: string,
  request: string,
  newId: () => string,
  fresh: ReadonlyMap<number, Attributes> = new Map(),
): Written {
  const paragraphs = newParagraphs(text, selection, fresh);
  const { proposals, next } = proposalsOf(request, newId);
  const notes: CitationNotes = { removed: new Set(), added: new Set() };
  // Only the paragraphs the model was shown: an empty one between two isn't in its text.
  const segments = selection.segments.filter((segment) => segment.shown);
  const pairs = Math.min(paragraphs.length, segments.length);

  for (let index = 0; index < pairs; index++) {
    writeHunks(segments[index] as Segment, paragraphs[index] as NewToken[], next, newId, notes);
  }

  // Fewer paragraphs: the rest of the selection goes.
  for (const segment of segments.slice(pairs)) {
    if (segment.tokens.length === 0) continue;
    if (segment.whole) {
      const removed = removedWith(segment.element);
      setProposal(removed, { ...refOf(next(targetText(removed))), kind: "delete" });
    } else {
      markRemoved(segment.element, segment.start, segment.end, next(targetOf(segment.tokens)));
    }
    for (const token of segment.tokens) noteCitation(notes.removed, token);
  }

  // More paragraphs: each a new Block after the last one paired.
  const anchor = segments[Math.max(0, pairs - 1)];
  if (anchor && paragraphs.length > pairs) {
    const added = paragraphs.slice(pairs).map((tokens) => {
      for (const token of tokens) noteCitation(notes.added, token);
      return paragraphOf(tokens, next(""), newId);
    });
    addAfter(anchor.element, added);
  }

  return { proposals, droppedCitations: droppedOf(notes) };
}

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
