/**
 * Writing Proposals into a Mind's Yjs document (see `PROPOSAL_INSERT_MARK` in
 * ../api), in the shape the editor's binding reads (see ../answers/blocks):
 *
 * - `readSelection` reads a stretch of a Mind's text, from one paragraph or
 *   heading to another (`TextPlace`), as the model will read it: Citations
 *   as numbered markers, formulas as $…$, paragraphs apart.
 * - `proposeRewrite` compares that stretch with the model's new text, word by
 *   word (./diff), and marks each place that changed as a Proposal of its
 *   own: the words it removes stay, under `proposalDelete`, and the words it
 *   adds go after them, under `proposalInsert`. Paragraphs the new text adds
 *   or drops are Proposals on whole Blocks.
 * - `proposeAfter` adds new paragraphs under the stretch, each a Proposal.
 *
 * Nothing in the Mind is changed but by marks, attributes and the new words:
 * the User's text stays where and what it was until they accept.
 */
import * as Y from "yjs";
import { createElement, type NodeJSON } from "../answers/blocks";
import {
  BLOCK_ID_ATTRIBUTE,
  CITATION_NODE,
  PROPOSAL_ATTRIBUTE,
  PROPOSAL_DELETE_MARK,
  PROPOSAL_INSERT_MARK,
  type ProposalAttribute,
  type ProposalRef,
  type TextPlace,
} from "../api";
import { diffSteps, hunksOf, splitWords } from "./diff";

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
/** How the editor reads an inline node in a text (see `TextPlace`). */
const NODE_CHARACTER = "\uFFFC";

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
  /** Its text, or for an inline node its name and attributes. */
  text?: string;
  attributes?: Attributes;
  node?: { name: string; attributes: Attributes };
}

/** The part of one paragraph or heading the selection covers. */
interface Segment {
  element: Y.XmlElement;
  from: number;
  to: number;
  /** The selection covers all of its text. */
  whole: boolean;
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
  /** The selected text as the editor reads it: an inline node as U+FFFC, a line per paragraph. */
  plain: string;
  /** The selected text as the model reads it: markers for Citations, paragraphs apart. */
  text: string;
  /** The whole paragraphs it is in, the same way. */
  context: string;
  /** How many characters it holds, for the model's window. */
  characters: number;
}

export type ReadResult =
  | { ok: true; selection: Selection }
  | { ok: false; reason: "changed" | "has-proposals" | "busy" };

const stringOf = (value: unknown) => (typeof value === "string" ? value : "");

/** The paragraphs and headings of the Mind, in order, at any depth, with whether an Answer being written holds them. */
function textBlocks(blocks: Y.XmlFragment): { element: Y.XmlElement; busy: boolean }[] {
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
function layout(element: Y.XmlElement) {
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
 * after a character no text holds (`n`: a Citation's marker).
 */
const keyOf = {
  text: (text: string) => text,
  citation: (n: number) => `\u0000c${n}`,
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
  const segments: Segment[] = [];
  const plain: string[] = [];
  const text: string[] = [];
  const context: string[] = [];
  let characters = 0;
  for (const [index, { element }] of chosen.entries()) {
    const size = sizeOf(element);
    const start = index === 0 ? from.offset : 0;
    const end = index === chosen.length - 1 ? to.offset : size;
    if (start < 0 || end > size || start > end) return { ok: false, reason: "changed" };
    const tokens: OldToken[] = [];
    const words: string[] = [];
    const seen: string[] = [];
    for (const child of layout(element)) {
      if (child.node instanceof Y.XmlText) {
        let at = child.start;
        for (const run of child.node.toDelta() as Run[]) {
          if (typeof run.insert !== "string") continue;
          const runFrom = at;
          at += run.insert.length;
          const cut = run.insert.slice(Math.max(0, start - runFrom), Math.max(0, end - runFrom));
          if (at <= start || runFrom >= end || cut === "") continue;
          if (hasProposal(run.attributes)) return { ok: false, reason: "has-proposals" };
          let tokenFrom = Math.max(start, runFrom);
          for (const word of splitWords(cut)) {
            tokens.push({
              key: keyOf.text(word),
              from: tokenFrom,
              to: tokenFrom + word.length,
              text: word,
              attributes: run.attributes ?? {},
            });
            tokenFrom += word.length;
          }
          seen.push(cut);
          words.push(cut);
        }
        continue;
      }
      if (child.start < start || child.start >= end) continue;
      const attributes = child.node.getAttributes() as Attributes;
      if (PROPOSAL_ATTRIBUTE in attributes) return { ok: false, reason: "has-proposals" };
      const name = child.node.nodeName;
      let key: string;
      let shown: string;
      if (name === CITATION_NODE) {
        citations.push(attributes);
        key = keyOf.citation(citations.length);
        shown = `[${citations.length}]`;
      } else if (name === MATH) {
        key = keyOf.math(stringOf(attributes.latex));
        shown = `$${stringOf(attributes.latex)}$`;
      } else if (name === BREAK) {
        key = keyOf.break();
        shown = "\n";
      } else {
        // Nothing the model can keep: it comes back without it, for the User to review.
        key = keyOf.node(name, tokens.length);
        shown = "";
      }
      tokens.push({ key, from: child.start, to: child.start + 1, node: { name, attributes } });
      seen.push(NODE_CHARACTER);
      words.push(shown);
    }
    // The spaces at either end stay as they are.
    let lo = 0;
    let hi = tokens.length;
    while (lo < hi && /^\s+$/.test(tokens[lo]?.text ?? "x")) lo++;
    while (hi > lo && /^\s+$/.test(tokens[hi - 1]?.text ?? "x")) hi--;
    const kept = tokens.slice(lo, hi);
    segments.push({
      element,
      from: start,
      to: end,
      whole: start === 0 && end === size,
      tokens: kept,
      start: kept[0]?.from ?? start,
      end: kept.at(-1)?.to ?? start,
    });
    plain.push(seen.join(""));
    const shownText = words.join("").trim();
    text.push(shownText);
    context.push(blockText(element));
    characters += shownText.length;
  }
  return {
    ok: true,
    selection: {
      segments,
      citations,
      plain: plain.join("\n"),
      text: text.filter((each) => each !== "").join("\n\n"),
      context: context.join("\n\n"),
      characters,
    },
  };
}

/** A paragraph's whole text as context for the model: inline nodes as they read, Citations as "[·]". */
function blockText(element: Y.XmlElement): string {
  return layout(element)
    .map(({ node }) => {
      if (node instanceof Y.XmlText) {
        return (node.toDelta() as Run[]).map((run) => stringOf(run.insert)).join("");
      }
      if (node.nodeName === MATH) return `$${stringOf(node.getAttribute("latex"))}$`;
      if (node.nodeName === BREAK) return "\n";
      return node.nodeName === CITATION_NODE ? "[·]" : "";
    })
    .join("")
    .trim();
}

// ---------------------------------------------------------------------------
// The model's text

/** A token of the new text: words, or an inline node of the selection kept (or another). */
interface NewToken {
  key: string;
  text?: string;
  node?: { name: string; attributes: Attributes };
}

const MARKER = /\[(\d+)\]|\$([^$\n]+)\$|\n/g;

/**
 * The new text's paragraphs as tokens. A marker "[n]" is Citation n of the
 * selection, with `fresh` attributes when its check came out differently
 * now; a marker for none it had is left out. A formula the selection had is
 * kept; another stays text.
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
        if (match[0] === "\n") {
          tokens.push({ key: keyOf.break(), node: { name: BREAK, attributes: {} } });
        } else if (match[1] !== undefined) {
          const n = Number(match[1]);
          const attributes = fresh.get(n) ?? selection.citations[n - 1];
          if (attributes) {
            tokens.push({
              key: `${keyOf.citation(n)}${fresh.has(n) ? "*" : ""}`,
              node: { name: CITATION_NODE, attributes },
            });
          }
          // Spaces left where an unknown marker was are one.
          else if (/\s$/.test(paragraph.slice(0, match.index)) && /^\s/.test(paragraph.slice(at)))
            at++;
        } else if (match[2] !== undefined && formulas.has(match[2])) {
          tokens.push({
            key: keyOf.math(match[2]),
            node: { name: MATH, attributes: { latex: match[2] } },
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
      child.node.format(start - child.start, end - start, {
        [PROPOSAL_DELETE_MARK]: { id: ref.id, request: ref.request },
      });
    } else {
      setProposal(child.node, { ...ref, kind: "delete" });
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

/** A Citation, formula or line break the Proposal adds. */
function nodeElement(node: NonNullable<NewToken["node"]>, ref: ProposalRef): Y.XmlElement {
  return createElement({
    type: node.name,
    attrs: {
      ...withoutProposalAttribute(node.attributes),
      [PROPOSAL_ATTRIBUTE]: { ...ref, kind: "insert" } satisfies ProposalAttribute,
    },
  });
}

function withoutProposalAttribute(attributes: Attributes): Attributes {
  const { [PROPOSAL_ATTRIBUTE]: _proposal, ...rest } = attributes;
  return rest;
}

/** Adds the new tokens at `at` in a Block's text, as the Proposal's words. */
function insertTokens(
  segment: Segment,
  at: number,
  tokens: readonly NewToken[],
  marks: Attributes,
  ref: ProposalRef,
): void {
  const inserted = { ...marks, [PROPOSAL_INSERT_MARK]: { id: ref.id, request: ref.request } };
  let position = at;
  let words = "";
  const flush = () => {
    if (words === "") return;
    insertText(segment.element, position, words, inserted);
    position += words.length;
    words = "";
  };
  for (const token of tokens) {
    if (token.node) {
      flush();
      insertNode(segment.element, position, nodeElement(token.node, ref));
      position += 1;
    } else {
      words += token.text ?? "";
    }
  }
  flush();
}

/** A new paragraph, as the Proposal's whole Block, holding these tokens. */
function paragraphOf(tokens: readonly NewToken[], ref: ProposalRef): NodeJSON {
  const content: NodeJSON[] = [];
  for (const token of tokens) {
    if (token.node) {
      content.push({
        type: token.node.name,
        attrs: withoutProposalAttribute(token.node.attributes),
      });
      continue;
    }
    const last = content.at(-1);
    if (last?.type === "text") last.text = `${last.text ?? ""}${token.text ?? ""}`;
    else content.push({ type: "text", text: token.text ?? "" });
  }
  return {
    type: "paragraph",
    attrs: { [PROPOSAL_ATTRIBUTE]: { ...ref, kind: "insert" } satisfies ProposalAttribute },
    content,
  };
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

/** The selection's Citation a token is, by its marker; null for any other token. */
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
  const proposals: ProposalRef[] = [];
  const next = () => {
    const ref = { id: newId(), request };
    proposals.push(ref);
    return ref;
  };
  /** The Citations (by marker) the Proposals remove, and those they add back. */
  const removedCitations = new Set<string>();
  const addedCitations = new Set<string>();
  const note = (into: Set<string>, token: { key: string }) => {
    const marker = markerOf(token);
    if (marker) into.add(marker);
  };
  const pairs = Math.min(paragraphs.length, selection.segments.length);

  for (let index = 0; index < pairs; index++) {
    const segment = selection.segments[index] as Segment;
    const tokens = paragraphs[index] as NewToken[];
    const steps = diffSteps(
      segment.tokens.map((token) => token.key),
      tokens.map((token) => token.key),
    );
    const hunks = hunksOf(steps, joinsAcross(segment.tokens));
    const refs = hunks.map(() => next());
    // From the end, so the places read before each are still right.
    for (let at = hunks.length - 1; at >= 0; at--) {
      const hunk = hunks[at] as (typeof hunks)[number];
      const ref = refs[at] as ProposalRef;
      const removed = segment.tokens.slice(hunk.oldFrom, hunk.oldTo);
      const added = tokens.slice(hunk.newFrom, hunk.newTo);
      const end = removed.at(-1)?.to ?? segment.tokens[hunk.oldFrom]?.from ?? segment.end;
      const firstWords = removed.find((token) => token.text !== undefined);
      const marks = firstWords
        ? withoutProposals(firstWords.attributes)
        : besideMarks(segment, end);
      insertTokens(segment, end, added, marks, ref);
      if (removed.length > 0) markRemoved(segment.element, removed[0]?.from ?? end, end, ref);
      for (const token of removed) note(removedCitations, token);
      for (const token of added) note(addedCitations, token);
    }
  }

  // Fewer paragraphs: the rest of the selection goes.
  for (const segment of selection.segments.slice(pairs)) {
    if (segment.tokens.length === 0) continue;
    const ref = next();
    if (segment.whole) {
      setProposal(removedWith(segment.element), { ...ref, kind: "delete" });
    } else {
      markRemoved(segment.element, segment.start, segment.end, ref);
    }
    for (const token of segment.tokens) note(removedCitations, token);
  }

  // More paragraphs: each a new Block after the last one paired.
  const anchor = selection.segments[Math.max(0, pairs - 1)];
  if (anchor && paragraphs.length > pairs) {
    const added = paragraphs.slice(pairs).map((tokens) => {
      for (const token of tokens) note(addedCitations, token);
      return createElement(paragraphOf(tokens, next()));
    });
    addAfter(anchor.element, added);
  }

  return {
    proposals,
    droppedCitations: [...removedCitations].filter((marker) => !addedCitations.has(marker)).length,
  };
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
  const proposals: ProposalRef[] = [];
  const paragraphs = text
    .replace(/\s*\[\d+\]/g, "")
    .split(/\n[ \t]*\n+/)
    .map((paragraph) => paragraph.trim())
    .filter((paragraph) => paragraph !== "");
  const added = paragraphs.map((paragraph) => {
    const ref = { id: newId(), request };
    proposals.push(ref);
    return createElement(
      paragraphOf(
        splitWords(paragraph).map((word) => ({ key: keyOf.text(word), text: word })),
        ref,
      ),
    );
  });
  addAfter(top, added);
  return { proposals, droppedCitations: 0 };
}
