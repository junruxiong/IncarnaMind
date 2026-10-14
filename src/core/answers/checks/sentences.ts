/**
 * The User's sentences in a Mind's Yjs document, for their checks (#96):
 * which sentences a selection touches, and the `CLAIM_CHECK_MARK` that
 * holds each one's result on its text. Offsets are `TextPlace` offsets: an
 * inline node counts as one character.
 */
import * as Y from "yjs";
import { readClaimCheck } from "../../../shared/claimChecks";
import { INLINE_NODE, sentenceSpans } from "../../../shared/sentences";
import {
  BLOCK_ID_ATTRIBUTE,
  CITATION_NODE,
  CLAIM_CHECK_MARK,
  type ClaimCheckAttributes,
  PROPOSAL_ATTRIBUTE,
  PROPOSAL_DELETE_MARK,
  PROPOSAL_INSERT_MARK,
  type TextPlace,
} from "../../api";
import { layout, textBlocks } from "../../proposals/write";

const BREAK = "hardBreak";

type Attributes = Record<string, unknown>;

interface Run {
  insert: unknown;
  attributes?: Attributes;
}

/** A sentence the selection touches, to check. */
export interface SelectedSentence {
  element: Y.XmlElement;
  blockId: string;
  from: number;
  to: number;
  /** As checks read it: an inline node as U+FFFC, a line break as "\n". */
  text: string;
  /** Its paragraph's text, for what its words refer to. */
  context: string;
  /** What its Citations quote. */
  cited: { documentName: string; quote: string }[];
}

export type SelectedSentences =
  | { ok: true; sentences: SelectedSentence[] }
  | { ok: false; reason: "nothing" | "has-proposals" };

/** A Block's text as checks read it. */
function textOf(element: Y.XmlElement): string {
  return layout(element)
    .map(({ node }) =>
      node instanceof Y.XmlText
        ? (node.toDelta() as Run[])
            .map((run) => (typeof run.insert === "string" ? run.insert : INLINE_NODE))
            .join("")
        : node.nodeName === BREAK
          ? "\n"
          : INLINE_NODE,
    )
    .join("");
}

/** Whether `from`..`to` of a Block holds a change still waiting. */
function holdsProposal(element: Y.XmlElement, from: number, to: number): boolean {
  for (let at: unknown = element; at instanceof Y.XmlElement; at = at.parent) {
    if (at.getAttribute(PROPOSAL_ATTRIBUTE)) return true;
  }
  for (const { node, start, size } of layout(element)) {
    if (start + size <= from || start >= to) continue;
    if (node instanceof Y.XmlElement) {
      if (node.getAttribute(PROPOSAL_ATTRIBUTE)) return true;
      continue;
    }
    let offset = start;
    for (const run of node.toDelta() as Run[]) {
      const length = typeof run.insert === "string" ? run.insert.length : 1;
      const overlaps = offset < to && offset + length > from;
      if (
        overlaps &&
        run.attributes &&
        (PROPOSAL_INSERT_MARK in run.attributes || PROPOSAL_DELETE_MARK in run.attributes)
      ) {
        return true;
      }
      offset += length;
    }
  }
  return false;
}

/** What the Citations in `from`..`to` of a Block quote. */
function citedIn(element: Y.XmlElement, from: number, to: number) {
  return layout(element).flatMap(({ node, start }) => {
    if (start < from || start >= to || !(node instanceof Y.XmlElement)) return [];
    if (node.nodeName !== CITATION_NODE) return [];
    const { documentName, quote } = node.getAttributes() as Attributes;
    return typeof quote === "string" && quote !== ""
      ? [{ documentName: typeof documentName === "string" ? documentName : "", quote }]
      : [];
  });
}

/**
 * The sentences from `from` to `to`, each whole: a sentence the selection
 * only touches is checked all the same. Headings aren't checked.
 */
export function selectedSentences(
  blocks: Y.XmlFragment,
  from: TextPlace,
  to: TextPlace,
): SelectedSentences {
  const all = textBlocks(blocks);
  const idOf = (element: Y.XmlElement) => element.getAttribute(BLOCK_ID_ATTRIBUTE);
  const first = all.findIndex(({ element }) => idOf(element) === from.blockId);
  const last = all.findIndex(({ element }) => idOf(element) === to.blockId);
  if (first < 0 || last < first) return { ok: false, reason: "nothing" };
  const sentences: SelectedSentence[] = [];
  for (const [index, { element, busy }] of all.slice(first, last + 1).entries()) {
    if (busy || element.nodeName !== "paragraph") continue;
    const blockId = idOf(element);
    if (typeof blockId !== "string") continue;
    const text = textOf(element);
    const start = index === 0 ? from.offset : 0;
    const end = index === last - first ? to.offset : text.length;
    for (const span of sentenceSpans(text)) {
      const touched =
        start === end ? span.from <= start && start <= span.to : span.from < end && span.to > start;
      if (!touched) continue;
      if (holdsProposal(element, span.from, span.to)) return { ok: false, reason: "has-proposals" };
      sentences.push({
        element,
        blockId,
        from: span.from,
        to: span.to,
        text: text.slice(span.from, span.to),
        context: text,
        cited: citedIn(element, span.from, span.to),
      });
    }
  }
  return sentences.length > 0 ? { ok: true, sentences } : { ok: false, reason: "nothing" };
}

/** Every sentence of the Mind's paragraphs, in order, but those holding a change still waiting. */
export function everySentence(blocks: Y.XmlFragment): SelectedSentence[] {
  return textBlocks(blocks).flatMap(({ element, busy }) => {
    const blockId = element.getAttribute(BLOCK_ID_ATTRIBUTE);
    if (busy || element.nodeName !== "paragraph" || typeof blockId !== "string") return [];
    const text = textOf(element);
    return sentenceSpans(text).flatMap((span) =>
      holdsProposal(element, span.from, span.to)
        ? []
        : [
            {
              element,
              blockId,
              from: span.from,
              to: span.to,
              text: text.slice(span.from, span.to),
              context: text,
              cited: citedIn(element, span.from, span.to),
            },
          ],
    );
  });
}

/** Puts a check's mark on `from`..`to` of a Block's text, in place of any check there. */
export function markSentence(
  element: Y.XmlElement,
  from: number,
  to: number,
  check: ClaimCheckAttributes,
): void {
  for (const { node, start, size } of layout(element)) {
    if (!(node instanceof Y.XmlText)) continue;
    const begin = Math.max(from, start);
    const end = Math.min(to, start + size);
    if (begin >= end) continue;
    node.format(begin - start, end - begin, { [CLAIM_CHECK_MARK]: { ...check } });
  }
}

/** Where a check's mark is: its Block and the stretch of text from its first marked character to its last. */
export interface MarkedSentence {
  element: Y.XmlElement;
  from: number;
  to: number;
  check: ClaimCheckAttributes;
}

/** Every check's mark in the Mind, in order. */
export function markedSentences(blocks: Y.XmlFragment): MarkedSentence[] {
  const found = new Map<string, MarkedSentence>();
  for (const { element } of textBlocks(blocks)) {
    for (const { node, start } of layout(element)) {
      if (!(node instanceof Y.XmlText)) continue;
      let offset = start;
      for (const run of node.toDelta() as Run[]) {
        const length = typeof run.insert === "string" ? run.insert.length : 1;
        const check = readClaimCheck(run.attributes?.[CLAIM_CHECK_MARK]);
        if (check) {
          const known = found.get(check.id);
          if (known && known.element === element) known.to = offset + length;
          else if (!known)
            found.set(check.id, { element, from: offset, to: offset + length, check });
        }
        offset += length;
      }
    }
  }
  return [...found.values()];
}

/** Changes a check's result where its mark is now; false when its mark has gone (its sentence was deleted). */
export function updateCheck(
  blocks: Y.XmlFragment,
  id: string,
  change: (check: ClaimCheckAttributes) => ClaimCheckAttributes,
): boolean {
  const marked = markedSentences(blocks).find((each) => each.check.id === id);
  if (!marked) return false;
  for (const { node } of layout(marked.element)) {
    if (!(node instanceof Y.XmlText)) continue;
    let offset = 0;
    for (const run of node.toDelta() as Run[]) {
      const length = typeof run.insert === "string" ? run.insert.length : 1;
      const check = readClaimCheck(run.attributes?.[CLAIM_CHECK_MARK]);
      if (check?.id === id)
        node.format(offset, length, { [CLAIM_CHECK_MARK]: { ...change(check) } });
      offset += length;
    }
  }
  return true;
}
