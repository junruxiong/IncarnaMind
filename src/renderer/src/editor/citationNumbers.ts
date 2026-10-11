import { Extension, getChangedRanges } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin, PluginKey, type Transaction } from "@tiptap/pm/state";
import { Decoration, DecorationSet, type EditorView } from "@tiptap/pm/view";
import {
  BLOCK_ID_ATTRIBUTE,
  CITATION_NODE,
  type CitationAttributes,
  FOOTNOTE_NODE,
} from "../../../core/api";

/**
 * A Citation in the Mind, with its number: its place among the Mind's
 * footnotes (see `numberFootnotes`), from 1.
 */
export interface NumberedCitation {
  /** Where the Citation node is. */
  pos: number;
  number: number;
  /** Names the Citation across edits: its Block's ID (or index) and its order within the Block. */
  key: string;
  attributes: CitationAttributes;
}

/** A footnote the User wrote, with its number among the Mind's footnotes, from 1. */
export interface NumberedFootnote {
  /** Where the footnote node is. */
  pos: number;
  number: number;
  /** Its stable ID (`BLOCK_ID_ATTRIBUTE`). */
  id: string;
  node: ProseMirrorNode;
}

/** The Mind's footnotes, numbered together: Citations and the User's own. */
export interface NumberedFootnotes {
  citations: NumberedCitation[];
  footnotes: NumberedFootnote[];
}

/**
 * Every Citation and every footnote the User wrote, top to bottom, numbered
 * together through the Mind, as they are when exported: a Citation's
 * footnote is its reference. A Citation inside a footnote is part of that
 * footnote's text and has no number of its own.
 */
export function numberFootnotes(doc: ProseMirrorNode): NumberedFootnotes {
  const citations: NumberedCitation[] = [];
  const footnotes: NumberedFootnote[] = [];
  let number = 0;
  doc.forEach((block, offset, index) => {
    const id = block.attrs[BLOCK_ID_ATTRIBUTE];
    const blockKey = typeof id === "string" ? id : `#${index}`;
    let inBlock = 0;
    block.descendants((node, pos) => {
      const name = node.type.name;
      if (name === FOOTNOTE_NODE) {
        number += 1;
        const footnoteId: unknown = node.attrs[BLOCK_ID_ATTRIBUTE];
        footnotes.push({
          pos: offset + 1 + pos,
          number,
          id: typeof footnoteId === "string" ? footnoteId : `${blockKey}:footnote:${number}`,
          node,
        });
        return false;
      }
      if (name !== CITATION_NODE) return true;
      number += 1;
      inBlock += 1;
      citations.push({
        pos: offset + 1 + pos,
        number,
        key: `${blockKey}:${inBlock}`,
        attributes: node.attrs as CitationAttributes,
      });
      return false;
    });
  });
  return { citations, footnotes };
}

/** Every Citation, top to bottom, numbered among the Mind's footnotes (see `numberFootnotes`). */
export function numberCitations(doc: ProseMirrorNode): NumberedCitation[] {
  return numberFootnotes(doc).citations;
}

/** The nodes in a text block that take a footnote's number: Citations and footnotes. */
const numberedIn = (textBlock: ProseMirrorNode) => {
  const found: ProseMirrorNode[] = [];
  textBlock.forEach((child) => {
    const name = child.type.name;
    if (name === CITATION_NODE || name === FOOTNOTE_NODE) found.push(child);
  });
  return found;
};

/**
 * Where the text block is (its position before it, after the change) when a
 * change edits only inside one text block and leaves its Citations and
 * footnotes as they were, as typing does; null for any other change. Such a
 * change renumbers nothing, and moves the Citations and footnotes outside the
 * block only along the document: on screen too, unless the block grows or shrinks.
 */
export function textBlockEdited(tr: Transaction): number | null {
  if (!tr.docChanged) return null;
  const changes = getChangedRanges(tr);
  const [change] = changes;
  if (!change || changes.length > 1) return null;
  const within = (doc: ProseMirrorNode, { from, to }: { from: number; to: number }) => {
    const $from = doc.resolve(from);
    return $from.depth > 0 && $from.parent.isTextblock && to <= $from.end() ? $from : null;
  };
  const before = within(tr.before, change.oldRange);
  const after = within(tr.doc, change.newRange);
  if (!before || !after) return null;
  // An edit beside a Citation keeps the very node; one added, removed or replaced doesn't.
  const were = numberedIn(before.parent);
  const are = numberedIn(after.parent);
  return were.length === are.length && were.every((node, index) => node === are[index])
    ? after.before()
    : null;
}

/**
 * The attribute a Citation's or footnote's number is drawn in. It is part of
 * the decoration's attributes, not just its spec, so renumbering counts as a
 * change and the node's view is updated.
 */
const NUMBER_ATTRIBUTE = "data-citation-number";
const FOOTNOTE_NUMBER_ATTRIBUTE = "data-footnote-number";

/** A Citation's number, from the decorations its node view is given; 0 if it has none. */
export function citationNumberOf(decorations: readonly Decoration[]): number {
  for (const decoration of decorations) {
    const number: unknown = decoration.spec.citationNumber;
    if (typeof number === "number") return number;
  }
  return 0;
}

/** A footnote's number, from the decorations its node view is given; 0 if it has none. */
export function footnoteNumberOf(decorations: readonly Decoration[]): number {
  for (const decoration of decorations) {
    const number: unknown = decoration.spec.footnoteNumber;
    if (typeof number === "number") return number;
  }
  return 0;
}

/** The Citations and footnotes of a document, numbered, and their numbers as decorations. */
interface Numbers extends Readonly<NumberedFootnotes> {
  decorations: DecorationSet;
  /** Raised to redraw every footnote's marker, e.g. in the interface's new language. */
  generation: number;
}

const numbersKey = new PluginKey<Numbers>("citationNumbers");

/** A transaction with this meta redraws every footnote's marker (see `redrawFootnotes`). */
const REDRAW = "redrawFootnotes";

function numbersOf(doc: ProseMirrorNode, generation: number): Numbers {
  const { citations, footnotes } = numberFootnotes(doc);
  const node = (pos: number, attrs: Record<string, string>, spec: Record<string, number>) =>
    Decoration.node(pos, pos + (doc.nodeAt(pos)?.nodeSize ?? 1), attrs, spec);
  const decorations = DecorationSet.create(doc, [
    ...citations.map(({ pos, number }) =>
      node(pos, { [NUMBER_ATTRIBUTE]: String(number) }, { citationNumber: number }),
    ),
    ...footnotes.map(({ pos, number }) =>
      node(
        pos,
        { [FOOTNOTE_NUMBER_ATTRIBUTE]: String(number) },
        { footnoteNumber: number, generation },
      ),
    ),
  ]);
  return { citations, footnotes, decorations, generation };
}

/**
 * The editor's Citations, numbered (see `numberFootnotes`), as of its last
 * change: worked out once per change, for every view of them. Empty without
 * `CitationNumbers`.
 */
export function numberedCitations(state: EditorState): readonly NumberedCitation[] {
  return numbersKey.getState(state)?.citations ?? [];
}

/** The editor's footnotes, numbered (see `numberFootnotes`), as of its last change. */
export function numberedFootnotes(state: EditorState): readonly NumberedFootnote[] {
  return numbersKey.getState(state)?.footnotes ?? [];
}

/** Redraws every footnote's marker, e.g. once the interface's language has changed. */
export function redrawFootnotes(view: EditorView): void {
  view.dispatch(view.state.tr.setMeta(REDRAW, true));
}

/** The plugin behind `CitationNumbers`, which keeps the numbers in the editor's state. */
export function citationNumbersPlugin(): Plugin<Numbers> {
  return new Plugin<Numbers>({
    key: numbersKey,
    state: {
      init: (_config, state) => numbersOf(state.doc, 0),
      apply: (tr, numbers) => {
        if (tr.getMeta(REDRAW)) return numbersOf(tr.doc, numbers.generation + 1);
        if (!tr.docChanged) return numbers;
        if (textBlockEdited(tr) === null) return numbersOf(tr.doc, numbers.generation);
        // Typing: the Citations and footnotes move along, numbered as they were.
        return {
          citations: numbers.citations.map((each) => ({ ...each, pos: tr.mapping.map(each.pos) })),
          footnotes: numbers.footnotes.map((each) => ({ ...each, pos: tr.mapping.map(each.pos) })),
          decorations: numbers.decorations.map(tr.mapping, tr.doc),
          generation: numbers.generation,
        };
      },
    },
    props: {
      decorations: (state) => numbersKey.getState(state)?.decorations,
    },
  });
}

/**
 * Numbers the Citations and footnotes (see `numberFootnotes`) with a node
 * decoration each, which reaches the node's view: it shows the number, and
 * redraws when an edit renumbers it.
 */
export const CitationNumbers = Extension.create({
  name: "citationNumbers",

  addProseMirrorPlugins() {
    return [citationNumbersPlugin()];
  },
});
