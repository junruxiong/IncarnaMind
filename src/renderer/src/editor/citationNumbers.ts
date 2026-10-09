import { Extension, getChangedRanges } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin, PluginKey, type Transaction } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { BLOCK_ID_ATTRIBUTE, CITATION_NODE, type CitationAttributes } from "../../../core/api";

/** A Citation in the Mind, with its number: its order within its Block, from 1. */
export interface NumberedCitation {
  /** Where the Citation node is. */
  pos: number;
  number: number;
  /** Names the Citation across edits: its Block's ID (or index) and its number. */
  key: string;
  attributes: CitationAttributes;
}

/**
 * Every Citation, top to bottom, numbered within the top-level Block it sits
 * in: an Answer's Citations are 1, 2, 3…, and so are those copied into a Note.
 */
export function numberCitations(doc: ProseMirrorNode): NumberedCitation[] {
  const citations: NumberedCitation[] = [];
  doc.forEach((block, offset, index) => {
    const id = block.attrs[BLOCK_ID_ATTRIBUTE];
    const blockKey = typeof id === "string" ? id : `#${index}`;
    let number = 0;
    block.descendants((node, pos) => {
      if (node.type.name !== CITATION_NODE) return true;
      number += 1;
      citations.push({
        pos: offset + 1 + pos,
        number,
        key: `${blockKey}:${number}`,
        attributes: node.attrs as CitationAttributes,
      });
      return false;
    });
  });
  return citations;
}

const citationsIn = (textBlock: ProseMirrorNode) => {
  const found: ProseMirrorNode[] = [];
  textBlock.forEach((child) => {
    if (child.type.name === CITATION_NODE) found.push(child);
  });
  return found;
};

/**
 * Where the text block is (its position before it, after the change) when a
 * change edits only inside one text block and leaves its Citations as they
 * were, as typing does; null for any other change. Such a change renumbers no
 * Citation, and moves those outside the block only along the document: on
 * screen too, unless the block grows or shrinks.
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
  const were = citationsIn(before.parent);
  const are = citationsIn(after.parent);
  return were.length === are.length && were.every((node, index) => node === are[index])
    ? after.before()
    : null;
}

/**
 * The attribute a Citation's number is drawn in. It is part of the
 * decoration's attributes, not just its spec, so renumbering counts as a
 * change and the Citation's view is updated.
 */
const NUMBER_ATTRIBUTE = "data-citation-number";

/** A Citation's number, from the decorations its node view is given; 0 if it has none. */
export function citationNumberOf(decorations: readonly Decoration[]): number {
  for (const decoration of decorations) {
    const number: unknown = decoration.spec.citationNumber;
    if (typeof number === "number") return number;
  }
  return 0;
}

/** The Citations of a document, numbered, and their numbers as decorations. */
interface Numbers {
  citations: readonly NumberedCitation[];
  decorations: DecorationSet;
}

const numbersKey = new PluginKey<Numbers>("citationNumbers");

function numbersOf(doc: ProseMirrorNode): Numbers {
  const citations = numberCitations(doc);
  const decorations = DecorationSet.create(
    doc,
    citations.map(({ pos, number }) => {
      const size = doc.nodeAt(pos)?.nodeSize ?? 1;
      return Decoration.node(
        pos,
        pos + size,
        { [NUMBER_ATTRIBUTE]: String(number) },
        { citationNumber: number },
      );
    }),
  );
  return { citations, decorations };
}

/**
 * The editor's Citations, numbered (see `numberCitations`), as of its last
 * change: worked out once per change, for every view of them. Empty without
 * `CitationNumbers`.
 */
export function numberedCitations(state: EditorState): readonly NumberedCitation[] {
  return numbersKey.getState(state)?.citations ?? [];
}

/** The plugin behind `CitationNumbers`, which keeps the numbers in the editor's state. */
export function citationNumbersPlugin(): Plugin<Numbers> {
  return new Plugin<Numbers>({
    key: numbersKey,
    state: {
      init: (_config, state) => numbersOf(state.doc),
      apply: (tr, numbers) => {
        if (!tr.docChanged) return numbers;
        if (textBlockEdited(tr) === null) return numbersOf(tr.doc);
        // Typing: the Citations move along, numbered as they were.
        return {
          citations: numbers.citations.map((each) => ({ ...each, pos: tr.mapping.map(each.pos) })),
          decorations: numbers.decorations.map(tr.mapping, tr.doc),
        };
      },
    },
    props: {
      decorations: (state) => numbersKey.getState(state)?.decorations,
    },
  });
}

/**
 * Numbers the Citations (see `numberCitations`) with a node decoration each,
 * which reaches the Citation's view: it shows the number, and redraws when
 * an edit renumbers it.
 */
export const CitationNumbers = Extension.create({
  name: "citationNumbers",

  addProseMirrorPlugins() {
    return [citationNumbersPlugin()];
  },
});
