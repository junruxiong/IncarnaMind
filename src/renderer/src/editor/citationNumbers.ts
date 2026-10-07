import { Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { Plugin, PluginKey } from "@tiptap/pm/state";
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

const numbersKey = new PluginKey<DecorationSet>("citationNumbers");

function decorate(doc: ProseMirrorNode): DecorationSet {
  return DecorationSet.create(
    doc,
    numberCitations(doc).map(({ pos, number }) => {
      const size = doc.nodeAt(pos)?.nodeSize ?? 1;
      return Decoration.node(
        pos,
        pos + size,
        { [NUMBER_ATTRIBUTE]: String(number) },
        { citationNumber: number },
      );
    }),
  );
}

/**
 * Numbers the Citations (see `numberCitations`) with a node decoration each,
 * which reaches the Citation's view: it shows the number, and redraws when
 * an edit renumbers it.
 */
export const CitationNumbers = Extension.create({
  name: "citationNumbers",

  addProseMirrorPlugins() {
    return [
      new Plugin<DecorationSet>({
        key: numbersKey,
        state: {
          init: (_config, state) => decorate(state.doc),
          apply: (tr, numbers) => (tr.docChanged ? decorate(tr.doc) : numbers),
        },
        props: {
          decorations: (state) => numbersKey.getState(state),
        },
      }),
    ];
  },
});
