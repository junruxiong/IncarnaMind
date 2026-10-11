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
import type { CitationForm } from "../../../shared/citationStyles";

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
  /**
   * Whether it is a footnote of its own in an export: each Citation in Numbered, the first of
   * Citations side by side in a note style, none in a style that cites in the text (#289).
   */
  footnote: boolean;
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

/** Whether nothing but spaces stands between two Citations, at these positions, of one text block. */
export function sideBySide(doc: ProseMirrorNode, before: number, after: number): boolean {
  if (doc.resolve(before).parent !== doc.resolve(after).parent) return false;
  let only = true;
  doc.nodesBetween(before + 1, after, (node) => {
    if (node.isText ? (node.text ?? "").trim() !== "" : node.isInline) only = false;
    return only;
  });
  return only;
}

/**
 * Every Citation and every footnote the User wrote, top to bottom, numbered
 * together through the Mind, as they are when exported. How a Citation counts
 * depends on the Mind's citation style (#289): in Numbered, each is a footnote,
 * its reference; in a note style, Citations side by side are one note, one
 * number; in a style that cites in the text, none is a footnote, and Citations
 * are numbered among themselves (their marks in the margin say these numbers),
 * the User's footnotes on their own. A Citation inside a footnote is part of
 * that footnote's text and has no number of its own.
 */
export function numberFootnotes(
  doc: ProseMirrorNode,
  form: CitationForm = "numbered",
): NumberedFootnotes {
  const citations: NumberedCitation[] = [];
  const footnotes: NumberedFootnote[] = [];
  let number = 0;
  let cited = 0;
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
      const at = offset + 1 + pos;
      const previous = citations.at(-1);
      const joined = form === "note" && previous !== undefined && sideBySide(doc, previous.pos, at);
      const footnote = form === "numbered" || (form === "note" && !joined);
      if (footnote) number += 1;
      cited += 1;
      inBlock += 1;
      citations.push({
        pos: at,
        number: form === "in-text" ? cited : number,
        key: `${blockKey}:${inBlock}`,
        attributes: node.attrs as CitationAttributes,
        footnote,
      });
      return false;
    });
  });
  return { citations, footnotes };
}

/** Every Citation, top to bottom, numbered among the Mind's footnotes (see `numberFootnotes`). */
export function numberCitations(
  doc: ProseMirrorNode,
  form: CitationForm = "numbered",
): NumberedCitation[] {
  return numberFootnotes(doc, form).citations;
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
  /** How the Mind's citation style counts Citations (see `numberFootnotes`). */
  form: CitationForm;
}

const numbersKey = new PluginKey<Numbers>("citationNumbers");

/** A transaction with this meta redraws every footnote's marker (see `redrawFootnotes`). */
const REDRAW = "redrawFootnotes";

/** A transaction with this meta numbers again for another citation style (`setCitationForm`). */
const FORM = "citationForm";

function numbersOf(doc: ProseMirrorNode, generation: number, form: CitationForm): Numbers {
  const { citations, footnotes } = numberFootnotes(doc, form);
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
  return { citations, footnotes, decorations, generation, form };
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

/** Numbers the Citations and footnotes again for the Mind's citation style, when it changes. */
export function setCitationForm(view: EditorView, form: CitationForm): void {
  if (numbersKey.getState(view.state)?.form === form) return;
  view.dispatch(view.state.tr.setMeta(FORM, form));
}

/** How many Citations a text block holds. */
const citationsIn = (block: ProseMirrorNode | null) => {
  let count = 0;
  block?.forEach((child) => {
    if (child.type.name === CITATION_NODE) count++;
  });
  return count;
};

const isCitationForm = (value: unknown): value is CitationForm =>
  value === "numbered" || value === "note" || value === "in-text";

/**
 * The plugin behind `CitationNumbers`, which keeps the numbers in the editor's state, for
 * the citation style's form as it starts (`form`) and as `setCitationForm` changes it.
 */
export function citationNumbersPlugin(
  form: () => CitationForm = () => "numbered",
): Plugin<Numbers> {
  return new Plugin<Numbers>({
    key: numbersKey,
    state: {
      init: (_config, state) => numbersOf(state.doc, 0, form()),
      apply: (tr, numbers) => {
        const changedForm: unknown = tr.getMeta(FORM);
        if (isCitationForm(changedForm)) {
          return numbersOf(tr.doc, numbers.generation + 1, changedForm);
        }
        if (tr.getMeta(REDRAW)) return numbersOf(tr.doc, numbers.generation + 1, numbers.form);
        if (!tr.docChanged) return numbers;
        const typedIn = textBlockEdited(tr);
        // In a note style, typing between two Citations can join or part their note.
        if (
          typedIn === null ||
          (numbers.form === "note" && citationsIn(tr.doc.nodeAt(typedIn)) > 1)
        ) {
          return numbersOf(tr.doc, numbers.generation, numbers.form);
        }
        // Typing: the Citations and footnotes move along, numbered as they were.
        return {
          citations: numbers.citations.map((each) => ({ ...each, pos: tr.mapping.map(each.pos) })),
          footnotes: numbers.footnotes.map((each) => ({ ...each, pos: tr.mapping.map(each.pos) })),
          decorations: numbers.decorations.map(tr.mapping, tr.doc),
          generation: numbers.generation,
          form: numbers.form,
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
 * redraws when an edit renumbers it. `form` says how the Mind's citation style
 * counts Citations when the editor opens.
 */
export const CitationNumbers = Extension.create<{ form: () => CitationForm }>({
  name: "citationNumbers",

  addOptions() {
    return { form: () => "numbered" };
  },

  addProseMirrorPlugins() {
    return [citationNumbersPlugin(this.options.form)];
  },
});
