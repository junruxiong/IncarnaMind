import { type Editor, Extension } from "@tiptap/core";
import type { MarkType, Node as ProseMirrorNode } from "@tiptap/pm/model";
import {
  type EditorState,
  Plugin,
  PluginKey,
  type Selection,
  type Transaction,
} from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { ySyncPluginKey, yUndoPluginKey } from "@tiptap/y-tiptap";
import type * as Y from "yjs";
import {
  BLOCK_ID_ATTRIBUTE,
  CITATION_NODE,
  PROPOSAL_ATTRIBUTE,
  PROPOSAL_DELETE_MARK,
  PROPOSAL_INSERT_MARK,
} from "../../../core/api";
import {
  type ResolvedProposal,
  readProposalAttribute,
  readProposalRef,
  resolveProposals,
} from "../../../shared/proposals";

/** A Proposal waiting in the Mind, as the editor shows it. */
export interface PendingProposal {
  id: string;
  request: string;
  /** Where it is: from the start of its first part to the end of its last. */
  from: number;
  to: number;
  /**
   * Its parts: runs of the words it removes and adds, and each Block or
   * inline node it removes or adds (`node`).
   */
  ranges: { from: number; to: number; node?: "insert" | "delete" }[];
  /** The text it removes and adds, an inline node as U+FFFC; a whole Block's text for a Block. */
  removed: string;
  added: string;
  /** "insert" or "delete" for a whole Block; null for words in a text. */
  block: "insert" | "delete" | null;
  /** Citations it removes and doesn't add back. */
  droppedCitations: number;
}

interface ReviewState {
  proposals: readonly PendingProposal[];
  /** The Proposal under the pointer, by id. */
  hovered: string | null;
  /** Text waiting for Edit with AI to come back, or the selection the composer is editing. */
  target: { from: number; to: number } | null;
  decorations: DecorationSet;
}

type Meta = { hovered: string | null } | { target: { from: number; to: number } | null };

export const proposalReviewKey = new PluginKey<ReviewState>("proposalReview");

/** The text of an inline node, as the editor reads it in a selection. */
const NODE_CHARACTER = "￼";

/** The key that tells one Citation from another, to know whether a Proposal adds it back. */
const citationKey = (node: ProseMirrorNode) =>
  `${String(node.attrs.documentId)}\u0000${String(node.attrs.quote)}`;

/**
 * Every Proposal in the document, in the order they start. A Citation a
 * Proposal removes and doesn't add back counts as dropped; one the same
 * request adds elsewhere (moved) doesn't.
 */
export function proposalsIn(doc: ProseMirrorNode): PendingProposal[] {
  const found = new Map<string, PendingProposal>();
  const removedCitations = new Map<string, string[]>();
  const addedCitations = new Map<string, Set<string>>();
  const entry = (ref: { id: string; request: string }) => {
    let proposal = found.get(ref.id);
    if (!proposal) {
      proposal = {
        id: ref.id,
        request: ref.request,
        from: Number.POSITIVE_INFINITY,
        to: 0,
        ranges: [],
        removed: "",
        added: "",
        block: null,
        droppedCitations: 0,
      };
      found.set(ref.id, proposal);
    }
    return proposal;
  };
  const extend = (
    proposal: PendingProposal,
    from: number,
    to: number,
    node?: "insert" | "delete",
  ) => {
    const last = proposal.ranges.at(-1);
    if (!node && last && !last.node && last.to === from) last.to = to;
    else proposal.ranges.push(node ? { from, to, node } : { from, to });
    proposal.from = Math.min(proposal.from, from);
    proposal.to = Math.max(proposal.to, to);
  };
  const noteCitations = (
    node: ProseMirrorNode,
    request: string,
    kind: "insert" | "delete",
    id: string,
  ) => {
    const each = (citation: ProseMirrorNode) => {
      if (citation.type.name !== CITATION_NODE) return;
      if (kind === "delete") {
        const list = removedCitations.get(id) ?? [];
        list.push(`${request}\u0000${citationKey(citation)}`);
        removedCitations.set(id, list);
      } else {
        const set = addedCitations.get(request) ?? new Set<string>();
        set.add(citationKey(citation));
        addedCitations.set(request, set);
      }
    };
    each(node);
    node.descendants((child) => {
      each(child);
    });
  };

  doc.descendants((node, pos) => {
    const attribute = readProposalAttribute(node.attrs[PROPOSAL_ATTRIBUTE]);
    if (attribute) {
      const proposal = entry(attribute);
      extend(proposal, pos, pos + node.nodeSize, attribute.kind);
      const text = node.isLeaf
        ? NODE_CHARACTER
        : node.textBetween(0, node.content.size, "\n", NODE_CHARACTER);
      if (attribute.kind === "insert") proposal.added += text;
      else proposal.removed += text;
      if (node.isBlock) proposal.block = attribute.kind;
      noteCitations(node, attribute.request, attribute.kind, attribute.id);
      // A whole Block's words are its own: nothing inside it is another Proposal.
      return false;
    }
    if (!node.isText) return true;
    for (const mark of node.marks) {
      const name = mark.type.name;
      if (name !== PROPOSAL_INSERT_MARK && name !== PROPOSAL_DELETE_MARK) continue;
      const ref = readProposalRef(mark.attrs);
      if (!ref) continue;
      const proposal = entry(ref);
      extend(proposal, pos, pos + node.nodeSize);
      if (name === PROPOSAL_INSERT_MARK) proposal.added += node.text ?? "";
      else proposal.removed += node.text ?? "";
    }
    return false;
  });

  for (const proposal of found.values()) {
    const added = addedCitations.get(proposal.request);
    proposal.droppedCitations = (removedCitations.get(proposal.id) ?? []).filter(
      (key) => !added?.has(key.slice(key.indexOf("\u0000") + 1)),
    ).length;
  }
  return [...found.values()].sort((a, b) => a.from - b.from);
}

/**
 * The Proposals a selection is on: those a selection of text touches (a
 * section of the Mind, as in Word), or the one the cursor is in or beside.
 */
export function proposalsAt(
  proposals: readonly PendingProposal[],
  selection: Pick<Selection, "from" | "to" | "empty">,
): PendingProposal[] {
  return proposals.filter((proposal) =>
    proposal.ranges.some((range) =>
      selection.empty
        ? range.from <= selection.from && selection.from <= range.to
        : range.from < selection.to && selection.from < range.to,
    ),
  );
}

/** The Proposals waiting in the editor's Mind. */
export const pendingIn = (state: EditorState): readonly PendingProposal[] =>
  proposalReviewKey.getState(state)?.proposals ?? [];

/** The Proposals the editor's selection is on. */
export const pendingAtSelection = (state: EditorState): PendingProposal[] =>
  proposalsAt(pendingIn(state), state.selection);

/** The Proposal the bar is for: the one under the pointer, else the one the cursor is in. */
export function shownProposal(state: EditorState): PendingProposal | null {
  const review = proposalReviewKey.getState(state);
  if (!review) return null;
  const hovered = review.hovered && review.proposals.find((each) => each.id === review.hovered);
  return hovered || (pendingAtSelection(state)[0] ?? null);
}

/** The text waiting for Edit with AI, or being edited from the composer. */
export const editTarget = (state: EditorState) => proposalReviewKey.getState(state)?.target ?? null;

/** Marks text as waiting for Edit with AI, or clears it (null). */
export function setEditTarget(editor: Editor, target: { from: number; to: number } | null): void {
  if (editor.isDestroyed) return;
  editor.commands.command(({ tr }) => {
    tr.setMeta(proposalReviewKey, { target } satisfies Meta);
    return true;
  });
}

/** Whether content a transaction puts in holds a Proposal (all but typing's few characters are looked at). */
function bringsProposals(tr: Transaction): boolean {
  const holds = (node: ProseMirrorNode) =>
    readProposalAttribute(node.attrs[PROPOSAL_ATTRIBUTE]) !== null ||
    node.marks.some(
      (mark) => mark.type.name === PROPOSAL_INSERT_MARK || mark.type.name === PROPOSAL_DELETE_MARK,
    );
  return tr.steps.some((step) => {
    const slice = (step as { slice?: { content: ProseMirrorNode["content"] } }).slice;
    if (!slice) return true;
    let found = false;
    slice.content.descendants((node) => {
      found ||= holds(node);
      return !found;
    });
    return found;
  });
}

/** A change that came from the Mind's document (the core, another window, Undo), not the User typing here. */
const fromDocument = (tr: Transaction) => !!tr.getMeta(ySyncPluginKey)?.isChangeOrigin;

function decorationsFor(
  doc: ProseMirrorNode,
  proposals: readonly PendingProposal[],
  current: string | null,
  target: { from: number; to: number } | null,
): DecorationSet {
  const decorations: Decoration[] = [];
  for (const proposal of proposals) {
    for (const range of proposal.ranges) {
      // A node a Proposal adds or removes is drawn like its words (node views draw no marks).
      if (range.node) {
        decorations.push(
          Decoration.node(range.from, range.to, {
            class: `proposal-node proposal-node--${range.node}${proposal.id === current ? " proposal-current" : ""}`,
            "data-proposal-id": proposal.id,
          }),
        );
      } else if (proposal.id === current) {
        decorations.push(Decoration.inline(range.from, range.to, { class: "proposal-current" }));
      }
    }
  }
  if (target && target.from < target.to) {
    decorations.push(Decoration.inline(target.from, target.to, { class: "proposal-target" }));
  }
  return decorations.length > 0 ? DecorationSet.create(doc, decorations) : DecorationSet.empty;
}

/** The marks of words the User typed or pasted into a Proposal: their words are their own. */
function proposalMarks(state: EditorState): MarkType[] {
  return [
    state.schema.marks[PROPOSAL_INSERT_MARK],
    state.schema.marks[PROPOSAL_DELETE_MARK],
  ].filter((type): type is MarkType => type !== undefined);
}

/**
 * Proposals waiting in the Mind, reviewed in place: each found from its marks
 * and attributes (`proposalsIn`), the one the cursor is in washed, Blocks
 * and inline nodes a Proposal adds or removes drawn like its words, and the
 * text Edit with AI is working on marked until its Proposals come back.
 * Words the User types into a Proposal are theirs: they carry no Proposal.
 */
export const ProposalReview = Extension.create({
  name: "proposalReview",

  addProseMirrorPlugins() {
    return [
      new Plugin<ReviewState>({
        key: proposalReviewKey,
        state: {
          init: (_config, state) => {
            const proposals = proposalsIn(state.doc);
            return {
              proposals,
              hovered: null,
              target: null,
              decorations: decorationsFor(
                state.doc,
                proposals,
                proposalsAt(proposals, state.selection)[0]?.id ?? null,
                null,
              ),
            };
          },
          apply(tr, value, oldState, state): ReviewState {
            const meta = tr.getMeta(proposalReviewKey) as Meta | undefined;
            let { proposals, hovered, target } = value;
            // Typing brings no Proposal, only moves or removes one: a Mind with none stays without.
            if (
              tr.docChanged &&
              (proposals.length > 0 || fromDocument(tr) || bringsProposals(tr))
            ) {
              proposals = proposalsIn(state.doc);
            }
            if (target && tr.docChanged) {
              const from = tr.mapping.map(target.from, 1);
              const to = tr.mapping.map(target.to, -1);
              target = from < to ? { from, to } : null;
            }
            if (meta && "hovered" in meta) hovered = meta.hovered;
            if (meta && "target" in meta) target = meta.target;
            if (hovered && !proposals.some((each) => each.id === hovered)) hovered = null;
            const current = proposalsAt(proposals, state.selection)[0]?.id ?? null;
            const unchanged =
              proposals === value.proposals &&
              target === value.target &&
              current === (proposalsAt(value.proposals, oldState.selection)[0]?.id ?? null);
            return {
              proposals,
              hovered,
              target,
              decorations: unchanged
                ? value.decorations.map(tr.mapping, tr.doc)
                : decorationsFor(state.doc, proposals, current, target),
            };
          },
        },

        // What the User types or pastes into a Proposal's words is theirs, not the Proposal's.
        appendTransaction(transactions, before, after) {
          if (pendingIn(before).length === 0) return null;
          const typed = transactions.filter((tr) => tr.docChanged && !fromDocument(tr));
          if (typed.length === 0) return null;
          const marks = proposalMarks(after);
          const ranges: { from: number; to: number }[] = [];
          for (const [index, tr] of transactions.entries()) {
            if (!typed.includes(tr)) continue;
            for (const map of tr.mapping.maps) {
              map.forEach((_oldStart, _oldEnd, newStart, newEnd) => {
                if (newEnd > newStart) ranges.push({ from: newStart, to: newEnd });
              });
            }
            // Into the last document: through the transactions after this one.
            for (const later of transactions.slice(index + 1)) {
              for (const range of ranges) {
                range.from = later.mapping.map(range.from, 1);
                range.to = later.mapping.map(range.to, -1);
              }
            }
          }
          let tr: Transaction | null = null;
          for (const range of ranges) {
            if (range.from >= range.to) continue;
            for (const type of marks) {
              if (!after.doc.rangeHasMark(range.from, range.to, type)) continue;
              tr ??= after.tr;
              tr.removeMark(range.from, range.to, type);
            }
          }
          return tr;
        },

        props: {
          decorations: (state) => proposalReviewKey.getState(state)?.decorations ?? null,

          handleDOMEvents: {
            mouseover(view, event) {
              const target = event.target instanceof Element ? event.target : null;
              const id =
                target?.closest("[data-proposal-id]")?.getAttribute("data-proposal-id") || null;
              if (!id || id === proposalReviewKey.getState(view.state)?.hovered) return false;
              view.dispatch(
                view.state.tr.setMeta(proposalReviewKey, { hovered: id } satisfies Meta),
              );
              return false;
            },
          },
        },
      }),
    ];
  },
});

/** The pointer left the Proposal it was on, for the bar (which keeps it while it is pointed at). */
export function clearHovered(editor: Editor): void {
  if (editor.isDestroyed || !proposalReviewKey.getState(editor.state)?.hovered) return;
  editor.commands.command(({ tr }) => {
    tr.setMeta(proposalReviewKey, { hovered: null } satisfies Meta);
    return true;
  });
}

/**
 * Accepts or rejects Proposals in the editor's copy of the Mind, as one step
 * of its Undo (⌘Z takes it back), the same way the core does anywhere
 * (`resolveProposals`). The change reaches the core as the User's edits do.
 */
export function resolveInEditor(
  editor: Editor,
  ids: ReadonlySet<string>,
  outcome: "accept" | "reject",
): ResolvedProposal[] {
  const binding: { doc: Y.Doc; type: Y.XmlFragment } | undefined = ySyncPluginKey.getState(
    editor.state,
  )?.binding;
  if (!binding || ids.size === 0) return [];
  const undo: { stopCapturing(): void } | undefined = yUndoPluginKey.getState(
    editor.state,
  )?.undoManager;
  undo?.stopCapturing();
  let resolved: ResolvedProposal[] = [];
  // As the editor's own change, so its Undo takes it back.
  binding.doc.transact(() => {
    resolved = resolveProposals(binding.type, (proposal) => ids.has(proposal.id), outcome);
  }, ySyncPluginKey);
  undo?.stopCapturing();
  return resolved;
}

/** The ids of the Blocks a Proposal is in, for the record of AI use. */
export function blockIdsOf(doc: ProseMirrorNode, proposal: PendingProposal): string[] {
  const ids = new Set<string>();
  for (const range of proposal.ranges) {
    const $from = doc.resolve(range.from);
    for (let depth = $from.depth; depth > 0; depth--) {
      const id: unknown = $from.node(depth).attrs[BLOCK_ID_ATTRIBUTE];
      if (typeof id === "string") {
        ids.add(id);
        break;
      }
    }
  }
  return [...ids];
}
