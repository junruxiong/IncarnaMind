/**
 * Proposals in a Mind's Yjs document (see `PROPOSAL_INSERT_MARK` in
 * core/api.ts): reading them, and accepting or rejecting them. One
 * definition for the core, which resolves them in a Mind that isn't open,
 * and the editor, which resolves them in its own copy of the document, so
 * that its Undo takes them back.
 *
 * Accepting a Proposal keeps the words it adds (their mark goes) and deletes
 * the words it removes; a Block or inline node it adds loses its attribute,
 * and one it removes is deleted. Rejecting does the opposite. Nothing else in
 * the Mind changes.
 */
import * as Y from "yjs";
import {
  BLOCK_ID_ATTRIBUTE,
  FOOTNOTE_NODE,
  MIND_PROPOSAL_REQUESTS_FIELD,
  PROPOSAL_ATTRIBUTE,
  PROPOSAL_DELETE_MARK,
  PROPOSAL_INSERT_MARK,
  type ProposalAttribute,
  type ProposalRef,
  type ProposalRequest,
  type RecordAiUseInput,
} from "../core/api";

const isText = (value: unknown): value is string => typeof value === "string" && value !== "";

/** The Proposal a mark's attributes name, or null for anything malformed. */
export function readProposalRef(value: unknown): ProposalRef | null {
  if (typeof value !== "object" || value === null) return null;
  const { id, request } = value as Record<string, unknown>;
  return isText(id) && isText(request) ? { id, request } : null;
}

/** A node's `proposal` attribute, or null for none or anything malformed. */
export function readProposalAttribute(value: unknown): ProposalAttribute | null {
  const ref = readProposalRef(value);
  const kind = ref && (value as Record<string, unknown>).kind;
  return ref && (kind === "insert" || kind === "delete") ? { ...ref, kind } : null;
}

/** What resolving a Proposal did, for the record of AI use (#292). */
export interface ResolvedProposal extends ProposalRef {
  /** How much text it added that stays (an inline node counts as one): none once rejected. */
  characters: number;
  /** The Blocks (by their ids) holding what it added that stays: none once rejected. */
  blockIds: string[];
}

interface Run {
  insert: unknown;
  attributes?: Record<string, unknown>;
}

/** The length of a node's text, an inline node counting as one. */
function textLength(node: Y.XmlElement | Y.XmlText): number {
  if (node instanceof Y.XmlText) return node.length;
  const children = node.toArray();
  if (children.length === 0) return 1;
  let length = 0;
  for (const child of children) {
    if (child instanceof Y.XmlText || child instanceof Y.XmlElement) length += textLength(child);
  }
  return length;
}

const idOf = (element: Y.XmlElement): string | null => {
  const id = element.getAttribute(BLOCK_ID_ATTRIBUTE);
  return typeof id === "string" && id !== "" ? id : null;
};

/**
 * Accepts or rejects the Proposals `wanted` picks, everywhere in the Mind's
 * Blocks, in the caller's Yjs transaction. Returns what each one did, in the
 * order they come in the Mind.
 */
export function resolveProposals(
  blocks: Y.XmlFragment,
  wanted: (proposal: ProposalRef) => boolean,
  outcome: "accept" | "reject",
): ResolvedProposal[] {
  const resolved = new Map<string, ResolvedProposal>();
  const order: string[] = [];
  const note = (ref: ProposalRef, characters: number, blockId: string | null) => {
    let entry = resolved.get(ref.id);
    if (!entry) {
      entry = { id: ref.id, request: ref.request, characters: 0, blockIds: [] };
      resolved.set(ref.id, entry);
      order.push(ref.id);
    }
    entry.characters += characters;
    if (characters > 0 && blockId && !entry.blockIds.includes(blockId))
      entry.blockIds.push(blockId);
  };

  const resolveText = (text: Y.XmlText, owner: string | null) => {
    const edits: { at: number; length: number; mark: string; keep: boolean }[] = [];
    let at = 0;
    for (const run of text.toDelta() as Run[]) {
      const length = typeof run.insert === "string" ? run.insert.length : 1;
      for (const mark of [PROPOSAL_INSERT_MARK, PROPOSAL_DELETE_MARK]) {
        const ref = readProposalRef(run.attributes?.[mark]);
        if (!ref || !wanted(ref)) continue;
        const keep = (mark === PROPOSAL_INSERT_MARK) === (outcome === "accept");
        edits.push({ at, length, mark, keep });
        note(ref, keep && mark === PROPOSAL_INSERT_MARK ? length : 0, owner);
      }
      at += length;
    }
    // From the end, so each edit's place is still where it was read.
    for (const edit of edits.reverse()) {
      if (edit.keep) text.format(edit.at, edit.length, { [edit.mark]: null });
      else text.delete(edit.at, edit.length);
    }
  };

  const visit = (parent: Y.XmlFragment | Y.XmlElement, owner: string | null) => {
    const children = parent.toArray();
    for (let index = children.length - 1; index >= 0; index--) {
      const child = children[index];
      if (child instanceof Y.XmlText) {
        resolveText(child, owner);
        continue;
      }
      if (!(child instanceof Y.XmlElement)) continue;
      // A footnote has an id but isn't a Block, and one a Proposal moves holds the User's text.
      const footnote = child.nodeName === FOOTNOTE_NODE;
      const own = (footnote ? null : idOf(child)) ?? owner;
      const attribute = readProposalAttribute(child.getAttribute(PROPOSAL_ATTRIBUTE));
      if (attribute && wanted(attribute)) {
        const keep = (attribute.kind === "insert") === (outcome === "accept");
        const written = keep && attribute.kind === "insert" && !footnote;
        note(attribute, written ? textLength(child) : 0, own);
        if (!keep) {
          parent.delete(index, 1);
          continue;
        }
        child.removeAttribute(PROPOSAL_ATTRIBUTE);
      }
      visit(child, own);
    }
  };

  visit(blocks, null);
  // Walked from the end: the first one met is the last in the Mind.
  return order.reverse().map((id) => resolved.get(id) as ResolvedProposal);
}

/** The Proposals waiting in the Mind's Blocks, each once, in the order they start. */
export function pendingProposals(blocks: Y.XmlFragment): ProposalRef[] {
  const found = new Map<string, ProposalRef>();
  const add = (ref: ProposalRef | null) => {
    if (ref && !found.has(ref.id)) found.set(ref.id, ref);
  };
  const visit = (parent: Y.XmlFragment | Y.XmlElement) => {
    for (const child of parent.toArray()) {
      if (child instanceof Y.XmlText) {
        for (const run of child.toDelta() as Run[]) {
          add(readProposalRef(run.attributes?.[PROPOSAL_INSERT_MARK]));
          add(readProposalRef(run.attributes?.[PROPOSAL_DELETE_MARK]));
        }
      } else if (child instanceof Y.XmlElement) {
        add(readProposalAttribute(child.getAttribute(PROPOSAL_ATTRIBUTE)));
        visit(child);
      }
    }
  };
  visit(blocks);
  return [...found.values()];
}

const requestsOf = (doc: Y.Doc) => doc.getMap<unknown>(MIND_PROPOSAL_REQUESTS_FIELD);

/** Records what made a request's Proposals, in the Mind's document beside its Blocks. */
export function recordProposalRequest(doc: Y.Doc, request: string, made: ProposalRequest): void {
  requestsOf(doc).set(request, { ...made });
}

/** What made a request's Proposals, or null when the Mind doesn't say. */
export function proposalRequestOf(doc: Y.Doc, request: string): ProposalRequest | null {
  const value = requestsOf(doc).get(request);
  if (typeof value !== "object" || value === null) return null;
  const { source, providerId, modelId, createdAt } = value as Record<string, unknown>;
  return source === "edit-with-ai" && isText(providerId) && isText(modelId) && isText(createdAt)
    ? { source, providerId, modelId, createdAt }
    : null;
}

/**
 * The record of AI use's entries (#292) for Proposals just accepted: one per
 * Proposal that added text which stays, with the model that wrote it and the
 * Blocks holding it. A Proposal that only removed words wrote nothing.
 */
export function aiUseOfAccepted(
  doc: Y.Doc,
  mindId: string,
  accepted: readonly ResolvedProposal[],
  occurredAt: string,
): RecordAiUseInput[] {
  return accepted
    .filter((proposal) => proposal.characters > 0)
    .map((proposal) => {
      const made = proposalRequestOf(doc, proposal.request);
      return {
        mindId,
        kind: "edit-with-ai-accepted",
        providerId: made?.providerId ?? null,
        modelId: made?.modelId ?? null,
        characters: proposal.characters,
        blockIds: proposal.blockIds,
        occurredAt,
      };
    });
}
