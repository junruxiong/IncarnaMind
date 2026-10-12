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
 * the Mind changes. A Proposal out of date (`outOfDateProposals`) is never
 * accepted: the text it would change isn't the text it was made for.
 */
import * as Y from "yjs";
import {
  BLOCK_ID_ATTRIBUTE,
  CITATION_NODE,
  editActions,
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
  const { id, request, target } = value as Record<string, unknown>;
  if (!isText(id) || !isText(request)) return null;
  return isText(target) ? { id, request, target } : { id, request };
}

/**
 * The hash a Proposal records of the text it changes (`ProposalRef.target`):
 * cyrb53, 53 bits in base 36. Not for secrecy, only to tell whether the text
 * is still the same; the same in the core and the editor.
 */
export function targetHash(text: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let index = 0; index < text.length; index++) {
    const code = text.charCodeAt(index);
    h1 = Math.imul(h1 ^ code, 2654435761);
    h2 = Math.imul(h2 ^ code, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}

/** How a Proposal's text reads an inline node, whatever it holds (a footnote, its text). */
export const NODE_CHARACTER = "\uFFFC";

/** The inline nodes that read as `NODE_CHARACTER` in a Proposal's text. */
export const INLINE_NODES: ReadonlySet<string> = new Set([
  CITATION_NODE,
  FOOTNOTE_NODE,
  "inlineMath",
  "hardBreak",
  "image",
]);

/**
 * A node's text as a Proposal's target reads it: its characters, an inline
 * node as `NODE_CHARACTER`, a Block as the text of everything in it, run
 * together. The editor reads its own nodes the same way (`proposalsIn`).
 */
export function targetText(node: Y.XmlElement | Y.XmlText): string {
  if (node instanceof Y.XmlText) {
    return (node.toDelta() as Run[])
      .map((run) => (typeof run.insert === "string" ? run.insert : NODE_CHARACTER))
      .join("");
  }
  if (INLINE_NODES.has(node.nodeName)) return NODE_CHARACTER;
  return node
    .toArray()
    .map((child) =>
      child instanceof Y.XmlText || child instanceof Y.XmlElement ? targetText(child) : "",
    )
    .join("");
}

/**
 * The Proposals out of date, by id: each records the text it changes
 * (`ProposalRef.target`), and that text isn't there as it was any more. Its
 * removed words (or the Block it removes) must still be what they were, in
 * one piece: words typed between them, or a paragraph split among them, put
 * it out of date as much as an edit to them does. Words typed into what it
 * adds don't: the User may finish a change before accepting it. A Proposal
 * that records no text (made before version 5) is never out of date.
 */
export function outOfDateProposals(blocks: Y.XmlFragment): Set<string> {
  /** Per Proposal: the text it records, the text it removes now, and where each removed part is. */
  const seen = new Map<string, { target: string | undefined; text: string; at: number[] }>();
  /** Each run of text, Block boundary and node counts one: parts in one piece are numbered in a row. */
  let atom = 0;
  const note = (ref: ProposalRef, removes: string | null) => {
    let entry = seen.get(ref.id);
    if (!entry) {
      entry = { target: ref.target, text: "", at: [] };
      seen.set(ref.id, entry);
    }
    entry.target ??= ref.target;
    if (removes !== null) {
      entry.text += removes;
      entry.at.push(atom);
    }
  };
  const visit = (parent: Y.XmlFragment | Y.XmlElement) => {
    for (const child of parent.toArray()) {
      if (child instanceof Y.XmlText) {
        for (const run of child.toDelta() as Run[]) {
          const removed = readProposalRef(run.attributes?.[PROPOSAL_DELETE_MARK]);
          const added = readProposalRef(run.attributes?.[PROPOSAL_INSERT_MARK]);
          if (removed) note(removed, typeof run.insert === "string" ? run.insert : NODE_CHARACTER);
          if (added) note(added, null);
          atom++;
        }
      } else if (child instanceof Y.XmlElement) {
        const attribute = readProposalAttribute(child.getAttribute(PROPOSAL_ATTRIBUTE));
        if (attribute) {
          note(attribute, attribute.kind === "delete" ? targetText(child) : null);
          atom++;
          continue;
        }
        atom++;
        visit(child);
        atom++;
      }
    }
  };
  visit(blocks);
  const stale = new Set<string>();
  for (const [id, entry] of seen) {
    if (entry.target === undefined) continue;
    const apart = entry.at.some((at, index) => index > 0 && at !== (entry.at[index - 1] ?? 0) + 1);
    if (apart || targetHash(entry.text) !== entry.target) stale.add(id);
  }
  return stale;
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
 * order they come in the Mind. One out of date is left as it is when accepting.
 */
export function resolveProposals(
  blocks: Y.XmlFragment,
  picked: (proposal: ProposalRef) => boolean,
  outcome: "accept" | "reject",
): ResolvedProposal[] {
  const stale = outcome === "accept" ? outOfDateProposals(blocks) : new Set<string>();
  const wanted = (proposal: ProposalRef) => picked(proposal) && !stale.has(proposal.id);
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
  const record = value as Record<string, unknown>;
  const { source, providerId, modelId, createdAt } = record;
  if (!isText(providerId) || !isText(modelId) || !isText(createdAt)) return null;
  const made = { providerId, modelId, createdAt };
  if (source === "answer") {
    return isText(record.answerId) ? { ...made, source, answerId: record.answerId } : null;
  }
  if (source === "answer-added") return { ...made, source };
  if (source !== "edit-with-ai") return null;
  const action = editActions.find((each) => each === record.action);
  return {
    ...made,
    source,
    ...(action && { action }),
    ...(typeof record.instruction === "string" && { instruction: record.instruction }),
  };
}

/**
 * The record of AI use's entries (#292) for Proposals just accepted: one per
 * Proposal that added text which stays, with the model that wrote it and the
 * Blocks holding it. A Proposal that only removed words wrote nothing, and
 * one from an Answer's Regenerate stays in its Answer, which isn't recorded
 * (an Answer in a Mind is the AI's as it is). An Answer from the assistant
 * made into a new Mind (#201) is recorded as an Answer added.
 */
export function aiUseOfAccepted(
  doc: Y.Doc,
  mindId: string,
  accepted: readonly ResolvedProposal[],
  occurredAt: string,
): RecordAiUseInput[] {
  return accepted
    .filter((proposal) => proposal.characters > 0)
    .flatMap((proposal): RecordAiUseInput[] => {
      const made = proposalRequestOf(doc, proposal.request);
      if (made?.source === "answer") return [];
      return [
        {
          mindId,
          kind: made?.source === "answer-added" ? "answer-added" : "edit-with-ai-accepted",
          providerId: made?.providerId ?? null,
          modelId: made?.modelId ?? null,
          characters: proposal.characters,
          blockIds: proposal.blockIds,
          occurredAt,
        },
      ];
    });
}
