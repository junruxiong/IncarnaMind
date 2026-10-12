/**
 * Writing Proposals into a Mind the User has edited since the model read it.
 * They are written into a copy of the Mind as the model read it, and what
 * that copy changed is merged into the Mind as Yjs merges two devices' edits:
 * the User's edits stay theirs, and a Proposal whose text they changed
 * arrives out of date (`outOfDateProposals` in src/shared/proposals.ts).
 * Merged, the changes may add Proposals and nothing else: the Mind's text
 * without them must read as it does now, or nothing is merged.
 */
import * as Y from "yjs";
import { INLINE_NODES, NODE_CHARACTER, readProposalAttribute } from "../../shared/proposals";
import { MIND_CONTENT_FIELD, PROPOSAL_ATTRIBUTE, PROPOSAL_INSERT_MARK } from "../api";

interface Run {
  insert: unknown;
  attributes?: Record<string, unknown>;
}

/** A copy of a Mind's document, from its whole state (`MindContent.state`). */
export function docOf(state: Uint8Array): Y.Doc {
  const doc = new Y.Doc();
  Y.applyUpdate(doc, state);
  return doc;
}

/**
 * The Blocks' text without what Proposals add, with what they remove, Block
 * by Block and an inline node as one character: the text as the User has it.
 */
export function textWithoutProposals(parent: Y.XmlFragment | Y.XmlElement): string {
  let text = "";
  for (const child of parent.toArray()) {
    if (child instanceof Y.XmlText) {
      for (const run of child.toDelta() as Run[]) {
        if (run.attributes && PROPOSAL_INSERT_MARK in run.attributes) continue;
        text += typeof run.insert === "string" ? run.insert : NODE_CHARACTER;
      }
    } else if (child instanceof Y.XmlElement) {
      if (readProposalAttribute(child.getAttribute(PROPOSAL_ATTRIBUTE))?.kind === "insert")
        continue;
      text += INLINE_NODES.has(child.nodeName)
        ? NODE_CHARACTER
        : `<${child.nodeName}>${textWithoutProposals(child)}</${child.nodeName}>`;
    }
  }
  return text;
}

/**
 * What `fork`, a copy of the Mind made from `base`, has changed since, as an
 * update to apply to the Mind as it is now (`now`, its whole state); or null
 * when, merged there, it would change the User's text and not only add
 * Proposals (as when text was split around words the User typed meanwhile).
 */
export function forkChanges(base: Uint8Array, fork: Y.Doc, now: Uint8Array): Uint8Array | null {
  const update = Y.encodeStateAsUpdate(fork, Y.encodeStateVectorFromUpdate(base));
  const merged = docOf(now);
  try {
    const blocks = merged.getXmlFragment(MIND_CONTENT_FIELD);
    const before = textWithoutProposals(blocks);
    Y.applyUpdate(merged, update);
    return textWithoutProposals(blocks) === before ? update : null;
  } finally {
    merged.destroy();
  }
}
