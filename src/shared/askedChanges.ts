/**
 * Whether the changes the User asked the assistant for wait for them instead
 * of applying (#880): when they would take out most of the User's own text in
 * the Mind, unless they translate it, which rewrites all of it by asking. The
 * core tells the model so in a call's result, and the window holds them back,
 * by this one measure.
 */
import * as Y from "yjs";
import {
  GENERATED_MARK,
  MIND_CONTENT_FIELD,
  PROPOSAL_ATTRIBUTE,
  PROPOSAL_DELETE_MARK,
  PROPOSAL_INSERT_MARK,
} from "../core/api";
import { proposalRequestOf, readProposalAttribute, readProposalRef } from "./proposals";

const MOST_OF_YOUR_TEXT = 0.5;

const TRANSLATION =
  /\btranslat|\bin(?:to)? (?:chinese|english)\b|翻译|译成|(?:改成|用)(?:中文|英文)/i;

interface Run {
  insert: unknown;
  attributes?: Record<string, unknown>;
}

const marked = (run: Run, mark: string) => {
  const value = run.attributes?.[mark];
  return value !== undefined && value !== null;
};

/** How much of the User's own text in the Mind, not AI text, the requests' changes take out: a share from 0 to 1. */
export function ownTextRemoved(blocks: Y.XmlFragment, requests: ReadonlySet<string>): number {
  let own = 0;
  let removed = 0;
  const visit = (parent: Y.XmlFragment | Y.XmlElement, goes: boolean) => {
    for (const child of parent.toArray()) {
      if (child instanceof Y.XmlText) {
        for (const run of child.toDelta() as Run[]) {
          if (typeof run.insert !== "string") continue;
          if (marked(run, GENERATED_MARK) || marked(run, PROPOSAL_INSERT_MARK)) continue;
          own += run.insert.length;
          const deleted = readProposalRef(run.attributes?.[PROPOSAL_DELETE_MARK]);
          if (goes || (deleted && requests.has(deleted.request))) removed += run.insert.length;
        }
      } else if (child instanceof Y.XmlElement) {
        const proposal = readProposalAttribute(child.getAttribute(PROPOSAL_ATTRIBUTE));
        if (proposal?.kind === "insert") continue;
        visit(child, goes || (proposal?.kind === "delete" && requests.has(proposal.request)));
      }
    }
  };
  visit(blocks, false);
  return own === 0 ? 0 : removed / own;
}

/** Whether the User asked for a request's changes, in the words of their Question. */
export function askedFor(doc: Y.Doc, request: string): boolean {
  const made = proposalRequestOf(doc, request);
  return made !== null && "requested" in made && made.requested !== undefined;
}

function translates(doc: Y.Doc, requests: ReadonlySet<string>): boolean {
  return [...requests].every((request) => {
    const made = proposalRequestOf(doc, request);
    return (
      made?.source === "edit-with-ai" &&
      (made.action === "translate" || TRANSLATION.test(made.instruction ?? ""))
    );
  });
}

/** Whether the requests' changes wait for the User to apply or discard them. */
export function waitsForUser(doc: Y.Doc, requests: ReadonlySet<string>): boolean {
  return (
    requests.size > 0 &&
    !translates(doc, requests) &&
    ownTextRemoved(doc.getXmlFragment(MIND_CONTENT_FIELD), requests) > MOST_OF_YOUR_TEXT
  );
}
