import { Extension, Mark } from "@tiptap/core";
import { PROPOSAL_ATTRIBUTE, PROPOSAL_DELETE_MARK, PROPOSAL_INSERT_MARK } from "../../../core/api";
import { readProposalAttribute, readProposalRef } from "../../../shared/proposals";

/**
 * The node types that can carry a `proposal` attribute: Blocks a Proposal
 * adds or removes whole, and the inline nodes it adds or removes in a text
 * (see `PROPOSAL_ATTRIBUTE`).
 */
export const PROPOSAL_NODE_TYPES = [
  "paragraph",
  "heading",
  "blockquote",
  "bulletList",
  "orderedList",
  "listItem",
  "codeBlock",
  "blockMath",
  "horizontalRule",
  "table",
  "citation",
  "footnote",
  "inlineMath",
  "hardBreak",
] as const;

/**
 * The Proposal a mark belongs to, and the text it changes (`ProposalRef`):
 * kept in the Mind, never read from pasted HTML.
 */
const refAttributes = () => ({
  id: { default: null, parseHTML: () => null, renderHTML: () => ({}) },
  request: { default: null, parseHTML: () => null, renderHTML: () => ({}) },
  target: { default: null, parseHTML: () => null, renderHTML: () => ({}) },
});

/**
 * Words a Proposal adds, drawn as an insertion: underlined, in the Answer's
 * ink. Not inclusive, so what the User types beside it is their own.
 */
export const ProposalInsert = Mark.create({
  name: PROPOSAL_INSERT_MARK,
  inclusive: false,
  excludes: `${PROPOSAL_INSERT_MARK} ${PROPOSAL_DELETE_MARK}`,
  addAttributes: refAttributes,
  // A copied Proposal pastes as plain text: a Proposal waits only where the AI made it.
  parseHTML: () => [],
  renderHTML({ mark }) {
    const ref = readProposalRef(mark.attrs);
    return ["ins", { class: "proposal-insert", "data-proposal-id": ref?.id ?? "" }, 0];
  },
});

/** Words a Proposal removes, kept until it is resolved, drawn struck. */
export const ProposalDelete = Mark.create({
  name: PROPOSAL_DELETE_MARK,
  inclusive: false,
  excludes: `${PROPOSAL_INSERT_MARK} ${PROPOSAL_DELETE_MARK}`,
  addAttributes: refAttributes,
  parseHTML: () => [],
  renderHTML({ mark }) {
    const ref = readProposalRef(mark.attrs);
    return ["del", { class: "proposal-delete", "data-proposal-id": ref?.id ?? "" }, 0];
  },
});

/** The `proposal` attribute of whole Blocks and inline nodes a Proposal adds or removes. */
export const ProposalAttributes = Extension.create({
  name: "proposalAttributes",

  addGlobalAttributes() {
    return [
      {
        types: [...PROPOSAL_NODE_TYPES],
        attributes: {
          [PROPOSAL_ATTRIBUTE]: {
            default: null,
            parseHTML: () => null,
            renderHTML: (attributes: Record<string, unknown>) => {
              const value = readProposalAttribute(attributes[PROPOSAL_ATTRIBUTE]);
              return value ? { "data-proposal": value.kind, "data-proposal-id": value.id } : {};
            },
          },
        },
      },
    ];
  },
});
