import type { EditorState } from "@tiptap/pm/state";
import { type CitationAttributes, PROPOSAL_ATTRIBUTE } from "../../../core/api";
import { readProposalAttribute } from "../../../shared/proposals";
import { citationClusters } from "./citationClusters";

/** What a source needs of a Citation: its Document, the version quoted and where. */
export type CountedCitation = Pick<
  CitationAttributes,
  "documentId" | "documentName" | "contentHash" | "pageFrom" | "pageTo" | "location"
>;

/**
 * The Citations the Mind's Bibliography counts, top to bottom: those in the Mind, not the ones
 * a pending change would add. One a pending change removes is still the Mind's, until it is accepted.
 */
export function countedCitations(state: EditorState): CountedCitation[] {
  return citationClusters(state)
    .flatMap((cluster) => cluster.members)
    .filter((member) => {
      const attributes: Record<string, unknown> = { ...member.attributes };
      return readProposalAttribute(attributes[PROPOSAL_ATTRIBUTE])?.kind !== "insert";
    })
    .map(({ attributes: a }) => ({
      documentId: a.documentId,
      documentName: a.documentName,
      contentHash: a.contentHash,
      pageFrom: a.pageFrom,
      pageTo: a.pageTo,
      location: a.location,
    }));
}
