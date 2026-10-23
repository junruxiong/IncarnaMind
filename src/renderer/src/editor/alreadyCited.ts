import type { EditorState } from "@tiptap/pm/state";
import { overlapsCitation } from "../../../shared/citationOverlap";
import { type NumberedCitation, numberedCitations } from "./citationNumbers";
import type { ShownClaim } from "./claimChecks";

/** The Citations a check's sentence holds, with their numbers in the Mind. */
export const citationsOf = (state: EditorState, claim: ShownClaim): NumberedCitation[] =>
  numberedCitations(state).filter(
    (each) => each.pos >= claim.sentence.from && each.pos < claim.sentence.to,
  );

/**
 * The sentence's Citation that already cites the Passage its check found, or
 * null. A figure worked out has inputs, not one Passage to cite.
 */
export function alreadyCitedBy(state: EditorState, claim: ShownClaim): NumberedCitation | null {
  const { evidence, calculation } = claim.check;
  if (!evidence || calculation) return null;
  return (
    citationsOf(state, claim).find(({ attributes }) => overlapsCitation(attributes, evidence)) ??
    null
  );
}
