/**
 * The fixes for a cited sentence found unsupported (#1111), from its card and
 * ⌘K: Cut the unsupported part, an Edit with AI that applies at once with
 * Undo; Find support, its check again in every Document; and Keep as my view,
 * which makes the sentence the User's own, as Mark as mine does, and keeps its
 * check from showing or counting, one step of Undo.
 */
import type { Editor } from "@tiptap/core";
import type * as Y from "yjs";
import { CLAIM_CHECK_MARK } from "../../core/api";
import { checkAgain, useChecks } from "./checks";
import { makeMine } from "./editor/aiText";
import { type NumberedCitation, numberedCitations } from "./editor/citationNumbers";
import { type ShownClaim, shownClaims } from "./editor/claimChecks";
import { asOneStep, offerEditorUndo } from "./editor/editorUndo";
import { requestEdit, useEdits } from "./editWithAi";
import { translateNow } from "./i18n";

/** The Citations a check's sentence holds, with their numbers in the Mind. */
export const citedIn = (editor: Editor, claim: ShownClaim): NumberedCitation[] =>
  numberedCitations(editor.state).filter(
    (each) => each.pos >= claim.sentence.from && each.pos < claim.sentence.to,
  );

/**
 * Whether a check is of a cited sentence nothing backs: not one where the
 * Documents say something different, one only the User's own workings say,
 * or one citing a court decision no Document holds, which have cards of their own.
 */
export function offersFixes(editor: Editor, claim: ShownClaim): boolean {
  const { check } = claim;
  return (
    claim.shown.state === "unsupported" &&
    check.evidence === null &&
    !check.citedCaseUnchecked &&
    !check.excluded.some((left) => left.reason === "own-workings") &&
    citedIn(editor, claim).length > 0
  );
}

/** The part a check named that the sentence's Citations don't support; null when it named none. */
export const uncoveredOf = (claim: ShownClaim): string | null =>
  claim.check.unsupported?.covered ? claim.check.unsupported.uncovered : null;

/** The cited sentence nothing backs that the selection is in, for ⌘K; null for none. */
export function unsupportedAt(editor: Editor): ShownClaim | null {
  const { from, to } = editor.state.selection;
  return (
    shownClaims(editor.state).find(
      (claim) =>
        claim.sentence.from <= from && to <= claim.sentence.to && offersFixes(editor, claim),
    ) ?? null
  );
}

/** Whether a fix can start now: no edit or check runs in the Mind. */
export const fixesFree = (mindId: string): boolean =>
  !useEdits.getState().pending[mindId] && !useChecks.getState().running[mindId];

/** Takes the part the Citations don't support out of the sentence, keeping them on the rest. */
export function cutUnsupported(
  editor: Editor,
  mindId: string,
  doc: Y.Doc,
  claim: ShownClaim,
): Promise<boolean> {
  const uncovered = uncoveredOf(claim);
  if (!uncovered) return Promise.resolve(false);
  return requestEdit(editor, mindId, doc, "cut-unsupported", uncovered, {
    from: claim.sentence.from,
    to: claim.sentence.to,
  });
}

/** Checks the sentence again in every Document: one that backs it is offered as a Citation. */
export const findSupport = (editor: Editor, mindId: string, doc: Y.Doc, claim: ShownClaim) =>
  checkAgain(editor, mindId, doc, claim, true);

/** Makes the sentence the User's own view: their words, and a check that no longer counts. */
export function keepAsView(editor: Editor, claim: ShownClaim): void {
  const { state } = editor;
  const type = state.schema.marks[CLAIM_CHECK_MARK];
  if (!type) return;
  const kept = type.create({ ...claim.check, kept: true });
  const tr = state.tr;
  // Only its words: the binding keeps no marks on a Citation.
  state.doc.nodesBetween(claim.from, claim.to, (node, pos) => {
    const marked = node.marks.some(
      (mark) => mark.type === type && mark.attrs.id === claim.check.id,
    );
    if (!node.isText || !marked) return;
    tr.addMark(Math.max(pos, claim.from), Math.min(pos + node.nodeSize, claim.to), kept);
  });
  makeMine(state, tr, claim.sentence.from, claim.sentence.to);
  asOneStep(editor, () => editor.view.dispatch(tr));
  offerEditorUndo(editor, translateNow("claim.keptAsView"));
}
