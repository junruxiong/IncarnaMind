import * as Y from "yjs";
import { MIND_CONTENT_FIELD, MIND_PROPOSAL_REQUESTS_FIELD } from "../../../core/api";
import type { WordsInLine } from "../lineNames";
import { type Undoable, useUndo } from "../undo";

type Step = Y.UndoManager["undoStack"][number];

/**
 * The decisions made from the reviewing pane (#697), as a history of their
 * own beside the note's: in the pane, ⌘Z and ⇧⌘Z take back and do again
 * these and nothing else, whatever the User did in the Mind between them.
 * Yjs takes back a step's own changes only, so a later edit stays.
 */
export interface ReviewHistory {
  manager: Y.UndoManager;
  /** The origin of the decisions' Yjs transactions, which only this history follows. */
  origin: object;
}

const histories = new WeakMap<Y.Doc, ReviewHistory>();
// So that ⌘Z in the pane goes through the undo line when its entry is next, and the line stays true.
const offers = new WeakMap<Step, Undoable>();

export function reviewHistoryOf(doc: Y.Doc): ReviewHistory {
  let history = histories.get(doc);
  if (!history) {
    const origin = {};
    const manager = new Y.UndoManager(
      [doc.getXmlFragment(MIND_CONTENT_FIELD), doc.getMap(MIND_PROPOSAL_REQUESTS_FIELD)],
      { trackedOrigins: new Set([origin]), captureTimeout: 0 },
    );
    history = { manager, origin };
    histories.set(doc, history);
  }
  return history;
}

/**
 * Puts the last decision on the app's undo stack, with its line ("Accepted 1
 * change · Undo"): Undo and Redo there act on this decision alone, and do
 * nothing once it isn't the next to take back or do again here.
 */
export function offerReviewUndo(history: ReviewHistory, said: WordsInLine): void {
  const { manager } = history;
  let step = manager.undoStack.at(-1);
  if (!step) return;
  let undone = false;
  const change: Undoable = {
    said,
    undo: async () => {
      if (undone || !step || manager.undoStack.at(-1) !== step) return;
      manager.undo();
      undone = true;
      step = manager.redoStack.at(-1);
      if (step) offers.set(step, change);
    },
    redo: async () => {
      if (!undone || !step || manager.redoStack.at(-1) !== step) return;
      manager.redo();
      undone = false;
      step = manager.undoStack.at(-1);
      if (step) offers.set(step, change);
    },
  };
  offers.set(step, change);
  useUndo.getState().push(change);
}

/** ⌘Z in the pane: the last decision made from it, through its undo line when that is next. */
export function undoInPane(history: ReviewHistory): void {
  const step = history.manager.undoStack.at(-1);
  if (!step) return;
  const offer = offers.get(step);
  const app = useUndo.getState();
  if (offer && app.done.at(-1) === offer) {
    void app.undo();
    return;
  }
  if (offer) app.withdraw(offer);
  history.manager.undo();
}

/** ⇧⌘Z in the pane: the decision taken back last, done again. */
export function redoInPane(history: ReviewHistory): void {
  const step = history.manager.redoStack.at(-1);
  if (!step) return;
  const offer = offers.get(step);
  const app = useUndo.getState();
  if (offer && app.undone.at(-1) === offer) {
    void app.redo();
    return;
  }
  if (offer) app.withdraw(offer);
  history.manager.redo();
}
