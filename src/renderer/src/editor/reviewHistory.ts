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
interface Entry {
  change: Undoable;
  step: Step;
  undone: boolean;
}

// So that ⌘Z in the pane goes through the undo line when its entry is next, and the line stays true.
const entries = new WeakMap<Step, Entry>();

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

// Each Undo and Redo makes the step anew on the other stack: the entry follows it.
function step(manager: Y.UndoManager, entry: Entry, undo: boolean): void {
  if (undo) manager.undo();
  else manager.redo();
  const now = (undo ? manager.redoStack : manager.undoStack).at(-1);
  if (!now) return;
  entry.step = now;
  entry.undone = undo;
  entries.set(now, entry);
}

/**
 * Puts the last decision on the app's undo stack, with its line ("Accepted 1
 * change · Undo"): Undo and Redo there act on this decision alone, and do
 * nothing once it isn't the next to take back or do again here.
 */
export function offerReviewUndo(history: ReviewHistory, said: WordsInLine): void {
  const { manager } = history;
  const last = manager.undoStack.at(-1);
  if (!last) return;
  const entry: Entry = {
    change: {
      said,
      undo: async () => {
        if (!entry.undone && manager.undoStack.at(-1) === entry.step) step(manager, entry, true);
      },
      redo: async () => {
        if (entry.undone && manager.redoStack.at(-1) === entry.step) step(manager, entry, false);
      },
    },
    step: last,
    undone: false,
  };
  entries.set(last, entry);
  useUndo.getState().push(entry.change);
}

/** ⌘Z in the pane: the last decision made from it, through its undo line when that is next. */
export function undoInPane(history: ReviewHistory): void {
  const { manager } = history;
  const top = manager.undoStack.at(-1);
  if (!top) return;
  const entry = entries.get(top);
  const app = useUndo.getState();
  if (!entry) {
    manager.undo();
    return;
  }
  if (app.done.at(-1) === entry.change) {
    void app.undo();
    return;
  }
  // Taken back here, out of turn with the line: off the line until it is done again.
  app.withdraw(entry.change);
  step(manager, entry, true);
}

/** ⇧⌘Z in the pane: the decision taken back last, done again, and back on the undo line. */
export function redoInPane(history: ReviewHistory): void {
  const { manager } = history;
  const top = manager.redoStack.at(-1);
  if (!top) return;
  const entry = entries.get(top);
  const app = useUndo.getState();
  if (!entry) {
    manager.redo();
    return;
  }
  if (app.undone.at(-1) === entry.change) {
    void app.redo();
    return;
  }
  step(manager, entry, false);
  app.renew(entry.change);
}
