import type { Meeting } from "../../core/api";
import { core } from "./core";
import type { Translate } from "./objects";
import { UNDO_SHOWN_MS, type Undoable, useUndo } from "./undo";

interface Offer {
  change: Undoable;
  seen: boolean;
  timer?: ReturnType<typeof setTimeout>;
}

const offers = new Map<string, Offer>();

const capturedAnything = (meeting: Meeting) =>
  meeting.audio.kept ||
  meeting.recordedMs > 0 ||
  meeting.status === "paused" ||
  meeting.status === "stopped";

function drop(meetingId: string): void {
  const offer = offers.get(meetingId);
  if (!offer) return;
  clearTimeout(offer.timer);
  offers.delete(meetingId);
  useUndo.getState().withdraw(offer.change);
}

/**
 * Undo for the Mind a Record just made, while nothing is recorded in it. It can't be done again:
 * the Meeting goes with the Mind, so Redo would bring back a Mind with no recording.
 */
export function offerTakeBack(
  meeting: Meeting,
  recording: boolean,
  current: () => Meeting | undefined,
): void {
  const change: Undoable = {
    said: (t: Translate) => t(recording ? "meetings.record.undo" : "undo.mindMade"),
    name: (t: Translate) => t(recording ? "meetings.record" : "menu.newMind"),
    undo: async () => {
      const now = current();
      if (now && capturedAnything(now)) return;
      await core.deleteMind(meeting.mindId);
    },
  };
  offers.set(meeting.id, { change, seen: false });
  useUndo.getState().push(change);
}

/** Ends the offer once audio is saved or the line has gone: Undo never deletes a recorded Meeting. */
export function settleTakeBacks(meetings: readonly Meeting[]): void {
  for (const [meetingId, offer] of offers) {
    const meeting = meetings.find((each) => each.id === meetingId);
    if (!meeting) {
      if (offer.seen) drop(meetingId);
      continue;
    }
    offer.seen = true;
    if (capturedAnything(meeting)) drop(meetingId);
    else if (meeting.recordingSince !== null && !offer.timer) {
      offer.timer = setTimeout(() => drop(meetingId), UNDO_SHOWN_MS);
    }
  }
}
