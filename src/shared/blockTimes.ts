/**
 * When, in its Meeting, each line of a Mind was written (#139): the Meeting's
 * time in ms by Block id, in a map beside the Blocks in the Mind's Yjs document
 * (`MIND_BLOCK_TIMES_FIELD`). It is a map, not an attribute on the Blocks: an
 * older version's editor strips attributes it doesn't know, which would need
 * the content schema version to rise and make every Mind written since open
 * read-only there (ADR-0003, amended for `textOrigins`). A Block deleted since
 * leaves its entry unused; a copy of a line elsewhere has no time.
 */
import type * as Y from "yjs";
import { type Meeting, MIND_BLOCK_TIMES_FIELD } from "../core/api";

const timesOf = (doc: Y.Doc): Y.Map<unknown> => doc.getMap(MIND_BLOCK_TIMES_FIELD);

/** The Meeting's time now, in ms: what is recorded, and the stretch being recorded. */
export const meetingTimeAt = (
  meeting: Pick<Meeting, "recordedMs" | "recordingSince">,
  now: number,
): number =>
  meeting.recordedMs +
  (meeting.recordingSince === null ? 0 : Math.max(0, now - Date.parse(meeting.recordingSince)));

/**
 * The time a line written now gets: only while audio is being recorded. A
 * paused Meeting's time stands still, so a line written then would claim a
 * moment the Transcript has no words for.
 */
export const writingTime = (
  meeting: Pick<Meeting, "status" | "recordedMs" | "recordingSince"> | undefined,
  now: number,
): number | null =>
  meeting?.status === "recording" && meeting.recordingSince !== null
    ? meetingTimeAt(meeting, now)
    : null;

/** When the Block with this id was written, in the Meeting's ms, or null: no time was taken. */
export function blockTimeOf(doc: Y.Doc, blockId: string): number | null {
  const value = timesOf(doc).get(blockId);
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

/**
 * Records the time of the Blocks that have none yet, in one Yjs transaction:
 * a line keeps the time it was begun at, however long it is edited after.
 * Returns the ids that took it.
 */
export function setBlockTimes(doc: Y.Doc, blockIds: readonly string[], ms: number): string[] {
  const fresh = blockIds.filter((id) => blockTimeOf(doc, id) === null);
  if (fresh.length === 0) return [];
  doc.transact(() => {
    for (const id of fresh) timesOf(doc).set(id, Math.round(ms));
  });
  return fresh;
}

/** Every Block with a time, by id. */
export function blockTimes(doc: Y.Doc): ReadonlyMap<string, number> {
  const all = new Map<string, number>();
  for (const id of timesOf(doc).keys()) {
    const ms = blockTimeOf(doc, id);
    if (ms !== null) all.set(id, ms);
  }
  return all;
}
