import { describe, expect, test } from "vitest";
import { toMeetingTime } from "../../src/core/transcription/diarization";

const SECOND_SAMPLES = 16_000;

describe("The speaker engine's times, in the Meeting's (#138)", () => {
  // protects=the engine reads a Meeting's parts joined end to end, so a turn is put back where its audio was recorded, cut where the recording paused, with voices numbered as first heard; fails_when=a turn after a pause is early, spans the pause, or keeps the engine's numbers; no_claim=the engine's turns themselves.
  test("turns go back to where their audio was recorded, cut at a pause, voices numbered as first heard", () => {
    const spans = [
      { at: 0, length: 10 * SECOND_SAMPLES, offsetMs: 0 },
      { at: 10 * SECOND_SAMPLES, length: 10 * SECOND_SAMPLES, offsetMs: 60_000 },
    ];
    expect(
      toMeetingTime(
        [
          { start: 2, end: 4, speaker: 7 },
          { start: 8, end: 12, speaker: 3 },
          { start: 15, end: 16, speaker: 7 },
        ],
        spans,
      ),
    ).toEqual([
      { startMs: 2000, endMs: 4000, voice: 0 },
      { startMs: 8000, endMs: 10_000, voice: 1 },
      { startMs: 60_000, endMs: 62_000, voice: 1 },
      { startMs: 65_000, endMs: 66_000, voice: 0 },
    ]);
  });

  test("a turn across two parts recorded one after the other stays one", () => {
    const spans = [
      { at: 0, length: 10 * SECOND_SAMPLES, offsetMs: 0 },
      { at: 10 * SECOND_SAMPLES, length: 10 * SECOND_SAMPLES, offsetMs: 10_000 },
    ];
    expect(toMeetingTime([{ start: 8, end: 12, speaker: 0 }], spans)).toEqual([
      { startMs: 8000, endMs: 12_000, voice: 0 },
    ]);
  });
});
