import { describe, expect, test } from "vitest";
import { bestAssignment, diarizationError, readRttm } from "../../eval/speakers/der";

const turn = (startMs: number, endMs: number, speaker: string) => ({ startMs, endMs, speaker });

describe("The speaker evaluation's DER (eval/speakers/der.ts)", () => {
  // protects=DER as md-eval scores it: a hypothesis that is the reference under other names scores 0, and each kind of error counts its time over the reference's speech; fails_when=names must match, overlap counts once, or an error is counted twice; no_claim=md-eval's own rounding at frame edges.
  test("names don't matter, and missed, false and confused speech each count their time", () => {
    const reference = [turn(0, 4000, "A"), turn(4000, 8000, "B")];
    expect(diarizationError(reference, [turn(0, 4000, "x"), turn(4000, 8000, "y")]).der).toBe(0);
    expect(diarizationError(reference, [turn(0, 8000, "x")])).toMatchObject({
      der: 0.5,
      confusionMs: 4000,
      missedMs: 0,
      falseAlarmMs: 0,
    });
    expect(diarizationError(reference, [turn(0, 4000, "x")])).toMatchObject({
      der: 0.5,
      missedMs: 4000,
    });
    expect(
      diarizationError(reference, [
        turn(0, 4000, "x"),
        turn(4000, 8000, "y"),
        turn(8000, 10_000, "y"),
      ]),
    ).toMatchObject({ falseAlarmMs: 2000, der: 0.25 });
  });

  // protects=overlapping speech counts for each speaker in it, and a collar leaves out the time around reference edges; fails_when=overlap is counted once or the collar is ignored; no_claim=forgiving overlap, which md-eval can and this doesn't.
  test("overlap counts each speaker, and a collar leaves out boundaries", () => {
    const reference = [turn(0, 4000, "A"), turn(2000, 6000, "B")];
    expect(diarizationError(reference, [turn(0, 6000, "x")])).toMatchObject({
      speechMs: 8000,
      missedMs: 2000,
    });
    const late = [turn(0, 4000, "x"), turn(4200, 8000, "y")];
    const ref = [turn(0, 4000, "A"), turn(4000, 8000, "B")];
    expect(diarizationError(ref, late).der).toBeGreaterThan(0);
    expect(diarizationError(ref, late, { collarMs: 250 }).der).toBe(0);
  });

  test("the best assignment takes the most shared time, with more of one side than the other", () => {
    expect(
      bestAssignment([
        [1, 9],
        [8, 2],
      ]),
    ).toEqual([1, 0]);
    expect(bestAssignment([[5], [7]])).toEqual([-1, 0]);
    expect(bestAssignment([[1, 2, 9]])).toEqual([2]);
  });

  test("RTTM lines are read in seconds", () => {
    expect(readRttm("SPEAKER ES2004a 1 0.37 1.39 <NA> <NA> MEO015 <NA> <NA>\n")).toEqual([
      { startMs: 370, endMs: 1760, speaker: "MEO015" },
    ]);
  });
});
