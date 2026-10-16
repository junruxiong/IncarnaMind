import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { LiveStep } from "../../src/renderer/src/answerSteps";
import { holdSteps, MIN_STEP_MS } from "../../src/renderer/src/heldStep";

/*
 * The live line's steps (#391): each stays up at least a second while another
 * replaces it. Against a clock the test moves, as the line's own timer runs on.
 */

const MIND = { mindId: "m1", name: "Notes", deleted: false };
const reading: LiveStep = { kind: "reading", mind: MIND };
const searching: LiveStep = { kind: "searching" };
const thinking: LiveStep = { kind: "thinking" };
const writing: LiveStep = { kind: "writing" };
const checking = (count: number): LiveStep => ({ kind: "checking", count });
const waiting: LiveStep = { kind: "waiting" };

/** A hold that records each step it shows, with when. */
function held(initial: LiveStep) {
  const shown: { step: LiveStep["kind"]; at: number }[] = [{ step: initial.kind, at: Date.now() }];
  const hold = holdSteps(initial, (step) => shown.push({ step: step.kind, at: Date.now() }));
  return { hold, shown, kinds: () => shown.map((each) => each.step) };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(0);
});
afterEach(() => {
  vi.useRealTimers();
});

describe("The steps of the live line", () => {
  // protects=a new step waits until the one before has been up a second, and shows then; fails_when=it shows before the second is up, or never; no_claim=none beyond protects.
  test("a new step shows once the one before has been up a second", () => {
    const { hold, kinds } = held(reading);
    vi.advanceTimersByTime(300);
    hold.offer(searching);
    expect(kinds()).toEqual(["reading"]);
    vi.advanceTimersByTime(MIN_STEP_MS - 300 - 1);
    expect(kinds()).toEqual(["reading"]);
    vi.advanceTimersByTime(1);
    expect(kinds()).toEqual(["reading", "searching"]);
  });

  // protects=a step that came when the one before had been up a second or more shows at once; fails_when=it is held again; no_claim=none beyond protects.
  test("a new step shows at once when the one before has been up long enough", () => {
    const { hold, kinds } = held(reading);
    vi.advanceTimersByTime(MIN_STEP_MS);
    hold.offer(searching);
    expect(kinds()).toEqual(["reading", "searching"]);
  });

  // protects=of the steps that come while one is held, only the latest shows; fails_when=a step that was over before its turn shows; no_claim=none beyond protects.
  test("of the steps that come meanwhile, only the latest shows", () => {
    const { hold, kinds } = held(reading);
    vi.advanceTimersByTime(100);
    hold.offer(searching);
    vi.advanceTimersByTime(100);
    hold.offer(thinking);
    vi.advanceTimersByTime(100);
    hold.offer(writing);
    vi.advanceTimersByTime(10 * MIN_STEP_MS);
    expect(kinds()).toEqual(["reading", "writing"]);
  });

  // protects=each step stays up at least a second whatever the pace the steps come at; fails_when=two steps show less than a second apart; no_claim=the steps are scripted, not an Answer's.
  test("each step is up at least a second, however fast the steps come", () => {
    const { hold, shown } = held(reading);
    const script = [searching, thinking, searching, thinking, writing, checking(1), checking(2)];
    for (const step of script) {
      vi.advanceTimersByTime(400);
      hold.offer(step);
    }
    vi.advanceTimersByTime(10 * MIN_STEP_MS);
    expect(shown.length).toBeGreaterThan(3);
    for (let index = 0; index + 1 < shown.length; index++) {
      expect((shown[index + 1]?.at ?? 0) - (shown[index]?.at ?? 0)).toBeGreaterThanOrEqual(
        MIN_STEP_MS,
      );
    }
  });

  // protects=a step that waits for the User shows at once, held or not; fails_when=an approval or a consent waits out the second; no_claim=none beyond protects.
  test("a step that waits for the User shows at once", () => {
    const { hold, kinds } = held(reading);
    vi.advanceTimersByTime(100);
    hold.offer(waiting);
    expect(kinds()).toEqual(["reading", "waiting"]);
    vi.advanceTimersByTime(100);
    hold.offer({ kind: "consent" });
    expect(kinds()).toEqual(["reading", "waiting", "consent"]);
  });

  // protects=a step that goes back to the one on show leaves nothing waiting; fails_when=the step that was over shows after all; no_claim=none beyond protects.
  test("a step that is over before its turn never shows", () => {
    const { hold, kinds } = held(reading);
    vi.advanceTimersByTime(100);
    hold.offer(searching);
    vi.advanceTimersByTime(100);
    hold.offer(reading);
    vi.advanceTimersByTime(10 * MIN_STEP_MS);
    expect(kinds()).toEqual(["reading"]);
  });

  // protects=stopping the hold drops the step that waits; fails_when=a step shows after the line is gone; no_claim=none beyond protects.
  test("a step that waits is dropped when the hold stops", () => {
    const { hold, kinds } = held(reading);
    vi.advanceTimersByTime(100);
    hold.offer(searching);
    hold.stop();
    vi.advanceTimersByTime(10 * MIN_STEP_MS);
    expect(kinds()).toEqual(["reading"]);
  });
});
