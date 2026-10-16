import { describe, expect, test } from "vitest";
import type { Cue } from "../../src/core/documents/formats/vtt";
import { echoedBy, isEcho } from "../../src/core/meetings/echoLines";

const line = (speaker: string, startMs: number, endMs: number, text: string): Cue => ({
  startMs,
  endMs,
  text,
  speaker,
});

const THEIRS = "The major religion in Moldova is Orthodox Christian.";

describe("Lines of the microphone that are the echo of the other side's (#941)", () => {
  test("the same words at the same time are the echo, whatever the case and punctuation", () => {
    const them = line("Them", 10_000, 13_000, THEIRS);
    const me = line("Me", 10_050, 13_100, "the major religion in moldova is orthodox christian");
    expect(isEcho(me, [them, me])).toBe(true);
  });

  test("so are some of their words, as an echo that is read badly gives them", () => {
    const them = line("Them", 10_000, 13_000, THEIRS);
    const me = line("Me", 10_300, 12_600, "major religion in Moldova is Orthodox");
    expect(isEcho(me, [them, me])).toBe(true);
  });

  test("an echo cut in different places is still found in what was said around it", () => {
    const first = line("Them", 10_000, 12_000, "The major religion in Moldova");
    const second = line("Them", 11_900, 14_000, "is Orthodox Christian.");
    const me = line("Me", 10_100, 13_900, THEIRS);
    expect(isEcho(me, [first, second, me])).toBe(true);
  });

  test("in Chinese too", () => {
    const them = line("Them", 4000, 7000, "西班牙人开始了长达三个世纪的殖民时期。");
    expect(isEcho(line("Me", 4100, 6900, "西班牙人开始了长达三个世纪的殖民"), [them])).toBe(true);
    expect(isEcho(line("Me", 4100, 6900, "他称，他制作了一个 WiFi 门铃。"), [them])).toBe(false);
  });

  test("other words at the same time are the User's own", () => {
    const them = line("Them", 10_000, 13_000, THEIRS);
    const me = line("Me", 10_200, 12_800, "Can I stop you there and ask about the budget?");
    expect(isEcho(me, [them, me])).toBe(false);
  });

  test("the same words said after the other side finished are a read-back, not an echo", () => {
    const them = line("Them", 10_000, 13_000, THEIRS);
    // A line is decoded from a quarter of a second before the speech: it only touches theirs.
    const readBack = line("Me", 12_750, 15_500, THEIRS);
    expect(isEcho(readBack, [them, readBack])).toBe(false);
    expect(isEcho(line("Me", 20_000, 23_000, THEIRS), [them])).toBe(false);
  });

  test("a short answer over their long line is the User's, unless it is a whole line of theirs", () => {
    const them = line("Them", 10_000, 16_000, "So we should go with the first option, right?");
    expect(isEcho(line("Me", 15_000, 15_600, "Right."), [them])).toBe(false);
    const yes = line("Them", 10_000, 10_700, "Yes.");
    expect(isEcho(line("Me", 10_000, 10_700, "Yes."), [yes])).toBe(true);
  });

  test("only a line of the microphone can be an echo, and only of the computer's sound", () => {
    const them = line("Them", 10_000, 13_000, THEIRS);
    expect(isEcho(line("Them", 10_000, 13_000, THEIRS), [them])).toBe(false);
    expect(isEcho(line("Speaker ?", 10_000, 13_000, THEIRS), [them])).toBe(false);
    expect(
      isEcho(line("Me", 10_000, 13_000, THEIRS), [line("Speaker ?", 10_000, 13_000, THEIRS)]),
    ).toBe(false);
  });

  test("a line of theirs that comes after the echo of it finds the echo", () => {
    const me = line("Me", 10_050, 13_100, THEIRS);
    const elsewhere = line("Me", 30_000, 32_000, "Let me check the date.");
    const them = line("Them", 10_000, 13_000, THEIRS);
    expect(echoedBy(them, [me, elsewhere])).toEqual([]);
    expect(echoedBy(them, [me, elsewhere, them])).toEqual([me]);
  });
});
