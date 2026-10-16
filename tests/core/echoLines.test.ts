import { describe, expect, test } from "vitest";
import type { Cue } from "../../src/core/documents/formats/vtt";
import { echoedBy, isEcho, repeats } from "../../src/core/meetings/echoLines";

const line = (speaker: string, startMs: number, endMs: number, text: string): Cue => ({
  startMs,
  endMs,
  text,
  speaker,
});

const THEIRS = "The major religion in Moldova is Orthodox Christian.";
const LONG_THEIRS =
  "So I think the contingency for the second phase is too small, and that works for me as long as we can agree that the review is done before the end of the month, thanks.";

describe("Lines of the microphone that are the echo of the other side's (#941)", () => {
  test("the same words at the same moment are the echo, whatever the case and punctuation", () => {
    const them = line("Them", 10_000, 13_000, THEIRS);
    const me = line("Me", 10_050, 13_100, "the major religion in moldova is orthodox christian");
    expect(isEcho(me, [them, me])).toBe(true);
  });

  test("so are most of their words in their order, as an echo that is read badly gives them", () => {
    const them = line("Them", 10_000, 13_000, THEIRS);
    expect(
      isEcho(line("Me", 10_300, 12_600, "major religion in Moldova is Orthodox"), [them]),
    ).toBe(true);
    expect(
      isEcho(line("Me", 10_100, 13_000, "The major region in Moldova is Orthodox Christian"), [
        them,
      ]),
    ).toBe(true);
  });

  test("an echo cut in different places is found in the lines of theirs around it, in time order", () => {
    const first = line("Them", 10_000, 12_000, "The major religion in Moldova");
    const second = line("Them", 11_900, 14_000, "is Orthodox Christian.");
    const me = line("Me", 10_100, 13_900, THEIRS);
    expect(isEcho(me, [second, first, me])).toBe(true);
  });

  test("in Chinese, a character at a time", () => {
    const them = line("Them", 4000, 7000, "西班牙人开始了长达三个世纪的殖民时期。");
    expect(isEcho(line("Me", 4100, 6900, "西班牙人开始了长达三个世纪的殖民"), [them])).toBe(true);
    expect(isEcho(line("Me", 4100, 6900, "他称，他制作了一个 WiFi 门铃。"), [them])).toBe(false);
  });

  test.each([
    "Dana thinks the contingency is too small",
    "Sure, that works for me, thanks",
    "That sounds good to me",
    "I think the review is done",
    "The second phase works for me as long as the month ends well",
  ])(
    "the User's own words are kept over a long line of theirs that has the same common words: %s",
    (said) => {
      const them = line("Them", 10_000, 20_000, LONG_THEIRS);
      const me = line("Me", 10_400, 14_000, said);
      expect(isEcho(me, [them, me])).toBe(false);
    },
  );

  test.each([
    "Yes, that works for me.",
    "Right, thanks, that sounds good.",
    "Sure, I agree with that.",
  ])("a polite reply over their sentence is the User's: %s", (said) => {
    const them = line(
      "Them",
      10_000,
      14_000,
      "Does that work for you, and do you agree with the plan?",
    );
    expect(isEcho(line("Me", 12_800, 14_500, said), [them])).toBe(false);
  });

  test("a short line with a word of its own is the User's, though most of its words are theirs", () => {
    const them = line(
      "Them",
      10_000,
      14_000,
      "That sounds like a plan to me, shall we start on Monday?",
    );
    expect(isEcho(line("Me", 10_300, 12_500, "That sounds good to me"), [them])).toBe(false);
    expect(isEcho(line("Me", 10_300, 12_500, "That sounds like a plan to me"), [them])).toBe(true);
    const good = line("Them", 10_000, 13_000, "That sounds good to me");
    expect(isEcho(line("Me", 10_200, 12_800, "That sounds good to me"), [good])).toBe(true);
    expect(isEcho(line("Me", 10_200, 12_800, "That sounds great to me"), [good])).toBe(false);
  });

  test("a longer line may miss a word of theirs: the echo is read badly", () => {
    const them = line(
      "Them",
      10_000,
      15_000,
      "Please send the revised budget to the whole committee before Friday",
    );
    expect(
      isEcho(
        line(
          "Me",
          10_200,
          14_800,
          "Please send the revised budget to the whole committee by Friday",
        ),
        [them],
      ),
    ).toBe(true);
  });

  test("the same words in another order are the User's", () => {
    const them = line("Them", 10_000, 13_000, THEIRS);
    expect(
      isEcho(line("Me", 10_100, 13_000, "Christian Orthodox is Moldova in religion major the"), [
        them,
      ]),
    ).toBe(false);
  });

  test("the same words said after the other side started a moment before are kept: only their echo is dropped", () => {
    const them = line("Them", 10_000, 13_000, THEIRS);
    expect(isEcho(line("Me", 12_750, 15_500, THEIRS), [them])).toBe(false);
    expect(isEcho(line("Me", 20_000, 23_000, THEIRS), [them])).toBe(false);
  });

  test("a short answer is the User's, unless it is a whole line of theirs", () => {
    const them = line("Them", 10_000, 16_000, "So we should go with the first option, right?");
    expect(isEcho(line("Me", 15_000, 15_600, "Right."), [them])).toBe(false);
    const yes = line("Them", 10_000, 10_700, "Yes.");
    expect(isEcho(line("Me", 10_000, 10_700, "Yes."), [yes])).toBe(true);
  });

  test("only a line of the microphone can be an echo, and only of the computer's sound", () => {
    const them = line("Them", 10_000, 13_000, THEIRS);
    expect(isEcho(line("Them", 10_000, 13_000, THEIRS), [them])).toBe(false);
    expect(isEcho(line("Speaker ?", 10_000, 13_000, THEIRS), [them])).toBe(false);
    const unknown = line("Speaker ?", 10_000, 13_000, THEIRS);
    expect(isEcho(line("Me", 10_000, 13_000, THEIRS), [unknown])).toBe(false);
  });

  test("a line of theirs that comes after the echo of it finds the echo", () => {
    const me = line("Me", 10_050, 13_100, THEIRS);
    const elsewhere = line("Me", 30_000, 32_000, "Let me check the date.");
    const them = line("Them", 10_000, 13_000, THEIRS);
    expect(echoedBy(them, [me, elsewhere])).toEqual([]);
    expect(echoedBy(them, [me, elsewhere, them])).toEqual([me]);
  });

  test("words said while something else is shown greyed follow the same rule", () => {
    expect(repeats("The major religion in Moldova", [THEIRS])).toBe(true);
    expect(repeats("Can I stop you there", [THEIRS])).toBe(false);
    expect(repeats("", [THEIRS])).toBe(false);
    expect(repeats(THEIRS, [])).toBe(false);
  });
});
