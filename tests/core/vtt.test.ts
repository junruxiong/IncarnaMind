import { describe, expect, test } from "vitest";
import type { CitationLocation } from "../../src/core";
import { withCueMarks } from "../../src/core/answers/cueMarks";
import { extractUnits } from "../../src/core/documents/formats";
import { readVtt, transcriptUnits, writeVtt } from "../../src/core/documents/formats/vtt";
import { badgeMessage } from "../../src/shared/citations";
import { clockTime, parseClockTime } from "../../src/shared/clock";
import { translate } from "../../src/shared/i18n";
import {
  formatLocation,
  locationOf,
  parseLocation,
  parseRequestedLocation,
  quoteInUnits,
  resolveRequested,
} from "../../src/shared/locations";

const FILE = `WEBVTT - Steering committee

NOTE
Written by hand.

1
00:00:40.000 --> 00:00:44.500 align:start
<v Dana>So the launch moves to the eighteenth
of November &amp; the review &lt;slips&gt;.</v>

NOTE gap 00:12:07.000 to 00:12:19.000 restarted

00:12:20.000 --> 00:12:25.000
Two weeks. Can the contingency cover it?

STYLE
::cue { color: white }

01:02:03.450 --> 01:02:05.000
发布日期推迟到十一月十八日。
`;

const en = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
  translate("en", key, params);

describe("Transcripts as WebVTT", () => {
  test("cues are read with their times and speakers, notes and styles left out, gaps kept", () => {
    expect(readVtt(FILE)).toEqual({
      cues: [
        {
          startMs: 40_000,
          endMs: 44_500,
          text: "So the launch moves to the eighteenth of November & the review <slips>.",
          speaker: "Dana",
        },
        {
          startMs: 740_000,
          endMs: 745_000,
          text: "Two weeks. Can the contingency cover it?",
          speaker: null,
        },
        {
          startMs: 3_723_450,
          endMs: 3_725_000,
          text: "发布日期推迟到十一月十八日。",
          speaker: null,
        },
      ],
      gaps: [{ fromMs: 727_000, toMs: 739_000, reason: "restarted" }],
    });
  });

  test("what is written reads back the same, with what WebVTT can't hold escaped", () => {
    const transcript = {
      cues: [
        { startMs: 0, endMs: 1500, text: "A --> B & <c>", speaker: "Speaker 1" },
        { startMs: 2000, endMs: 3000, text: "第二句。", speaker: null },
      ],
      gaps: [{ fromMs: 1500, toMs: 2000, reason: "lost" as const }],
    };
    const written = writeVtt(transcript);
    expect(written).toBe(
      "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n<v Speaker 1>A --&gt; B &amp; &lt;c&gt;\n\n" +
        "NOTE gap 00:00:01.500 to 00:00:02.000 lost\n\n00:00:02.000 --> 00:00:03.000\n第二句。\n",
    );
    expect(readVtt(written)).toEqual(transcript);
    const ampersand = {
      cues: [{ startMs: 0, endMs: 900, text: "Yes.", speaker: "R&D" }],
      gaps: [],
    };
    expect(writeVtt(ampersand)).toContain("<v R&amp;D>Yes.");
    expect(readVtt(writeVtt(ampersand))).toEqual(ampersand);
  });

  test("a file that isn't WebVTT is refused", () => {
    expect(() => readVtt("1\n00:00:01,000 --> 00:00:02,000\nAn SRT file\n")).toThrow(
      "doesn't start with WEBVTT",
    );
  });

  test("Units are blocks of cues, each cue a line anchored by its times", async () => {
    const units = await extractUnits("transcript", new TextEncoder().encode(FILE));
    expect(units).toEqual([
      {
        page: 1,
        kind: "cues",
        label: { from: 40_000, to: 3_725_000, speakers: ["Dana", null, null] },
        text: [
          "So the launch moves to the eighteenth of November & the review <slips>.",
          "Two weeks. Can the contingency cover it?",
          "发布日期推迟到十一月十八日。",
        ].join("\n"),
        anchors: [
          { start: 0, end: 71, target: "t40000-44500" },
          { start: 72, end: 112, target: "t740000-745000" },
          { start: 113, end: 127, target: "t3723450-3725000" },
        ],
      },
    ]);
    const many = transcriptUnits({
      cues: Array.from({ length: 100 }, (_, index) => ({
        startMs: index * 1000,
        endMs: index * 1000 + 900,
        text: `Line ${index}`,
        speaker: null,
      })),
      gaps: [],
    });
    expect(many.map((unit) => unit.label)).toEqual([
      { from: 0, to: 39_900 },
      { from: 40_000, to: 79_900 },
      { from: 80_000, to: 99_900 },
    ]);
  });
});

describe("Citations of a Transcript", () => {
  const units = transcriptUnits(readVtt(FILE));

  test("a Location is the time its quote starts, through the end of the cue it ends in", () => {
    const quote = "the contingency cover it";
    const ranges = quoteInUnits(units, quote);
    expect(locationOf(units, ranges)).toEqual({ kind: "time", from: 740_000, to: 745_000 });
    // A quote across two cues spans both.
    const across = quoteInUnits(units, "the review <slips>. Two weeks.");
    expect(locationOf(units, across)).toEqual({ kind: "time", from: 40_000, to: 745_000 });
    // Not found, it is the cited block's.
    expect(locationOf(units, null)).toEqual({ kind: "time", from: 40_000, to: 3_725_000 });
  });

  test("its label is the clock time, and its check is labelled the transcript's", () => {
    const location: CitationLocation = { kind: "time", from: 740_000, to: 745_000 };
    expect(formatLocation(location, en)).toBe("12:20");
    expect(formatLocation({ kind: "time", from: 3_723_450, to: 3_725_000 }, en)).toBe("1:02:03");
    expect(parseLocation(JSON.stringify(location))).toEqual(location);
    expect(parseLocation({ kind: "time", from: -1, to: 2 })).toBeNull();
    const state = {
      check: "found" as const,
      reason: null,
      documentId: "d",
      changedAfterCited: false,
    };
    const badge = badgeMessage(state, { location, documentName: "Transcript · Steering" }, en);
    expect(en(badge.key, badge.params)).toBe("Quote found in the transcript at 12:20");
    const zh = badgeMessage({ ...state, check: "not-found" }, { location }, (key, params) =>
      translate("zh-CN", key, params),
    );
    expect(translate("zh-CN", zh.key, zh.params)).toBe("未在转写稿 12:20 处找到引文");
  });

  test("a time the model names finds the block it falls in", () => {
    const requested = parseRequestedLocation("12:20");
    expect(requested).toEqual({ kind: "time", from: 740_000, to: 740_000 });
    expect(parseRequestedLocation("at 1:02:03")).toEqual({
      kind: "time",
      from: 3_723_000,
      to: 3_723_000,
    });
    expect(resolveRequested(requested as NonNullable<typeof requested>, units)).toEqual({
      from: 1,
      to: 1,
    });
    expect(parseRequestedLocation("lines 3–4")).toEqual({ kind: "lines", from: 3, to: 4 });
  });

  test("the model reads each cue with its time before it", () => {
    const text = units[0]?.text ?? "";
    expect(withCueMarks(text, units)).toBe(
      [
        "[0:40 Dana] So the launch moves to the eighteenth of November & the review <slips>.",
        "[12:20] Two weeks. Can the contingency cover it?",
        "[1:02:03] 发布日期推迟到十一月十八日。",
      ].join("\n"),
    );
    // A line said twice is marked twice, each time on its own line.
    const twice = transcriptUnits({
      cues: [
        { startMs: 1000, endMs: 2000, text: "Yes.", speaker: null },
        { startMs: 3000, endMs: 4000, text: "Yes.", speaker: null },
        { startMs: 5000, endMs: 6000, text: "Yes", speaker: null },
      ],
      gaps: [],
    });
    expect(withCueMarks(twice[0]?.text ?? "", twice)).toBe("[0:01] Yes.\n[0:03] Yes.\n[0:05] Yes");
    // A Passage that starts part way through keeps the marks of the cues it holds.
    const tail = text.slice(text.indexOf("Two weeks"));
    expect(withCueMarks(tail, units)).toMatch(/^\[12:20\] Two weeks/);
  });

  test("clock times read back", () => {
    expect(clockTime(0)).toBe("0:00");
    expect(clockTime(866_000)).toBe("14:26");
    expect(parseClockTime("14:26")).toBe(866_000);
    expect(parseClockTime("1:02:05")).toBe(3_725_000);
    expect(parseClockTime("1:72:05")).toBeNull();
    expect(parseClockTime("14:6")).toBeNull();
  });
});
