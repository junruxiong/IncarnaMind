import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import {
  blockTimeOf,
  blockTimes,
  meetingTimeAt,
  setBlockTimes,
  writingTime,
} from "../../src/shared/blockTimes";

const since = "2026-10-12T10:00:00.000Z";
const at = (seconds: number) => Date.parse(since) + seconds * 1000;
const recording = (recordedMs: number) => ({
  status: "recording" as const,
  recordedMs,
  recordingSince: since,
});

describe("the Meeting's time a line takes", () => {
  test("is what was recorded before, plus the stretch being recorded", () => {
    expect(meetingTimeAt(recording(60_000), at(5))).toBe(65_000);
    expect(meetingTimeAt({ recordedMs: 90_000, recordingSince: null }, at(500))).toBe(90_000);
  });

  test("never runs backwards when the clocks differ a little", () => {
    expect(meetingTimeAt(recording(1_000), at(-3))).toBe(1_000);
  });

  test("is taken only while audio is recorded: not paused, stopped or ready, and not in a Mind with no Meeting", () => {
    expect(writingTime(recording(0), at(2))).toBe(2_000);
    for (const status of ["paused", "stopped", "ready"] as const) {
      expect(writingTime({ status, recordedMs: 8_000, recordingSince: null }, at(2))).toBeNull();
    }
    expect(writingTime(undefined, at(2))).toBeNull();
  });
});

describe("the times kept beside a Mind's Blocks", () => {
  test("a line keeps the time it first took", () => {
    const doc = new Y.Doc();
    expect(setBlockTimes(doc, ["a", "b"], 1_234.6)).toEqual(["a", "b"]);
    expect(setBlockTimes(doc, ["a", "c"], 9_000)).toEqual(["c"]);
    expect(blockTimeOf(doc, "a")).toBe(1_235);
    expect(blockTimeOf(doc, "c")).toBe(9_000);
    expect(blockTimeOf(doc, "missing")).toBeNull();
    expect([...blockTimes(doc)].sort()).toEqual([
      ["a", 1_235],
      ["b", 1_235],
      ["c", 9_000],
    ]);
  });

  test("travel in the document's updates, as sync and reload carry them", () => {
    const first = new Y.Doc();
    setBlockTimes(first, ["a"], 42_000);
    const second = new Y.Doc();
    Y.applyUpdate(second, Y.encodeStateAsUpdate(first));
    expect(blockTimeOf(second, "a")).toBe(42_000);
  });

  test("a value that isn't a time reads as none", () => {
    const doc = new Y.Doc();
    doc.getMap("blockTimes").set("a", "soon");
    doc.getMap("blockTimes").set("b", -5);
    expect(blockTimeOf(doc, "a")).toBeNull();
    expect(blockTimeOf(doc, "b")).toBeNull();
    expect(blockTimes(doc).size).toBe(0);
  });
});
