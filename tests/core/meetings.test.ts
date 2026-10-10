import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, test, vi } from "vitest";
import type { Core, Meeting } from "../../src/core";
import { MEETING_AUDIO_KEPT_MS } from "../../src/core/api";
import { listParts, meetingFolder } from "../../src/core/meetings/audioFiles";
import { createFakeCapture, type FakeCaptureOptions } from "../../src/core/meetings/fakeCapture";
import { createTempDataFolder, manualClock, nextEvent, startCore } from "../helpers/core";

const SECOND = 1000;
const MINUTE = 60 * SECOND;

/**
 * A core on `dataDir` with a fake capture and a clock the test moves, with
 * Meetings turned on and the recording notice read, as Settings and the
 * one-time notice leave them.
 */
async function meetingCore(
  dataDir: string,
  clock = manualClock(),
  captureOptions: FakeCaptureOptions = {},
) {
  const capture = createFakeCapture(captureOptions);
  const core = startCore(dataDir, { audioCapture: capture, now: clock.now });
  await core.updateSettings({ device: { meetingsEnabled: true, meetingNoticeAccepted: true } });
  /** `ms` of audio from every source asked for, as captured, the clock moving with it. */
  const record = (ms: number, sources: ("microphone" | "system")[] = ["microphone", "system"]) => {
    for (let at = 0; at < ms; at += 500) {
      const step = Math.min(500, ms - at);
      clock.advance(step);
      for (const source of sources) capture.push(source, step);
    }
  };
  return { core, capture, clock, record };
}

const meetingOf = async (core: Core, meetingId: string): Promise<Meeting> => {
  const meeting = (await core.listMeetings()).find((each) => each.id === meetingId);
  if (!meeting) throw new Error(`No Meeting ${meetingId}.`);
  return meeting;
};

/** The parts on disk, as "source@offset+duration" in seconds. */
const partsOf = (dataDir: string, meetingId: string) =>
  listParts(meetingFolder(dataDir, meetingId)).map(
    (part) => `${part.source}@${part.offsetMs / SECOND}+${part.durationMs / SECOND}`,
  );

afterEach(() => {
  vi.useRealTimers();
});

describe("Recording a Meeting", () => {
  test("a Meeting starts in a new Mind named for the day, in the interface language", async () => {
    const dataDir = await createTempDataFolder();
    const { core } = await meetingCore(dataDir);

    const meeting = await core.createMeeting();
    const mind = (await core.listMinds()).find((each) => each.id === meeting.mindId);
    expect(mind?.title).toBe("Meeting, 6 October");
    expect(meeting).toMatchObject({
      status: "ready",
      systemAudio: true,
      sources: { microphone: "off", system: "off" },
      recordedMs: 0,
      audio: { kept: false, expiresAt: null, deletedAt: null },
    });

    await core.updateSettings({ user: { language: "zh-CN" }, device: { meetingSystemAudio: false } });
    const chinese = await core.createMeeting();
    const chineseMind = (await core.listMinds()).find((each) => each.id === chinese.mindId);
    expect(chineseMind?.title).toBe("会议，10月6日");
    expect(chinese.systemAudio).toBe(false);
  });

  test("the microphone and the computer's sound go to disk in 10-second parts as they come", async () => {
    const dataDir = await createTempDataFolder();
    const { core, record } = await meetingCore(dataDir);
    const { id } = await core.createMeeting();

    const started = await core.startMeeting({ meetingId: id });
    expect(started).toMatchObject({
      status: "recording",
      sources: { microphone: "recording", system: "recording" },
      recordingSince: "2026-10-06T09:00:00.000Z",
    });
    record(25 * SECOND);
    // Two whole parts each; the five seconds still filling aren't on disk yet.
    expect(partsOf(dataDir, id)).toEqual([
      "microphone@0+10",
      "system@0+10",
      "microphone@10+10",
      "system@10+10",
    ]);

    const stopped = await core.stopMeeting(id);
    expect(partsOf(dataDir, id).slice(-2)).toEqual(["microphone@20+5", "system@20+5"]);
    expect(stopped).toMatchObject({
      status: "stopped",
      recordedMs: 25 * SECOND,
      recordingSince: null,
      sources: { microphone: "off", system: "off" },
      gaps: [],
      audio: { kept: true, expiresAt: "2026-10-07T09:00:25.000Z", deletedAt: null },
    });
    // Each part is a playable WAV: 16-bit mono PCM at 16 kHz.
    const [first] = listParts(meetingFolder(dataDir, id));
    const wav = readFileSync(first?.path ?? "");
    expect(wav.subarray(0, 4).toString()).toBe("RIFF");
    expect(wav.readUInt32LE(24)).toBe(16_000);
    expect(wav.length).toBe(44 + 10 * 16_000 * 2);
  });

  test("pausing keeps the Meeting's time: what comes after a pause carries on from where it stopped", async () => {
    const dataDir = await createTempDataFolder();
    const { core, clock, record } = await meetingCore(dataDir);
    const { id } = await core.createMeeting();
    await core.startMeeting({ meetingId: id, systemAudio: false });
    record(12 * SECOND, ["microphone"]);

    const paused = await core.pauseMeeting(id);
    expect(paused).toMatchObject({ status: "paused", pausedBy: "user", recordedMs: 12 * SECOND });
    clock.advance(5 * MINUTE);
    await core.resumeMeeting(id);
    record(10 * SECOND, ["microphone"]);
    const stopped = await core.stopMeeting(id);

    expect(stopped.recordedMs).toBe(22 * SECOND);
    expect(stopped.gaps).toEqual([]);
    expect(partsOf(dataDir, id)).toEqual([
      "microphone@0+10",
      "microphone@10+2",
      "microphone@12+10",
    ]);
  });

  test("killed mid-recording, IncarnaMind recovers every part up to the last 10 seconds, and marks the gap", async () => {
    const dataDir = await createTempDataFolder();
    const clock = manualClock();
    const first = await meetingCore(dataDir, clock);
    const { id } = await first.core.createMeeting();
    await first.core.startMeeting({ meetingId: id });
    first.record(34 * SECOND);
    // A part being written when it died.
    writeFileSync(join(meetingFolder(dataDir, id), "microphone-0000030000.wav.writing"), "half");

    // Killed: never closed. Relaunched two minutes later.
    clock.advance(2 * MINUTE);
    const second = await meetingCore(dataDir, clock);
    const recovered = await meetingOf(second.core, id);
    expect(recovered).toMatchObject({
      status: "paused",
      pausedBy: "interrupted",
      recordedMs: 30 * SECOND,
      recordingSince: null,
      audio: { kept: true, expiresAt: "2026-10-07T09:00:30.000Z" },
    });
    expect(partsOf(dataDir, id)).toEqual([
      "microphone@0+10",
      "system@0+10",
      "microphone@10+10",
      "system@10+10",
      "microphone@20+10",
      "system@20+10",
    ]);
    expect(readdirSync(meetingFolder(dataDir, id)).some((name) => name.endsWith(".writing"))).toBe(
      false,
    );

    // Resumed, it records on from the last saved part; what was missed is a gap there.
    const resumed = await second.core.resumeMeeting(id);
    expect(resumed.gaps).toEqual([
      { atMs: 30 * SECOND, missedMs: 4 * SECOND + 2 * MINUTE, reason: "interrupted" },
    ]);
    second.record(10 * SECOND);
    expect(partsOf(dataDir, id).slice(-2)).toEqual(["microphone@30+10", "system@30+10"]);
  });

  test("quitting while recording saves what was still filling", async () => {
    const dataDir = await createTempDataFolder();
    const clock = manualClock();
    const first = await meetingCore(dataDir, clock);
    const { id } = await first.core.createMeeting();
    await first.core.startMeeting({ meetingId: id, systemAudio: false });
    first.record(14 * SECOND, ["microphone"]);
    first.core.close();
    expect(first.capture.running).toBe(false);

    const second = await meetingCore(dataDir, clock);
    expect(await meetingOf(second.core, id)).toMatchObject({
      status: "paused",
      pausedBy: "interrupted",
      recordedMs: 14 * SECOND,
    });
    expect(partsOf(dataDir, id)).toEqual(["microphone@0+10", "microphone@10+4"]);
  });

  test("the computer sleeping is a gap, and recording goes on when it wakes", async () => {
    const dataDir = await createTempDataFolder();
    const { core, clock, record, capture } = await meetingCore(dataDir);
    const { id } = await core.createMeeting();
    await core.startMeeting({ meetingId: id, systemAudio: false });
    record(15 * SECOND, ["microphone"]);

    core.systemSleeping();
    expect(capture.running).toBe(false);
    expect(partsOf(dataDir, id)).toEqual(["microphone@0+10", "microphone@10+5"]);
    expect(await meetingOf(core, id)).toMatchObject({
      status: "recording",
      recordingSince: null,
      recordedMs: 15 * SECOND,
    });

    clock.advance(30 * MINUTE);
    core.systemAwake();
    record(10 * SECOND, ["microphone"]);
    const meeting = await meetingOf(core, id);
    expect(meeting).toMatchObject({
      status: "recording",
      sources: { microphone: "recording" },
      gaps: [{ atMs: 15 * SECOND, missedMs: 30 * MINUTE, reason: "sleep" }],
    });
    expect(partsOf(dataDir, id).at(-1)).toBe("microphone@15+10");
  });

  test("a device change keeps recording: the new microphone's audio goes on at the right time", async () => {
    const dataDir = await createTempDataFolder();
    const { core, clock, capture, record } = await meetingCore(dataDir);
    const { id } = await core.createMeeting();
    await core.startMeeting({ meetingId: id, systemAudio: false });
    record(6 * SECOND, ["microphone"]);

    // The old device goes; the new one's samples start two seconds later.
    clock.advance(2 * SECOND);
    record(12 * SECOND, ["microphone"]);
    expect((await meetingOf(core, id)).status).toBe("recording");
    expect(capture.started).toBe(1);

    await core.stopMeeting(id);
    expect(partsOf(dataDir, id)).toEqual([
      "microphone@0+6",
      "microphone@8+10",
      "microphone@18+2",
    ]);
  });
});

describe("The computer's sound", () => {
  test("denied, the Meeting says so and records the microphone on its own", async () => {
    const dataDir = await createTempDataFolder();
    const { core, record } = await meetingCore(dataDir, manualClock(), { system: "denied" });
    const { id } = await core.createMeeting();
    const started = await core.startMeeting({ meetingId: id });
    expect(started.sources).toEqual({ microphone: "recording", system: "denied" });

    record(10 * SECOND);
    expect(partsOf(dataDir, id)).toEqual(["microphone@0+10"]);
  });

  test("a minute of nothing but silence is reported, and sound coming back clears it", async () => {
    const dataDir = await createTempDataFolder();
    const { core, capture, record } = await meetingCore(dataDir, manualClock(), {
      system: "silent",
    });
    const { id } = await core.createMeeting();
    await core.startMeeting({ meetingId: id });
    record(59 * SECOND);
    expect((await meetingOf(core, id)).sources.system).toBe("recording");
    const silent = nextEvent(core, "meetings.changed");
    record(SECOND);
    expect((await silent).find((each) => each.id === id)?.sources).toEqual({
      microphone: "recording",
      system: "silent",
    });

    capture.report("system", "ended");
    expect((await meetingOf(core, id)).sources.system).toBe("ended");
  });

  test("a computer that can't record its own sound records the microphone", async () => {
    const dataDir = await createTempDataFolder();
    const { core, record } = await meetingCore(dataDir, manualClock(), { systemAudio: false });
    const { id } = await core.createMeeting();
    const started = await core.startMeeting({ meetingId: id });
    expect(started.sources).toEqual({ microphone: "recording", system: "unavailable" });
    record(10 * SECOND);
    expect(partsOf(dataDir, id)).toEqual(["microphone@0+10"]);
  });

  test("the link to System Settings goes to the capture", async () => {
    const dataDir = await createTempDataFolder();
    const { core, capture } = await meetingCore(dataDir);
    await core.openAudioSettings("system");
    expect(capture.settingsOpened).toEqual(["system"]);
    await expect(core.openAudioSettings("camera" as "system")).rejects.toThrow(/source/);
  });
});

describe("A Meeting's audio is kept for at most a day", () => {
  test("it is deleted a day after the Meeting, while IncarnaMind runs or at its next start", async () => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    const dataDir = await createTempDataFolder();
    const clock = manualClock();
    const { core, record } = await meetingCore(dataDir, clock);
    const { id } = await core.createMeeting();
    await core.startMeeting({ meetingId: id });
    record(12 * SECOND);
    await core.stopMeeting(id);

    clock.advance(MEETING_AUDIO_KEPT_MS - MINUTE);
    vi.advanceTimersByTime(MINUTE);
    expect((await meetingOf(core, id)).audio.kept).toBe(true);

    clock.advance(2 * MINUTE);
    const deleted = nextEvent(core, "meetings.changed");
    vi.advanceTimersByTime(MINUTE);
    expect((await deleted).find((each) => each.id === id)?.audio).toEqual({
      kept: false,
      expiresAt: null,
      deletedAt: "2026-10-07T09:01:12.000Z",
    });
    expect(existsSync(meetingFolder(dataDir, id))).toBe(false);

    // A Meeting whose day ran out while IncarnaMind was closed loses its audio at the next start.
    const { id: other } = await core.createMeeting();
    await core.startMeeting({ meetingId: other });
    record(3 * SECOND);
    await core.pauseMeeting(other);
    core.close();
    clock.advance(MEETING_AUDIO_KEPT_MS + MINUTE);
    const restarted = await meetingCore(dataDir, clock);
    expect((await meetingOf(restarted.core, other)).audio.kept).toBe(false);
    expect(existsSync(meetingFolder(dataDir, other))).toBe(false);
  });

  test("the User can delete it at any time, while recording too", async () => {
    const dataDir = await createTempDataFolder();
    const { core, record } = await meetingCore(dataDir);
    const { id } = await core.createMeeting();
    await core.startMeeting({ meetingId: id, systemAudio: false });
    record(15 * SECOND, ["microphone"]);

    const deleted = await core.deleteMeetingAudio(id);
    expect(deleted.audio).toEqual({
      kept: false,
      expiresAt: null,
      deletedAt: "2026-10-06T09:00:15.000Z",
    });
    expect(existsSync(meetingFolder(dataDir, id))).toBe(false);

    // What comes after is kept, from the time it comes.
    record(10 * SECOND, ["microphone"]);
    await core.stopMeeting(id);
    expect(partsOf(dataDir, id)).toEqual(["microphone@15+10"]);
  });
});

describe("What a Meeting needs", () => {
  test("Meetings turned on, and the recording notice read, before recording starts", async () => {
    const dataDir = await createTempDataFolder();
    const capture = createFakeCapture();
    const core = startCore(dataDir, { audioCapture: capture });
    await expect(core.createMeeting()).rejects.toThrow(/turned off/);

    await core.updateSettings({ device: { meetingsEnabled: true } });
    const { id } = await core.createMeeting();
    await expect(core.startMeeting({ meetingId: id })).rejects.toThrow(/notice/);
    expect(capture.started).toBe(0);

    await core.updateSettings({ device: { meetingNoticeAccepted: true } });
    expect((await core.startMeeting({ meetingId: id })).status).toBe("recording");
  });

  test("one Meeting records at a time", async () => {
    const dataDir = await createTempDataFolder();
    const { core } = await meetingCore(dataDir);
    const first = await core.createMeeting();
    const second = await core.createMeeting();
    await core.startMeeting({ meetingId: first.id });
    await expect(core.startMeeting({ meetingId: second.id })).rejects.toThrow(/Another Meeting/);

    await core.pauseMeeting(first.id);
    await core.startMeeting({ meetingId: second.id });
    await expect(core.resumeMeeting(first.id)).rejects.toThrow(/Another Meeting/);
  });

  test("a Meeting not started yet can be taken back, keeping its Mind as a note", async () => {
    const dataDir = await createTempDataFolder();
    const { core } = await meetingCore(dataDir);
    const meeting = await core.createMeeting();
    await core.discardMeeting(meeting.id);
    expect(await core.listMeetings()).toEqual([]);
    expect((await core.listMinds()).map((mind) => mind.id)).toEqual([meeting.mindId]);

    // An empty Mind of the User's can hold a Meeting, once.
    const mind = await core.createMind({ title: "Stand-up" });
    const inMind = await core.createMeeting({ mindId: mind.id });
    expect(inMind.mindId).toBe(mind.id);
    await expect(core.createMeeting({ mindId: mind.id })).rejects.toThrow(/already/);
    await core.startMeeting({ meetingId: inMind.id });
    await expect(core.discardMeeting(inMind.id)).rejects.toThrow(/hasn't started/);
  });

  test("deleting the Mind stops its Meeting and deletes its audio", async () => {
    const dataDir = await createTempDataFolder();
    const { core, capture, record } = await meetingCore(dataDir);
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id });
    record(12 * SECOND);

    await core.deleteMind(meeting.mindId);
    expect(capture.running).toBe(false);
    expect(await core.listMeetings()).toEqual([]);
    expect(existsSync(meetingFolder(dataDir, meeting.id))).toBe(false);
  });
});
