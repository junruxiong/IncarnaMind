import { existsSync, readFileSync, realpathSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import type { Core } from "../../src/core";
import { readVtt } from "../../src/core/documents/formats/vtt";
import { FAKE_LINES, FAKE_THEIR_LINES } from "../../src/core/transcription/fake";
import { askAndFinish, citingModel, onlyCitation, type ShownPassage } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { transcribingCore, transcriptOf, until } from "../helpers/transcription";

const SECOND = 1000;

async function documentOf(core: Core, documentId: string) {
  const document = (await core.listDocuments()).find((each) => each.id === documentId);
  if (!document) throw new Error(`No Document ${documentId}.`);
  return document;
}

describe("A Meeting's live Transcript", { timeout: 60_000 }, () => {
  test("a recorded stream gives a Transcript Document from its first line; a Question cites a time, and the check finds it", async () => {
    const dataDir = await createTempDataFolder();
    const quote = FAKE_LINES[1] as string;
    const { core, record } = await transcribingCore(dataDir, {
      model: citingModel({
        query: "launch",
        records: (passages: ShownPassage[]) => [
          { marker: 1, passage: passages[0]?.id, quote: `[0:01] ${quote}` },
        ],
        answer: "The launch moves to the eighteenth of November [^1].",
      }),
    });
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id });
    expect((await transcriptOf(core, meeting.id)).state).toMatch(/preparing|live/);

    record(9 * SECOND);
    const live = await until(core, meeting.id, (transcript) => transcript.lines >= 4);
    expect(live).toMatchObject({ state: "live", savedPath: null });
    const documentId = live.documentId as string;
    const document = await documentOf(core, documentId);
    expect(document).toMatchObject({
      kind: "transcript",
      name: "Transcript · Meeting, 6 October",
      path: join(dataDir, "Transcripts", `${meeting.id}.vtt`),
    });
    const vtt = readVtt(readFileSync(document.path, "utf8"));
    const said = (speaker: string) => vtt.cues.filter((cue) => cue.speaker === speaker);
    expect(said("Me").slice(0, 2)).toEqual([
      { startMs: 0, endMs: 2000, text: FAKE_LINES[0], speaker: "Me" },
      // A stretch is decoded from a quarter of a second before the detector heard it.
      { startMs: 1750, endMs: 4000, text: FAKE_LINES[1], speaker: "Me" },
    ]);
    expect(said("Them").map((cue) => cue.text)).toEqual(
      FAKE_THEIR_LINES.slice(0, said("Them").length),
    );
    expect(vtt.cues.length).toBe(said("Me").length + said("Them").length);

    // Searchable and citable while the Meeting records, though other Documents wait.
    await vi.waitFor(
      async () => {
        const found = await core.searchPassages("eighteenth of November", { mode: "keyword" });
        expect(found.map((passage) => passage.documentId)).toContain(documentId);
      },
      { timeout: 20_000, interval: 50 },
    );
    const mind = await core.createMind({ title: "Launch" });
    const client = await connectToMind(core, mind.id);
    const { answerId } = await askAndFinish(core, client, mind.id, "When does the launch move to?");
    expect(onlyCitation(client, answerId)).toMatchObject({
      documentId,
      location: { kind: "time", from: 1750, to: 4000, speaker: "Me" },
      quote,
      check: "found",
    });

    // At Stop it is finished and saved as WebVTT in the Output location, the Document with it.
    await core.stopMeeting(meeting.id);
    const saved = await until(core, meeting.id, (transcript) => transcript.state === "saved");
    const path = join(
      realpathSync(dataDir),
      "Documents",
      "IncarnaMind",
      "Transcript · Meeting, 6 October.vtt",
    );
    expect(saved.savedPath).toBe(path);
    expect(readFileSync(path, "utf8")).toMatch(/^WEBVTT\n\n00:00:00\.000 --> 00:00:02\.000\n/);
    expect(existsSync(join(dataDir, "Transcripts", `${meeting.id}.vtt`))).toBe(false);
    await vi.waitFor(async () => expect((await documentOf(core, documentId)).path).toBe(path));
    // It stays where it was saved after a restart, not back in the data folder's Transcripts.
    core.close();
    expect(await documentOf(startCore(dataDir), documentId)).toMatchObject({
      path,
      fileStatus: "available",
    });
  });

  test("the engine's process crashes mid-meeting: it starts again, the gap is marked, and filled from the audio at Stop", async () => {
    const dataDir = await createTempDataFolder();
    const { core, record, engineHost } = await transcribingCore(dataDir);
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id });
    record(5 * SECOND);
    await until(core, meeting.id, (transcript) => transcript.lines >= 2);

    engineHost.crash();
    record(3 * SECOND);
    await vi.waitFor(() => expect(engineHost.started).toBe(2), { timeout: 10_000 });
    const path = join(dataDir, "Transcripts", `${meeting.id}.vtt`);
    await vi.waitFor(() => expect(readFileSync(path, "utf8")).toMatch(/^NOTE gap .* restarted$/m), {
      timeout: 10_000,
    });
    const gap = readVtt(readFileSync(path, "utf8")).gaps[0];
    expect(gap?.reason).toBe("restarted");
    record(6 * SECOND);
    await until(
      core,
      meeting.id,
      (transcript) => transcript.state === "live" && transcript.lines >= 4,
    );

    await core.stopMeeting(meeting.id);
    const saved = await until(core, meeting.id, (transcript) => transcript.state === "saved");
    const final = readVtt(readFileSync(saved.savedPath as string, "utf8"));
    expect(final.gaps).toEqual([]);
    expect(
      final.cues.some((cue) => cue.startMs >= (gap?.fromMs ?? 0) && cue.startMs < (gap?.toMs ?? 0)),
    ).toBe(true);
  });

  test("live text that falls behind says so, and the rest is transcribed from the audio after Stop", async () => {
    const dataDir = await createTempDataFolder();
    const { core, record } = await transcribingCore(dataDir, { engine: { decodeMs: 30 } });
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id });
    await until(core, meeting.id, (transcript) => transcript.state === "live");

    // Forty seconds arrive at once: far more than the engine has decoded.
    record(40 * SECOND);
    await until(core, meeting.id, (transcript) => transcript.state === "behind");

    await core.stopMeeting(meeting.id);
    const saved = await until(core, meeting.id, (transcript) => transcript.state === "saved");
    const final = readVtt(readFileSync(saved.savedPath as string, "utf8"));
    expect(final.cues.length).toBeGreaterThanOrEqual(19);
    expect(final.cues.at(-1)?.endMs).toBeGreaterThanOrEqual(38 * SECOND);
    // What was transcribed from the audio after Stop is labelled by its source too.
    const late = final.cues.filter((cue) => cue.startMs >= 30 * SECOND);
    expect(new Set(late.map((cue) => cue.speaker))).toEqual(new Set(["Me", "Them"]));
  });

  test("a Meeting where nothing was said saves no Transcript", async () => {
    const dataDir = await createTempDataFolder();
    const { core, record } = await transcribingCore(dataDir);
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id });
    record(5 * SECOND, []);
    await core.stopMeeting(meeting.id);
    const transcript = await until(core, meeting.id, (each) => each.state === "empty");
    expect(transcript).toMatchObject({ documentId: null, lines: 0, savedPath: null });
  });

  test("IncarnaMind closes while recording: what was recorded after live text got is transcribed once the Meeting stops", async () => {
    const dataDir = await createTempDataFolder();
    const first = await transcribingCore(dataDir, { engine: { decodeMs: 30 } });
    const meeting = await first.core.createMeeting();
    await first.core.startMeeting({ meetingId: meeting.id });
    await until(first.core, meeting.id, (transcript) => transcript.state === "live");
    first.record(20 * SECOND);
    first.core.close();

    const second = await transcribingCore(dataDir);
    const recovered = (await second.core.listMeetings())[0];
    expect(recovered).toMatchObject({ status: "paused", pausedBy: "interrupted" });
    await second.core.stopMeeting(meeting.id);
    const saved = await until(
      second.core,
      meeting.id,
      (transcript) => transcript.state === "saved",
    );
    const final = readVtt(readFileSync(saved.savedPath as string, "utf8"));
    expect(final.cues.at(-1)?.endMs).toBeGreaterThanOrEqual(18 * SECOND);
  });

  test("audio that waited for a slow engine goes to it when it starts, and live text doesn't fall behind for it", async () => {
    const dataDir = await createTempDataFolder();
    const loaded = Promise.withResolvers<void>();
    const { core, record } = await transcribingCore(dataDir, { loads: loaded.promise });
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id });
    // Fifty seconds while the engine starts: more than live text may lag, less than is kept for it.
    record(50 * SECOND);
    expect((await transcriptOf(core, meeting.id)).state).toBe("preparing");
    loaded.resolve();
    await until(
      core,
      meeting.id,
      (transcript) => transcript.state === "live" && transcript.lines >= 20,
    );
    record(5 * SECOND);
    const live = await until(core, meeting.id, (transcript) => transcript.lines >= 22);
    expect(live.state).toBe("live");
  });

  test("a Meeting in a Mind with no title yet names its Transcript plainly", async () => {
    const dataDir = await createTempDataFolder();
    const { core, record } = await transcribingCore(dataDir);
    const mind = await core.createMind({ title: "" });
    const meeting = await core.createMeeting({ mindId: mind.id });
    await core.startMeeting({ meetingId: meeting.id });
    record(3 * SECOND);
    const live = await until(core, meeting.id, (transcript) => transcript.documentId !== null);
    expect((await documentOf(core, live.documentId as string)).name).toBe("Transcript");
  });

  test("IncarnaMind closing while a Transcript is being finished leaves it to finish at the next start", async () => {
    const dataDir = await createTempDataFolder();
    const first = await transcribingCore(dataDir, { engine: { decodeMs: 30 } });
    const meeting = await first.core.createMeeting();
    await first.core.startMeeting({ meetingId: meeting.id });
    await until(first.core, meeting.id, (transcript) => transcript.state === "live");
    first.record(40 * SECOND);
    await first.core.stopMeeting(meeting.id);
    expect((await transcriptOf(first.core, meeting.id)).state).toBe("finishing");
    first.core.close();
    // Nothing reaches the closed core; the next start finishes what was left.
    const second = await transcribingCore(dataDir);
    const saved = await until(
      second.core,
      meeting.id,
      (transcript) => transcript.state === "saved",
    );
    expect(readVtt(readFileSync(saved.savedPath as string, "utf8")).cues.length).toBeGreaterThan(
      10,
    );
  });

  test("a Chinese interface names the Transcript 转写稿", async () => {
    const dataDir = await createTempDataFolder();
    const { core, record } = await transcribingCore(dataDir, {
      engine: { lines: ["发布日期推迟到十一月十八日。"] },
    });
    await core.updateSettings({ user: { language: "zh-CN" } });
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id });
    record(3 * SECOND);
    const live = await until(core, meeting.id, (transcript) => transcript.documentId !== null);
    expect((await documentOf(core, live.documentId as string)).name).toBe("转写稿 · 会议，10月6日");
  });
});

describe("Who said each line (#941)", { timeout: 60_000 }, () => {
  /** The lines in the Transcript's file now, in the order it keeps them. */
  async function linesOf(core: Core, meetingId: string) {
    const { documentId } = await transcriptOf(core, meetingId);
    if (!documentId) throw new Error("The Transcript has no Document yet.");
    const document = await documentOf(core, documentId);
    return readVtt(readFileSync(document.path, "utf8")).cues;
  }

  test("the microphone's lines read Me and the computer sound's read Them, from the first line, in time order", async () => {
    const { core, record } = await transcribingCore(await createTempDataFolder());
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id, systemAudio: true });
    record(7 * SECOND);
    await until(core, meeting.id, (transcript) => transcript.lines >= 4);

    const lines = await linesOf(core, meeting.id);
    expect(lines.slice(0, 2).map((line) => line.speaker)).toEqual(["Me", "Them"]);
    const said = (speaker: string) => lines.filter((line) => line.speaker === speaker);
    expect(said("Me")[0]?.text).toBe(FAKE_LINES[0]);
    expect(said("Them")[0]?.text).toBe(FAKE_THEIR_LINES[0]);
    expect(lines.every((line) => line.speaker === "Me" || line.speaker === "Them")).toBe(true);
    expect(lines.map((line) => line.startMs)).toEqual(
      lines.map((line) => line.startMs).sort((a, b) => a - b),
    );
  });

  test("both sides speaking at once give two lines, not one merged line", async () => {
    const { core, record } = await transcribingCore(await createTempDataFolder());
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id, systemAudio: true });
    record(6 * SECOND);
    await until(core, meeting.id, (transcript) => transcript.lines >= 4);

    const lines = await linesOf(core, meeting.id);
    const overlapping = lines.filter((mine) =>
      lines.some(
        (theirs) =>
          mine.speaker === "Me" &&
          theirs.speaker === "Them" &&
          mine.startMs < theirs.endMs &&
          theirs.startMs < mine.endMs,
      ),
    );
    expect(overlapping.length).toBeGreaterThanOrEqual(2);
    expect(lines.map((line) => line.startMs)).toEqual(
      lines.map((line) => line.startMs).sort((a, b) => a - b),
    );
  });

  test("a meeting with the microphone only reads Speaker ? and never Them", async () => {
    const { core, record } = await transcribingCore(await createTempDataFolder());
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id, systemAudio: false });
    record(5 * SECOND, ["microphone"]);
    await until(core, meeting.id, (transcript) => transcript.lines >= 2);

    const lines = await linesOf(core, meeting.id);
    expect(lines.map((line) => line.speaker)).toEqual(lines.map(() => "Speaker ?"));
    expect(lines[0]?.text).toBe(FAKE_LINES[0]);
  });

  test("the computer's sound that only gives silence, as a macOS that isn't allowing it does, reads Speaker ? until some sound comes", async () => {
    const { core, record } = await transcribingCore(await createTempDataFolder(), {
      capture: { system: "silent" },
    });
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id, systemAudio: true });
    record(5 * SECOND);
    await until(core, meeting.id, (transcript) => transcript.lines >= 2);

    const lines = await linesOf(core, meeting.id);
    expect(lines.map((line) => line.speaker)).toEqual(lines.map(() => "Speaker ?"));
  });

  test("the computer's sound turned down by the system gives Speaker ? too: nobody else is heard apart", async () => {
    const { core, record } = await transcribingCore(await createTempDataFolder(), {
      capture: { system: "denied" },
    });
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id, systemAudio: true });
    record(5 * SECOND);
    await until(core, meeting.id, (transcript) => transcript.lines >= 2);

    const lines = await linesOf(core, meeting.id);
    expect(lines.map((line) => line.speaker)).toEqual(lines.map(() => "Speaker ?"));
  });

  test("speech being transcribed shows greyed with whose it is", async () => {
    const { core, record } = await transcribingCore(await createTempDataFolder(), {
      engine: { maxSpeechMs: 20_000 },
    });
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id, systemAudio: true });
    record(5 * SECOND);

    const live = await until(core, meeting.id, (transcript) => transcript.partial !== null);
    const { partial } = live;
    if (!partial) throw new Error("Nothing was being transcribed.");
    expect(partial.speaker === "Me" ? FAKE_LINES : FAKE_THEIR_LINES).toContain(partial.text);
    expect(["Me", "Them"]).toContain(partial.speaker);
  });

  test("a line of the microphone that repeats the computer sound's in the same second is dropped, and the same words said alone are kept", async () => {
    const same = ["The launch moves to the eighteenth of November."];
    const { core, record } = await transcribingCore(await createTempDataFolder(), {
      engine: { lines: same, theirLines: same },
    });
    const meeting = await core.createMeeting();
    await core.startMeeting({ meetingId: meeting.id, systemAudio: true });
    // Both say it together: the microphone's copy is the echo of the other side's.
    record(4 * SECOND);
    await until(core, meeting.id, (transcript) => transcript.lines >= 1);
    // Then the User says it alone: that is theirs to keep.
    record(2 * SECOND, ["system"]);
    record(4 * SECOND, ["microphone"]);
    await until(
      core,
      meeting.id,
      (transcript) => transcript.state === "live" && transcript.lines >= 2,
    );
    await vi.waitFor(async () =>
      expect((await linesOf(core, meeting.id)).some((line) => line.speaker === "Me")).toBe(true),
    );

    const lines = await linesOf(core, meeting.id);
    const together = lines.filter((line) => line.startMs < 4 * SECOND);
    expect(together.map((line) => line.speaker)).toEqual(together.map(() => "Them"));
    expect(together.length).toBeGreaterThanOrEqual(1);
    // A stretch is decoded from a quarter of a second before the detector heard it.
    const alone = lines.filter((line) => line.speaker === "Me");
    expect(alone.length).toBeGreaterThanOrEqual(1);
    expect(alone.every((line) => line.startMs >= 5 * SECOND)).toBe(true);
  });
});
