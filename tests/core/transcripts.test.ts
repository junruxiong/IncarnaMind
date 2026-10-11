import { existsSync, readFileSync, realpathSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import type { Core, Meeting, MeetingTranscript } from "../../src/core";
import { readVtt } from "../../src/core/documents/formats/vtt";
import { createFakeCapture } from "../../src/core/meetings/fakeCapture";
import { FAKE_LINES, type FakeSpeechOptions } from "../../src/core/transcription/fake";
import { askAndFinish, citingModel, onlyCitation, type ShownPassage } from "../helpers/citations";
import { createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { scriptedModels } from "../helpers/models";
import { inProcessTranscriber, NO_TRANSCRIPTION_FILES } from "../helpers/transcription";

const SECOND = 1000;

/**
 * A core that records Meetings with the fake capture (a tone from each
 * source) and transcribes them with the fake engine behind the channel the
 * desktop app uses: every two seconds of sound is a line, saying the next of
 * `FAKE_LINES`. The clock moves with the audio the test records.
 */
async function transcribingCore(
  dataDir: string,
  {
    engine = {},
    model,
  }: { engine?: FakeSpeechOptions; model?: Parameters<typeof scriptedModels>[0] } = {},
) {
  const clock = manualClock();
  const capture = createFakeCapture();
  const engineHost = inProcessTranscriber(engine);
  const models = model ? scriptedModels(model) : undefined;
  const core = startCore(dataDir, {
    audioCapture: capture,
    transcriber: engineHost.transcriber,
    transcriptionModelSource: NO_TRANSCRIPTION_FILES,
    now: clock.now,
    ...(models && { createChatModel: models.createChatModel }),
  });
  await core.updateSettings({ device: { meetingsEnabled: true, meetingNoticeAccepted: true } });
  await core.setOutputLocation(join(dataDir, "Documents", "IncarnaMind"));
  const record = (ms: number, sources: ("microphone" | "system")[] = ["microphone", "system"]) => {
    for (let at = 0; at < ms; at += 500) {
      const step = Math.min(500, ms - at);
      clock.advance(step);
      for (const source of sources) capture.push(source, step);
    }
  };
  return { core, capture, clock, record, engineHost };
}

async function documentOf(core: Core, documentId: string) {
  const document = (await core.listDocuments()).find((each) => each.id === documentId);
  if (!document) throw new Error(`No Document ${documentId}.`);
  return document;
}

async function transcriptOf(core: Core, meetingId: string): Promise<MeetingTranscript> {
  const meeting = (await core.listMeetings()).find((each) => each.id === meetingId);
  if (!meeting?.transcript) throw new Error(`Meeting ${meetingId} has no transcript.`);
  return meeting.transcript;
}

/** Waits, with real time, until the Meeting's Transcript is as `ready` wants it. */
async function until(
  core: Core,
  meetingId: string,
  ready: (transcript: MeetingTranscript, meeting: Meeting) => boolean,
): Promise<MeetingTranscript> {
  return vi.waitFor(
    async () => {
      const meeting = (await core.listMeetings()).find((each) => each.id === meetingId);
      if (!meeting?.transcript || !ready(meeting.transcript, meeting)) {
        throw new Error(`Not yet: ${JSON.stringify(meeting?.transcript)}`);
      }
      return meeting.transcript;
    },
    { timeout: 20_000, interval: 20 },
  );
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
    expect(vtt.cues.slice(0, 2)).toEqual([
      { startMs: 0, endMs: 2000, text: FAKE_LINES[0], speaker: null },
      // A stretch is decoded from a quarter of a second before the detector heard it.
      { startMs: 1750, endMs: 4000, text: FAKE_LINES[1], speaker: null },
    ]);

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
      location: { kind: "time", from: 1750, to: 4000 },
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
