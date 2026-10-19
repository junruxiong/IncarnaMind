import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { AudioCapture, AudioSink } from "../../src/core";
import { readVtt } from "../../src/core/documents/formats/vtt";
import { listParts, meetingFolder } from "../../src/core/meetings/audioFiles";
import { createFakeSpeechEngine } from "../../src/core/transcription/fake";
import { createTempDataFolder, manualClock, nextTurn, startCore } from "../helpers/core";
import { inProcessTranscriber, NO_TRANSCRIPTION_FILES, until } from "../helpers/transcription";

const RATE = 16_000;
const TICK_MS = 100;
const FRAME = (RATE * TICK_MS) / 1000;
const SECONDS = 16;
const FIXTURES = join(__dirname, "..", "fixtures", "echo");

/** A recorded voice from the fixtures: our own 16-bit mono WAV files, with a 44-byte header. */
const clip = (name: string): Int16Array => {
  const bytes = readFileSync(join(FIXTURES, `${name}.wav`));
  return new Int16Array(
    bytes.buffer.slice(bytes.byteOffset + 44, bytes.byteOffset + bytes.byteLength),
  );
};

/** The call's sound for `SECONDS`, with each clip placed where it starts, in seconds. */
function timeline(...voices: [Int16Array, number][]): Int16Array {
  const out = new Int16Array(SECONDS * RATE);
  for (const [samples, at] of voices) out.set(samples, at * RATE);
  return out;
}

/** The other side speaks three times; the User answers, talks to someone in the room, and answers again. */
const SYSTEM = timeline([clip("en-far"), 0], [clip("en-far"), 6], [clip("en-far"), 12]);
const MICROPHONE = timeline([clip("en-near"), 1], [clip("zh-near"), 7], [clip("en-near"), 12]);
const PAUSED_AT = 5500;
const RESUMED_AT = 11_500;

/**
 * A Meeting played from the fixtures into the real capture pipeline, with a transcriber that
 * remembers what it was sent. `pause` pauses the microphone from `PAUSED_AT` to `RESUMED_AT`.
 * A capture that `leaks` goes on sending microphone frames, as ones already on their way do.
 * The engine starts `at` the start of the call, as a model that loads slowly does at 14 s, or
 * only after Stop, when what was left is transcribed from the audio on disk. With `route`, the
 * streams go to the pretend cloud service the User chose, with their key.
 */
async function playCall({
  pause,
  leaks = false,
  at: engineStarts = "start",
  route,
}: {
  pause: boolean;
  leaks?: boolean;
  at?: "start" | "14 s" | "stop";
  route?: "soniox";
}) {
  const dataDir = await createTempDataFolder();
  const clock = manualClock();
  const captured: { sink: AudioSink | null } = { sink: null };
  const microphone = { on: true, switches: [] as boolean[] };
  const capture: AudioCapture = {
    systemAudio: true,
    start(_request, given) {
      captured.sink = given;
      given.state("microphone", "recording");
      given.state("system", "recording");
      return {
        setMicrophone(on) {
          microphone.on = on;
          microphone.switches.push(on);
          if (on) given.state("microphone", "recording");
        },
        stop: async () => {
          captured.sink = null;
        },
      };
    },
    openSettings: async () => {},
  };
  let ready = () => {};
  const loads = new Promise<void>((resolve) => {
    ready = resolve;
  });
  if (engineStarts === "start") ready();
  const engine = inProcessTranscriber({}, async () => {
    await loads;
    return createFakeSpeechEngine();
  });
  const pushed: Record<string, Int16Array[]> = {};
  const opened: string[] = [];
  const core = startCore(dataDir, {
    audioCapture: capture,
    transcriber: {
      ...engine.transcriber,
      push(stream, samples) {
        pushed[stream] = [...(pushed[stream] ?? []), samples.slice()];
        engine.transcriber.push(stream, samples);
      },
      open(stream, cloud) {
        opened.push(stream);
        engine.transcriber.open(stream, cloud);
      },
    },
    transcriptionModelSource: NO_TRANSCRIPTION_FILES,
    transcriptionKeyFetch: async () => new Response(null, { status: 200 }),
    now: clock.now,
  });
  await core.updateSettings({ device: { meetingNoticeAccepted: true } });
  await core.setOutputLocation(join(dataDir, "Documents", "IncarnaMind"));
  if (route) {
    await core.setTranscriptionKey({ route, apiKey: "soniox-test-key-0123456789" });
    await core.updateSettings({ device: { transcriptionRoute: route } });
  }
  const { id } = await core.createMeeting();
  await core.startMeeting({ meetingId: id, systemAudio: true });
  const live = () => until(core, id, (transcript) => transcript.state === "live");
  if (route) await until(core, id, (transcript) => transcript.cloud === route);
  else if (engineStarts === "start") await live();

  let heardByCapture = 0;
  for (let tick = 0; tick * TICK_MS < SECONDS * 1000; tick++) {
    const at = tick * TICK_MS;
    if (engineStarts === "14 s" && at === 14_000) {
      ready();
      await live();
    }
    if (pause && at === PAUSED_AT) await core.setMeetingMicrophone({ meetingId: id, paused: true });
    if (pause && at === RESUMED_AT) {
      await core.setMeetingMicrophone({ meetingId: id, paused: false });
    }
    clock.advance(TICK_MS);
    const from = tick * FRAME;
    if (microphone.on || leaks) {
      captured.sink?.audio("microphone", MICROPHONE.slice(from, from + FRAME));
      if (microphone.on) heardByCapture += FRAME;
    }
    captured.sink?.audio("system", SYSTEM.slice(from, from + FRAME));
    await nextTurn();
  }
  await core.stopMeeting(id);
  ready();
  const saved = await until(core, id, (transcript) => transcript.state === "saved");
  if (!saved.savedPath) throw new Error("The Transcript wasn't saved.");
  const transcript = readVtt(readFileSync(saved.savedPath, "utf8"));
  const parts = listParts(meetingFolder(dataDir, id));
  return { core, id, pushed, opened, microphone, heardByCapture, transcript, parts };
}

type Played = Awaited<ReturnType<typeof playCall>>;

const streamsOf = (played: Played, source: "microphone" | "system") =>
  Object.entries(played.pushed)
    .filter(([stream]) => new RegExp(`:${source}(:\\d+)?$`).test(stream))
    .map(([, frames]) => frames);

const samplesIn = (frames: readonly Int16Array[]) =>
  frames.reduce((sum, each) => sum + each.length, 0);

/** What a source's streams were sent, in order, as one number. */
const hashOf = (frames: readonly Int16Array[]) => {
  const hash = createHash("sha256");
  for (const each of frames) hash.update(each);
  return hash.digest("hex");
};

const partsHash = (played: Played, source: "microphone" | "system") =>
  played.parts
    .filter((part) => part.source === source)
    .map(
      (part) =>
        `${part.offsetMs}+${part.durationMs}:${hashOf([new Int16Array(readFileSync(part.path).buffer.slice(44))])}`,
    );

/** Whether the engine heard it as it was recorded, or only from the audio on disk after Stop. */
const heardLive = (played: Played) =>
  Object.keys(played.pushed).every((stream) => !stream.includes("catch-up"));

const said = (played: Played, speaker: string) =>
  played.transcript.cues.filter((cue) => cue.speaker === speaker);

const isMicrophone = (stream: string) => /:microphone(:\d+)?$/.test(stream);

describe("Pausing the microphone, in the capture pipeline, with recorded voices", () => {
  test("nothing from the microphone reaches the audio, the transcriber or the Transcript for the stretch, and the other side is recorded and transcribed as before", async () => {
    const baseline = await playCall({ pause: false });
    const paused = await playCall({ pause: true });

    expect(heardLive(paused)).toBe(true);
    // The capture was told to stop the microphone at its source, and to start it again.
    expect(paused.microphone.switches).toEqual([false, true]);

    // The microphone's audio on disk has the stretch missing, and nothing else.
    const micParts = paused.parts.filter((part) => part.source === "microphone");
    expect(micParts.length).toBeGreaterThan(0);
    for (const part of micParts) {
      const overlaps = part.offsetMs < RESUMED_AT && part.offsetMs + part.durationMs > PAUSED_AT;
      expect(overlaps).toBe(false);
    }
    expect(micParts.reduce((sum, part) => sum + part.durationMs, 0)).toBe(
      (paused.heardByCapture / RATE) * 1000,
    );

    // The transcriber was sent the microphone's voice outside the stretch only: no silence for it either.
    const sentMicrophone = streamsOf(paused, "microphone").flat();
    expect(samplesIn(sentMicrophone)).toBeLessThanOrEqual(paused.heardByCapture + FRAME);
    expect(samplesIn(sentMicrophone)).toBeGreaterThanOrEqual(paused.heardByCapture - FRAME);
    expect(streamsOf(paused, "microphone").length).toBe(2);

    // No "Me" line in the stretch; the line after it is the User's second answer.
    const me = said(paused, "Me");
    expect(me.length).toBeGreaterThan(0);
    for (const line of me) {
      expect(line.startMs >= RESUMED_AT || line.endMs <= PAUSED_AT).toBe(true);
    }
    expect(me.some((line) => line.startMs >= RESUMED_AT)).toBe(true);
    expect(said(baseline, "Me").some((line) => line.startMs >= 6000 && line.endMs <= 11_000)).toBe(
      true,
    );
    expect(paused.transcript.gaps).toEqual([
      { fromMs: PAUSED_AT, toMs: RESUMED_AT, reason: "micPaused" },
    ]);

    // The computer's sound: the same audio went to disk and to the transcriber, and the same lines came out.
    expect(partsHash(paused, "system")).toEqual(partsHash(baseline, "system"));
    expect(hashOf(streamsOf(paused, "system").flat())).toBe(
      hashOf(streamsOf(baseline, "system").flat()),
    );
    expect(said(paused, "Them")).toEqual(said(baseline, "Them"));
    expect(said(paused, "Them").length).toBeGreaterThanOrEqual(3);
  });

  test("frames already on their way when the microphone is paused are dropped", async () => {
    const quiet = await playCall({ pause: true });
    const leaky = await playCall({ pause: true, leaks: true });

    expect(partsHash(leaky, "microphone")).toEqual(partsHash(quiet, "microphone"));
    expect(samplesIn(streamsOf(leaky, "microphone").flat())).toBe(
      samplesIn(streamsOf(quiet, "microphone").flat()),
    );
    expect(said(leaky, "Me")).toEqual(said(quiet, "Me"));
    expect(leaky.transcript.gaps).toEqual(quiet.transcript.gaps);
  });

  test("with the engine still starting, what is held for it has no microphone for the stretch either", async () => {
    const early = await playCall({ pause: true });
    const late = await playCall({ pause: true, at: "14 s" });

    expect(heardLive(late)).toBe(true);
    expect(streamsOf(late, "microphone").length).toBe(2);
    expect(samplesIn(streamsOf(late, "microphone").flat())).toBe(
      samplesIn(streamsOf(early, "microphone").flat()),
    );
    for (const line of said(late, "Me")) {
      expect(line.startMs >= RESUMED_AT || line.endMs <= PAUSED_AT).toBe(true);
    }
    expect(said(late, "Me").length).toBeGreaterThan(0);
  });

  test("what is transcribed after Stop, from the audio on disk, has no microphone for the stretch either", async () => {
    const baseline = await playCall({ pause: false, at: "stop" });
    const afterwards = await playCall({ pause: true, at: "stop" });

    expect(Object.keys(afterwards.pushed).every((stream) => stream.includes("catch-up"))).toBe(
      true,
    );
    expect(streamsOf(afterwards, "microphone").length).toBe(2);
    expect(samplesIn(streamsOf(afterwards, "microphone").flat())).toBe(afterwards.heardByCapture);
    for (const line of said(afterwards, "Me")) {
      expect(line.startMs >= RESUMED_AT || line.endMs <= PAUSED_AT).toBe(true);
    }
    expect(said(afterwards, "Me").length).toBeGreaterThan(0);
    expect(afterwards.transcript.gaps).toEqual([
      { fromMs: PAUSED_AT, toMs: RESUMED_AT, reason: "micPaused" },
    ]);
    expect(said(afterwards, "Them")).toEqual(said(baseline, "Them"));
  });
  test("sent to a cloud service, no microphone audio goes for the stretch either: its connection is finished at the pause and opened again at the resume", async () => {
    const paused = await playCall({ pause: true, route: "soniox" });

    expect(Object.keys(paused.pushed).every((stream) => paused.opened.includes(stream))).toBe(true);
    expect(paused.opened.filter(isMicrophone).length).toBe(2);
    expect(paused.opened.filter((stream) => stream.endsWith(":system")).length).toBe(1);
    const sentMicrophone = samplesIn(streamsOf(paused, "microphone").flat());
    expect(sentMicrophone).toBeLessThanOrEqual(paused.heardByCapture + FRAME);
    expect(sentMicrophone).toBeGreaterThanOrEqual(paused.heardByCapture - FRAME);
    expect(paused.microphone.switches).toEqual([false, true]);

    const me = said(paused, "Me");
    expect(me.some((line) => line.endMs <= PAUSED_AT)).toBe(true);
    expect(me.some((line) => line.startMs >= RESUMED_AT)).toBe(true);
    for (const line of me) {
      expect(line.startMs >= RESUMED_AT || line.endMs <= PAUSED_AT).toBe(true);
    }
    expect(paused.transcript.gaps).toEqual([
      { fromMs: PAUSED_AT, toMs: RESUMED_AT, reason: "micPaused" },
    ]);
    expect(paused.transcript.cues.filter((cue) => cue.speaker?.startsWith("Them")).length).toBe(
      (await playCall({ pause: false, route: "soniox" })).transcript.cues.filter((cue) =>
        cue.speaker?.startsWith("Them"),
      ).length,
    );
  });
});
