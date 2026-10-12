import { join } from "node:path";
import { vi } from "vitest";
import type {
  Core,
  Logger,
  Meeting,
  MeetingTranscript,
  Transcriber,
  TranscriptionModelFiles,
} from "../../src/core";
import { createFakeCapture } from "../../src/core/meetings/fakeCapture";
import {
  createChannelTranscriber,
  serveTranscriber,
  type TranscriberPort,
  type TranscriberRequest,
  type TranscriberResponse,
} from "../../src/core/transcription/channel";
import {
  createFakeSpeechEngine,
  type FakeTranscriberOptions,
  failingOver,
} from "../../src/core/transcription/fake";
import type { SpeechEngine } from "../../src/core/transcription/speech";
import { manualClock, startCore } from "./core";
import { scriptedModels } from "./models";

/** No files to download: the fake engine needs none. */
export const NO_TRANSCRIPTION_FILES = {
  model: { baseUrl: "http://127.0.0.1/", files: [] },
  vad: { baseUrl: "http://127.0.0.1/", files: [] },
} as const;

/**
 * A `Transcriber` whose engine (the fake one) runs in this process behind
 * the same channel as the desktop app's utility process, messages crossing a
 * turn of the event loop as they would between processes. `crash` kills the
 * running one, as an out-of-memory would. `engine` runs another engine, the
 * real one in the live tests. `options.failover` has it say it moved its
 * streams to the cloud and back.
 */
export function inProcessTranscriber(
  options: FakeTranscriberOptions = {},
  engine: (model: TranscriptionModelFiles) => Promise<SpeechEngine> = async () =>
    createFakeSpeechEngine(options),
) {
  let kill: (() => void) | null = null;
  let started = 0;
  const transcriber: Transcriber = createChannelTranscriber(() => {
    started++;
    let open = true;
    let deliver: (request: TranscriberRequest) => void = () => {};
    const responses: ((response: TranscriberResponse) => void)[] = [];
    const exits: ((code: number | null) => void)[] = [];
    const port: TranscriberPort = {
      onRequest: (listener) => {
        deliver = listener;
      },
      respond: (response) =>
        setImmediate(() => {
          if (open) for (const listener of responses) listener(response);
        }),
    };
    serveTranscriber(engine, options.failover ? failingOver(port, options.failover) : port);
    kill = () => {
      open = false;
      for (const listener of exits) listener(null);
    };
    return {
      send: (request) =>
        setImmediate(() => {
          if (open) deliver(request);
        }),
      onResponse: (listener) => responses.push(listener),
      onExit: (listener) => exits.push(listener),
      stop: () => {
        open = false;
      },
    };
  });
  return {
    transcriber,
    crash: () => kill?.(),
    get started() {
      return started;
    },
  };
}

/**
 * A core that records Meetings with the fake capture (a tone from each
 * source) and transcribes them with the fake engine behind the channel the
 * desktop app uses: every two seconds of sound is a line, saying the next of
 * `FAKE_LINES`. The clock moves with the audio the test records.
 */
export async function transcribingCore(
  dataDir: string,
  {
    engine = {},
    model,
    loads,
    log,
  }: {
    engine?: FakeTranscriberOptions;
    model?: Parameters<typeof scriptedModels>[0];
    /** The engine starts once this resolves, as a model loading slowly does. */
    loads?: Promise<void>;
    log?: Logger;
  } = {},
) {
  const clock = manualClock();
  const capture = createFakeCapture();
  const engineHost = inProcessTranscriber(engine, async () => {
    await loads;
    return createFakeSpeechEngine(engine);
  });
  const models = model ? scriptedModels(model) : undefined;
  const core = startCore(dataDir, {
    audioCapture: capture,
    transcriber: engineHost.transcriber,
    transcriptionModelSource: NO_TRANSCRIPTION_FILES,
    now: clock.now,
    ...(models && { createChatModel: models.createChatModel }),
    ...(log && { log }),
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

export async function transcriptOf(core: Core, meetingId: string): Promise<MeetingTranscript> {
  const meeting = (await core.listMeetings()).find((each) => each.id === meetingId);
  if (!meeting?.transcript) throw new Error(`Meeting ${meetingId} has no transcript.`);
  return meeting.transcript;
}

/** Waits, with real time, until the Meeting's Transcript is as `ready` wants it. */
export async function until(
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
