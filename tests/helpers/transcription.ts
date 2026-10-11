import type { Transcriber, TranscriptionModelFiles } from "../../src/core";
import {
  createChannelTranscriber,
  serveTranscriber,
  type TranscriberRequest,
  type TranscriberResponse,
} from "../../src/core/transcription/channel";
import { createFakeSpeechEngine, type FakeSpeechOptions } from "../../src/core/transcription/fake";
import type { SpeechEngine } from "../../src/core/transcription/speech";

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
 * real one in the live tests.
 */
export function inProcessTranscriber(
  options: FakeSpeechOptions = {},
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
    serveTranscriber(engine, {
      onRequest: (listener) => {
        deliver = listener;
      },
      respond: (response) =>
        setImmediate(() => {
          if (open) for (const listener of responses) listener(response);
        }),
    });
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
