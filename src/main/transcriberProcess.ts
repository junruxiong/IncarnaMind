/**
 * The transcription utility process (#136): runs the speech-to-text engine
 * and Silero VAD off the main process (see ./transcriber), serving the
 * protocol in src/core/transcription/channel.ts.
 *
 * Started with --fake by the smoke tests' INCARNAMIND_TEST_TRANSCRIBER=fake
 * flag: it then runs the fake engine, with no model, as
 * --fake-options=<JSON> asks (slow, failing to read stretches, failing over).
 */
import type { TranscriberPort, TranscriberRequest } from "../core/transcription/channel";
import { serveTranscriber } from "../core/transcription/channel";
import {
  createFakeSpeechEngine,
  type FakeTranscriberOptions,
  failingOver,
} from "../core/transcription/fake";
import { createSherpaEngine } from "../core/transcription/sherpa";

const fake = process.argv.includes("--fake");
const FAKE_OPTIONS = "--fake-options=";
const fakeOptions: FakeTranscriberOptions = JSON.parse(
  process.argv.find((each) => each.startsWith(FAKE_OPTIONS))?.slice(FAKE_OPTIONS.length) ?? "{}",
);
const parent = process.parentPort;

const port: TranscriberPort = {
  onRequest: (listener) =>
    parent.on("message", ({ data }: { data: TranscriberRequest }) => listener(data)),
  respond: (response) => parent.postMessage(response),
};

serveTranscriber(
  fake ? async () => createFakeSpeechEngine(fakeOptions) : createSherpaEngine,
  fake && fakeOptions.failover ? failingOver(port, fakeOptions.failover) : port,
);
