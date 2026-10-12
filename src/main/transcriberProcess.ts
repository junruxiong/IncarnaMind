/**
 * The transcription utility process (#136): runs the speech-to-text engine
 * and Silero VAD off the main process (see ./transcriber), serving the
 * protocol in src/core/transcription/channel.ts.
 *
 * Started with --fake by the smoke tests' INCARNAMIND_TEST_TRANSCRIBER=fake
 * flag: it then runs the fake engine, with no model.
 */
import type { TranscriberRequest } from "../core/transcription/channel";
import { serveTranscriber } from "../core/transcription/channel";
import { createFakeSpeechEngine } from "../core/transcription/fake";
import { createSherpaEngine } from "../core/transcription/sherpa";

const fake = process.argv.includes("--fake");
const port = process.parentPort;

serveTranscriber(fake ? async () => createFakeSpeechEngine() : createSherpaEngine, {
  onRequest: (listener) =>
    port.on("message", ({ data }: { data: TranscriberRequest }) => listener(data)),
  respond: (response) => port.postMessage(response),
});
