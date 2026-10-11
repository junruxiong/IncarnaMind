/**
 * The first speech-to-text engine (#136): Qwen3-ASR-0.6B, int8, with Silero
 * VAD, both through sherpa-onnx-node (Apache-2.0), on the CPU. Decoding runs
 * on sherpa's own threads (`decodeAsync`), so the process keeps taking audio
 * while a stretch is decoded. The desktop app runs it in a utility process
 * (src/main/transcriberProcess.ts), never on the core's thread.
 */
import type { TranscriptionModelFiles } from "../adapters";
import { MEETING_SAMPLE_RATE } from "../adapters";
import type { SpeechEngine, VoiceDetector } from "./speech";

/** Silero's window at 16 kHz: it reads audio 512 samples at a time. */
const WINDOW = 512;

/** The engine id the model definitions use (./model). */
export const QWEN3_ASR = "qwen3-asr";

export async function createSherpaEngine(model: TranscriptionModelFiles): Promise<SpeechEngine> {
  if (model.engine !== QWEN3_ASR)
    throw new Error(`Unknown speech-to-text engine "${model.engine}".`);
  // Loaded only when needed: the native runtime is large.
  const sherpa = await import("sherpa-onnx-node");
  const { files } = model;
  const required = (name: string) => {
    const path = files[name];
    if (!path) throw new Error(`The model has no ${name} file.`);
    return path;
  };
  const recognizer = await sherpa.OfflineRecognizer.createAsync({
    featConfig: { sampleRate: MEETING_SAMPLE_RATE, featureDim: 80 },
    modelConfig: {
      qwen3Asr: {
        convFrontend: required("convFrontend"),
        encoder: required("encoder"),
        decoder: required("decoder"),
        tokenizer: required("tokenizer"),
        hotwords: "",
      },
      tokens: "",
      numThreads: model.threads,
      provider: "cpu",
      debug: 0,
    },
  });

  const detector = (): VoiceDetector => {
    const vad = new sherpa.Vad(
      {
        sileroVad: {
          model: model.vad,
          threshold: 0.5,
          minSpeechDuration: 0.25,
          minSilenceDuration: 0.5,
          // Long speech is cut, so its final text never waits long.
          maxSpeechDuration: 20,
          windowSize: WINDOW,
        },
        sampleRate: MEETING_SAMPLE_RATE,
        numThreads: 1,
        debug: 0,
      },
      60,
    );
    let waiting = new Float32Array(0);
    return {
      accept(samples) {
        const all = new Float32Array(waiting.length + samples.length);
        all.set(waiting);
        all.set(samples, waiting.length);
        let at = 0;
        for (; at + WINDOW <= all.length; at += WINDOW)
          vad.acceptWaveform(all.subarray(at, at + WINDOW));
        waiting = all.slice(at);
      },
      speaking: () => vad.isDetected(),
      next() {
        if (vad.isEmpty()) return null;
        const segment = vad.front(false);
        vad.pop();
        return { start: segment.start, samples: segment.samples };
      },
      flush() {
        if (waiting.length > 0) {
          const padded = new Float32Array(WINDOW);
          padded.set(waiting);
          vad.acceptWaveform(padded);
          waiting = new Float32Array(0);
        }
        vad.flush();
      },
    };
  };

  return {
    detector,
    recognizer: {
      async decode(samples) {
        const stream = recognizer.createStream();
        stream.acceptWaveform({ samples, sampleRate: MEETING_SAMPLE_RATE });
        return (await recognizer.decodeAsync(stream)).text;
      },
    },
    // The recognizer and detectors are freed with the process; nothing holds them here.
    close() {},
  };
}
