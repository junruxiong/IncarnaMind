/**
 * A speech-to-text engine with no model, for the tests and the smoke tests
 * (#136): sound louder than a whisper is speech, cut into stretches of at
 * most two seconds, and each stretch says the next of its lines in turn.
 * Deterministic, and never a real model.
 */
import { MEETING_SAMPLE_RATE } from "../adapters";
import type { SpeechEngine, VoiceDetector } from "./speech";

export const FAKE_LINES: readonly string[] = [
  "Let's start with the launch date.",
  "The launch moves to the eighteenth of November.",
  "The payment provider's review won't be done before then.",
  "We keep the partner preview on the twenty-eighth of October.",
  "Marketing waits for the new date before announcing anything.",
  "I'll confirm the date with the payment provider by Friday.",
];

export interface FakeSpeechOptions {
  /** What each stretch says, in turn, from the first again after the last. */
  lines?: readonly string[];
  /** How long decoding takes, in ms. */
  decodeMs?: number;
  /** The longest stretch, in ms. */
  maxSpeechMs?: number;
}

/** 20 ms frames: one is speech when its level is above a whisper's. */
const FRAME = MEETING_SAMPLE_RATE / 50;
const LEVEL = 0.01;
/** Speech ends after this much quiet. */
const QUIET_FRAMES = 25;

export function createFakeSpeechEngine(options: FakeSpeechOptions = {}): SpeechEngine {
  const lines = options.lines ?? FAKE_LINES;
  const maxFrames = Math.round((options.maxSpeechMs ?? 2000) / 20);
  let said = 0;

  const detector = (): VoiceDetector => {
    let position = 0;
    let waiting: number[] = [];
    let speech: { start: number; samples: number[]; quiet: number } | null = null;
    const ended: { start: number; samples: Float32Array }[] = [];
    const end = () => {
      if (!speech) return;
      const kept = speech.samples.slice(0, speech.samples.length - speech.quiet * FRAME);
      if (kept.length > 0) ended.push({ start: speech.start, samples: Float32Array.from(kept) });
      speech = null;
    };
    const frame = (samples: number[]) => {
      let sum = 0;
      for (const sample of samples) sum += sample * sample;
      const loud = Math.sqrt(sum / samples.length) > LEVEL;
      if (loud && !speech) speech = { start: position, samples: [], quiet: 0 };
      if (speech) {
        speech.samples.push(...samples);
        speech.quiet = loud ? 0 : speech.quiet + 1;
        if (speech.quiet >= QUIET_FRAMES || speech.samples.length >= maxFrames * FRAME) end();
      }
      position += samples.length;
    };
    return {
      accept(samples) {
        for (const sample of samples) {
          waiting.push(sample);
          if (waiting.length === FRAME) {
            frame(waiting);
            waiting = [];
          }
        }
      },
      speaking: () => speech !== null,
      next: () => ended.shift() ?? null,
      flush() {
        if (waiting.length > 0) frame(waiting);
        waiting = [];
        end();
      },
    };
  };

  return {
    detector,
    recognizer: {
      async decode() {
        if (options.decodeMs) await new Promise((resolve) => setTimeout(resolve, options.decodeMs));
        const line = lines[said % lines.length] ?? "";
        said++;
        return line;
      },
    },
    close() {},
  };
}
