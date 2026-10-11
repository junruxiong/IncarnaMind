import { describe, expect, test, vi } from "vitest";
import type { SpeechSegment } from "../../src/core";
import { createMixer } from "../../src/core/meetings/mixer";
import {
  chooseModel,
  QWEN3_ASR_0_6B,
  type TranscriptionModelDefinition,
} from "../../src/core/transcription/model";
import {
  createSpeechHost,
  PARTIAL_EVERY_MS,
  type SpeechEngine,
  type VoiceDetector,
} from "../../src/core/transcription/speech";

const RATE = 16_000;

/** A detector that hears speech exactly where the test says, in seconds. */
function scriptedDetector(speech: readonly [number, number][]): VoiceDetector {
  let position = 0;
  let kept: number[] = [];
  const ended: { start: number; samples: Float32Array }[] = [];
  const inSpeech = (at: number) => speech.find(([from, to]) => at >= from * RATE && at < to * RATE);
  let open: [number, number] | undefined;
  return {
    accept(samples) {
      for (const sample of samples) {
        const now = inSpeech(position);
        if (open && now !== open) {
          ended.push({ start: open[0] * RATE, samples: Float32Array.from(kept) });
          kept = [];
        }
        open = now;
        if (now) kept.push(sample);
        position++;
      }
    },
    speaking: () => open !== undefined,
    next: () => ended.shift() ?? null,
    flush() {
      if (open) ended.push({ start: open[0] * RATE, samples: Float32Array.from(kept) });
      open = undefined;
      kept = [];
    },
  };
}

/** A recognizer that says how long the speech it was given is: "4.0 s". */
const lengthEngine = (speech: readonly [number, number][]): SpeechEngine => ({
  detector: () => scriptedDetector(speech),
  recognizer: { decode: async (samples) => `${(samples.length / RATE).toFixed(1)} s` },
  close() {},
});

describe("Live text from VAD segments, decoded again as speech goes on", () => {
  test("speech in progress is decoded again every few seconds; its final text comes when it ends, as the same line", async () => {
    const segments: SpeechSegment[] = [];
    const progress: number[] = [];
    const host = createSpeechHost(lengthEngine([[1, 9]]), {
      segment: (segment) => segments.push(segment),
      progress: (_stream, doneMs) => progress.push(doneMs),
      failed: () => {},
    });
    // Ten seconds in 100 ms frames, as the capture sends them.
    for (let at = 0; at < 100; at++) {
      host.push("live", new Float32Array(RATE / 10));
      await Promise.resolve();
      await new Promise((resolve) => setImmediate(resolve));
    }
    await host.finish("live");
    const partials = segments.filter((segment) => !segment.final);
    const finals = segments.filter((segment) => segment.final);
    expect(partials.length).toBeGreaterThanOrEqual(2);
    expect(partials.every((segment) => segment.id === 1)).toBe(true);
    // Each partial holds the speech so far: longer each time.
    const lengths = partials.map((segment) => segment.endMs - segment.startMs);
    expect(lengths).toEqual([...lengths].sort((a, b) => a - b));
    expect(lengths[1] as number).toBeGreaterThanOrEqual(
      (lengths[0] as number) + PARTIAL_EVERY_MS - 200,
    );
    expect(finals).toEqual([
      expect.objectContaining({
        stream: "live",
        id: 1,
        // From a quarter of a second before the detector heard it, as a word's first sound is often cut.
        startMs: 750,
        endMs: 9000,
        text: "8.3 s",
        final: true,
      }),
    ]);
    // Done up to where the speech started while it went on; all of it once its final text came.
    expect(progress).toContain(10_000);
    expect(Math.max(...progress.slice(0, 50))).toBeLessThanOrEqual(1000);
  });

  test("a stream dropped forgets what it hadn't decoded; another goes on", async () => {
    const segments: SpeechSegment[] = [];
    const host = createSpeechHost(
      lengthEngine([
        [0, 1],
        [2, 3],
      ]),
      {
        segment: (segment) => segments.push(segment),
        progress: () => {},
        failed: () => {},
      },
    );
    host.push("a", new Float32Array(4 * RATE));
    host.drop("a");
    host.push("b", new Float32Array(4 * RATE));
    await host.finish("b");
    await vi.waitFor(() => expect(segments.map((each) => each.stream)).toEqual(["b", "b"]));
  });

  test("a stretch the engine can't read loses its text, and the stream goes on", async () => {
    const failures: string[] = [];
    const segments: SpeechSegment[] = [];
    let calls = 0;
    const host = createSpeechHost(
      {
        ...lengthEngine([
          [0, 1],
          [2, 3],
        ]),
        recognizer: {
          decode: async () => {
            calls++;
            if (calls === 1) throw new Error("bad audio");
            return "fine";
          },
        },
      },
      {
        segment: (segment) => segments.push(segment),
        progress: () => {},
        failed: (_s, m) => failures.push(m),
      },
    );
    host.push("live", new Float32Array(4 * RATE));
    await host.finish("live");
    expect(failures).toEqual(["bad audio"]);
    expect(segments.map((each) => each.text)).toEqual(["", "fine"]);
  });
});

describe("Mixing the microphone and the computer's sound", () => {
  test("both sources are summed where they belong in the Meeting's time, clipped, and let out a moment behind", () => {
    const mixer = createMixer(1000);
    const tone = (value: number, length: number) => new Int16Array(length).fill(value);
    // The computer's sound, 100 ms: held back a moment for the microphone.
    expect(mixer.add(tone(30_000, 1600), 1000)).toBeNull();
    // The microphone, from 50 ms in: what is 300 ms older than the newest goes out.
    const out = mixer.add(tone(5000, 8000), 1050);
    expect(out?.atMs).toBe(1000);
    expect(out?.samples.length).toBe(4000);
    expect([out?.samples[0], out?.samples[800], out?.samples[1700]]).toEqual([
      30_000, 32_767, 5000,
    ]);
    const rest = mixer.flush();
    expect(rest?.atMs).toBe(1250);
    expect(rest?.samples.length).toBe(4800);
    // Samples later than what has gone out go to disk only.
    expect(mixer.add(tone(5, 160), 1000)).toBeNull();
    expect(mixer.flush()).toBeNull();
  });
});

describe("Choosing the speech model", () => {
  const smaller: TranscriptionModelDefinition = {
    ...QWEN3_ASR_0_6B,
    id: "smaller",
    memoryBytes: 500_000_000,
  };
  const models = [QWEN3_ASR_0_6B, smaller];
  const memory = { freeBytes: 8e9, totalBytes: 32e9 };

  test("the best one with the memory to run, unless it didn't keep up here last time", () => {
    expect(chooseModel(models, memory, {}).id).toBe(QWEN3_ASR_0_6B.id);
    expect(chooseModel(models, { freeBytes: 0.3e9, totalBytes: 4e9 }, {}).id).toBe("smaller");
    expect(chooseModel(models, memory, { [QWEN3_ASR_0_6B.id]: 1.4 }).id).toBe("smaller");
    expect(chooseModel(models, memory, { [QWEN3_ASR_0_6B.id]: 0.5 }).id).toBe(QWEN3_ASR_0_6B.id);
    // Too little memory for any: the smallest.
    expect(chooseModel(models, { freeBytes: 0.1e9, totalBytes: 1e9 }, {}).id).toBe("smaller");
  });
});
