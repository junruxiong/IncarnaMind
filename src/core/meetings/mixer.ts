/**
 * The microphone and the computer's sound mixed into one track for
 * transcription (#136), in the Meeting's time: what the User says and what
 * the others say are transcribed once, and the microphone picking up the
 * speakers doesn't give each line twice. Each source's samples are added
 * where they belong; audio is let out a moment behind the newest, so a
 * source a little late still joins in. Samples later than that go to disk
 * only. A stretch neither source covers is silence.
 */
import { MEETING_SAMPLE_RATE } from "../adapters";

/** How long audio waits for the other source before it is let out. */
const DELAY_MS = 300;

const sampleAt = (ms: number) => Math.round((ms * MEETING_SAMPLE_RATE) / 1000);
const msAt = (sample: number) => (sample * 1000) / MEETING_SAMPLE_RATE;

export interface MixedAudio {
  /** Where it starts, in the Meeting's time. */
  atMs: number;
  samples: Int16Array;
}

export function createMixer(startMs: number) {
  /** The first sample not yet let out, in the Meeting's time. */
  let out = sampleAt(startMs);
  /** Sums from `out` on. */
  let sums = new Int32Array(MEETING_SAMPLE_RATE);
  /** One past the newest sample heard. */
  let newest = out;

  function room(end: number): void {
    const needed = end - out;
    if (needed <= sums.length) return;
    const grown = new Int32Array(Math.max(needed, sums.length * 2));
    grown.set(sums);
    sums = grown;
  }

  function take(until: number): MixedAudio | null {
    if (until <= out) return null;
    const count = until - out;
    const samples = new Int16Array(count);
    for (let i = 0; i < count; i++) {
      samples[i] = Math.max(-32768, Math.min(32767, sums[i] as number));
    }
    sums.copyWithin(0, count);
    sums.fill(0, sums.length - count);
    const atMs = msAt(out);
    out = until;
    return { atMs, samples };
  }

  /** Adds a source's samples, which start at `atMs` in the Meeting's time. */
  function put(samples: Int16Array, atMs: number): void {
    let start = sampleAt(atMs);
    const end = start + samples.length;
    if (end > out) {
      const skip = Math.max(0, out - start);
      start += skip;
      room(end);
      for (let i = skip; i < samples.length; i++) {
        const at = start - out + (i - skip);
        sums[at] = (sums[at] as number) + (samples[i] as number);
      }
    }
    newest = Math.max(newest, end);
  }

  return {
    put,

    /** Adds a source's samples, as `put`, and returns what is ready to go. */
    add(samples: Int16Array, atMs: number): MixedAudio | null {
      put(samples, atMs);
      return take(newest - sampleAt(DELAY_MS));
    },

    /** Lets out everything heard. */
    flush(): MixedAudio | null {
      return take(newest);
    },

    /** Where the audio not yet let out starts, in the Meeting's time. */
    get atMs(): number {
      return msAt(out);
    },
  };
}

export type Mixer = ReturnType<typeof createMixer>;
