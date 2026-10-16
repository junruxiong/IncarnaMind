/**
 * The microphone and the computer's sound mixed into one track for
 * transcription (#136), in the Meeting's time. Before the microphone joins
 * the mix, the echo of the computer's sound it picks up from the speakers is
 * taken out (./echo, #846), with the computer's sound as the reference, so
 * what the others say is in the computer's sound only. Each source's samples
 * are placed where they belong; audio is let out a moment behind the newest,
 * so a source a little late still joins in. Samples later than that go to
 * disk only. A stretch neither source covers is silence.
 */
import { MEETING_SAMPLE_RATE } from "../adapters";
import type { AudioSource } from "../api";
import { createEchoCanceller, ECHO_BLOCK, type EchoCanceller } from "./echo";

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
  /** The computer's sound from `out` on. */
  let system = new Int32Array(MEETING_SAMPLE_RATE);
  /** The microphone from `out` on, as captured. */
  let microphone = new Int32Array(MEETING_SAMPLE_RATE);
  /** One past the newest sample heard. */
  let newest = out;
  /** Made when the computer's sound first comes: until then there is no echo to take out. */
  let canceller: EchoCanceller | null = null;

  function room(end: number): void {
    const needed = end - out;
    if (needed <= system.length) return;
    const length = Math.max(needed, system.length * 2);
    const grownSystem = new Int32Array(length);
    grownSystem.set(system);
    system = grownSystem;
    const grownMicrophone = new Int32Array(length);
    grownMicrophone.set(microphone);
    microphone = grownMicrophone;
  }

  /** The microphone from `out` on, `count` samples, with the echo taken out a block at a time. */
  function cleanMicrophone(count: number): Int32Array {
    if (!canceller) return microphone;
    const block = new Int16Array(ECHO_BLOCK);
    for (let at = 0; at < count; at += ECHO_BLOCK) {
      const length = Math.min(ECHO_BLOCK, count - at);
      block.fill(0);
      for (let i = 0; i < length; i++) {
        block[i] = Math.max(-32768, Math.min(32767, microphone[at + i] as number));
      }
      microphone.set(canceller.cancel(block, out + at).subarray(0, length), at);
    }
    return microphone;
  }

  function take(until: number): MixedAudio | null {
    if (until <= out) return null;
    const count = until - out;
    const cleaned = cleanMicrophone(count);
    const samples = new Int16Array(count);
    for (let i = 0; i < count; i++) {
      const sum = (system[i] as number) + (cleaned[i] as number);
      samples[i] = Math.max(-32768, Math.min(32767, sum));
    }
    for (const buffer of [system, microphone]) {
      buffer.copyWithin(0, count);
      buffer.fill(0, buffer.length - count);
    }
    const atMs = msAt(out);
    out = until;
    return { atMs, samples };
  }

  /** Adds a source's samples, which start at `atMs` in the Meeting's time. */
  function put(source: AudioSource, samples: Int16Array, atMs: number): void {
    let start = sampleAt(atMs);
    const end = start + samples.length;
    if (source === "system") {
      canceller ??= createEchoCanceller();
      canceller.reference(samples, start);
    }
    if (end > out) {
      const skip = Math.max(0, out - start);
      start += skip;
      room(end);
      const target = source === "system" ? system : microphone;
      for (let i = skip; i < samples.length; i++) {
        const at = start - out + (i - skip);
        target[at] = (target[at] as number) + (samples[i] as number);
      }
    }
    newest = Math.max(newest, end);
  }

  return {
    put,

    /** Adds a source's samples, as `put`, and returns what is ready to go, in whole blocks. */
    add(source: AudioSource, samples: Int16Array, atMs: number): MixedAudio | null {
      put(source, samples, atMs);
      const ready = newest - sampleAt(DELAY_MS) - out;
      return take(out + Math.floor(ready / ECHO_BLOCK) * ECHO_BLOCK);
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
