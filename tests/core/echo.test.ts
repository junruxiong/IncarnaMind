import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { AudioCapture, AudioSink, Transcriber } from "../../src/core";
import { createEchoCanceller, ECHO_BLOCK } from "../../src/core/meetings/echo";
import { createRealFft } from "../../src/core/meetings/fft";
import { createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { inProcessTranscriber, NO_TRANSCRIPTION_FILES, until } from "../helpers/transcription";

const RATE = 16_000;
const FIXTURES = join(__dirname, "..", "fixtures", "echo");

/** A fixture's samples: our own 16-bit mono WAV files, with a 44-byte header. */
const clip = (name: string): Float64Array => {
  const bytes = readFileSync(join(FIXTURES, `${name}.wav`));
  const samples = new Int16Array(
    bytes.buffer.slice(bytes.byteOffset + 44, bytes.byteOffset + bytes.byteLength),
  );
  return Float64Array.from(samples);
};

const silence = (seconds: number) => new Float64Array(Math.round(seconds * RATE));

function joined(...parts: Float64Array[]): Float64Array {
  const out = new Float64Array(parts.reduce((sum, part) => sum + part.length, 0));
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

const added = (...signals: Float64Array[]) =>
  Float64Array.from({ length: signals[0]?.length ?? 0 }, (_, i) =>
    signals.reduce((sum, signal) => sum + (signal[i] ?? 0), 0),
  );

function random(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1_103_515_245 + 12_345) % 2_147_483_648;
    return state / 2_147_483_648;
  };
}

/** A room's noise, about 65 dB below full scale. */
function noise(length: number, seed: number): Float64Array {
  const next = random(seed);
  return Float64Array.from({ length }, () => (next() + next() + next() - 1.5) * 35);
}

/**
 * The far end as laptop speakers play it back into the microphone:
 * `delayMs` late (fractional, and changing if a function), half as loud,
 * low-passed at 4 kHz, with a short room tail.
 */
function echoOf(far: Float64Array, delayMs: number | ((at: number) => number)): Float64Array {
  const delayed = new Float64Array(far.length);
  for (let i = 0; i < far.length; i++) {
    const t = i - ((typeof delayMs === "number" ? delayMs : delayMs(i)) * RATE) / 1000;
    const base = Math.floor(t);
    const fraction = t - base;
    let value = 0;
    for (let k = -12; k <= 12; k++) {
      const x = k - fraction;
      const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      value += (far[base + k] ?? 0) * sinc * (0.5 + 0.5 * Math.cos((Math.PI * x) / 13));
    }
    delayed[i] = value;
  }
  const w = Math.tan((Math.PI * 4000) / RATE);
  const n = 1 / (1 + Math.SQRT2 * w + w * w);
  const [b0, a1, a2] = [w * w * n, 2 * (w * w - 1) * n, (1 - Math.SQRT2 * w + w * w) * n];
  const direct = new Float64Array(far.length);
  let [x1, x2, y1, y2] = [0, 0, 0, 0];
  for (let i = 0; i < far.length; i++) {
    const x = delayed[i] as number;
    const y = b0 * (x + 2 * x1 + x2) - a1 * y1 - a2 * y2;
    [x2, x1, y2, y1] = [x1, x, y1, y];
    direct[i] = 0.5 * y;
  }
  const next = random(7);
  const out = new Float64Array(direct);
  for (let tap = 0; tap < 40; tap++) {
    const lag = Math.round((0.002 + 0.2 * next()) * RATE);
    const gain = (next() - 0.5) * 0.25 * Math.exp((-6.9 * lag) / (0.25 * RATE));
    for (let i = lag; i < out.length; i++)
      out[i] = (out[i] as number) + gain * (direct[i - lag] as number);
  }
  return out;
}

/** The microphone through the canceller, the far end reaching it 300 ms ahead, as the mixer gives it. */
function cancel(microphone: Float64Array, far: Float64Array): Float64Array {
  const canceller = createEchoCanceller();
  const pcm = (signal: Float64Array) =>
    Int16Array.from(signal, (value) => Math.max(-32768, Math.min(32767, Math.round(value))));
  const [mic, reference] = [pcm(microphone), pcm(far)];
  const out = new Float64Array(mic.length);
  let given = 0;
  for (let at = 0; at + ECHO_BLOCK <= mic.length; at += ECHO_BLOCK) {
    const until = Math.min(reference.length, at + 0.3 * RATE);
    if (until > given) {
      canceller.reference(reference.subarray(given, until), given);
      given = until;
    }
    out.set(canceller.cancel(mic.subarray(at, at + ECHO_BLOCK), at), at);
  }
  return out;
}

const energy = (signal: Float64Array, from: number, to: number) => {
  let sum = 0;
  for (let i = from; i < to; i++) sum += (signal[i] as number) ** 2;
  return sum;
};

/** How far below `wanted` the `unwanted` part is over a stretch, in dB. */
const below = (wanted: Float64Array, unwanted: Float64Array, from: number, to: number) =>
  10 * Math.log10(energy(wanted, from, to) / energy(unwanted, from, to));

/** The far end's line `times` times, with pauses: where each one starts and ends. */
function turns(line: Float64Array, times: number) {
  const pause = silence(0.8);
  const parts = Array.from({ length: times }, () => [pause, line]).flat();
  const far = joined(...parts, pause);
  const spans = Array.from({ length: times }, (_, i) => {
    const from = (i + 1) * pause.length + i * line.length;
    return [from, from + line.length] as const;
  });
  return { far, spans };
}

describe.each(["en", "zh"])("Taking the far end out of the microphone (%s)", (language) => {
  const near = clip(`${language}-near`);
  const line = clip(`${language}-far`);

  test("once it has heard the other side a few seconds, their echo is more than 15 dB down", () => {
    const { far, spans } = turns(line, 3);
    const echo = echoOf(far, 45);
    const floor = noise(far.length, 1);
    const out = cancel(added(echo, floor), far);
    const residual = Float64Array.from(out, (value, i) => value - (floor[i] as number));
    const [from, to] = spans[2] as readonly [number, number];
    expect(below(echo, residual, from, to)).toBeGreaterThan(15);
  });

  test("with headphones, nothing echoes and the User's voice passes through unchanged", () => {
    const { far } = turns(line, 3);
    const voice = new Float64Array(far.length);
    voice.set(near, 2 * RATE);
    const microphone = added(voice, noise(far.length, 2));
    const out = cancel(microphone, far);
    const change = Float64Array.from(out, (value, i) => value - (microphone[i] as number));
    expect(below(microphone, change, 0, microphone.length)).toBeGreaterThan(40);
  });

  test("talking over the other side, the User's voice is kept and their echo still goes", () => {
    const { far, spans } = turns(line, 4);
    const [from, to] = spans[3] as readonly [number, number];
    const voice = new Float64Array(far.length);
    voice.set(near.subarray(0, Math.min(near.length, far.length - from)), from);
    const echo = echoOf(far, 45);
    const floor = noise(far.length, 3);
    const out = cancel(added(voice, echo, floor), far);
    const left = Float64Array.from(
      out,
      (value, i) => value - (voice[i] as number) - (floor[i] as number),
    );
    const end = Math.min(to, from + near.length);
    expect(below(voice, left, from, end)).toBeGreaterThan(20);
    expect(below(echo, left, from, end)).toBeGreaterThan(15);
  });

  const { far: longer, spans: longerSpans } = turns(line, 9);
  test.each([
    ["stamped 50 ms before its source", () => -50],
    ["jumping from 40 to 150 ms", (at: number) => (at < longer.length / 2 ? 40 : 150)],
    ["drifting 50 ppm", (at: number) => 45 + (at / RATE) * 0.05],
  ])("an echo %s is followed", (_name, delay) => {
    const [far, spans] = [longer, longerSpans];
    const echo = echoOf(far, delay);
    const floor = noise(far.length, 4);
    const out = cancel(added(echo, floor), far);
    const residual = Float64Array.from(out, (value, i) => value - (floor[i] as number));
    const [from] = spans[7] as readonly [number, number];
    const [, to] = spans[8] as readonly [number, number];
    expect(below(echo, residual, from, to)).toBeGreaterThan(15);
  });
});

describe("A Meeting on laptop speakers", () => {
  test("what the others say reaches the Transcript from the computer's sound only, not again from the microphone", async () => {
    const dataDir = await createTempDataFolder();
    const clock = manualClock();
    let sink: AudioSink | null = null;
    const capture: AudioCapture = {
      systemAudio: true,
      start(_request, given) {
        sink = given;
        given.state("microphone", "recording");
        given.state("system", "recording");
        return { stop: async () => {} };
      },
      openSettings: async () => {},
    };
    const engine = inProcessTranscriber();
    const heard: Int16Array[] = [];
    let live: string | null = null;
    const transcriber: Transcriber = {
      ...engine.transcriber,
      push(stream, samples) {
        live ??= stream;
        if (stream === live) heard.push(samples.slice());
        engine.transcriber.push(stream, samples);
      },
    };
    const core = startCore(dataDir, {
      audioCapture: capture,
      transcriber,
      transcriptionModelSource: NO_TRANSCRIPTION_FILES,
      now: clock.now,
    });
    await core.updateSettings({ device: { meetingsEnabled: true, meetingNoticeAccepted: true } });
    await core.setOutputLocation(join(dataDir, "Documents", "IncarnaMind"));

    // The others speak three times, then the User alone.
    const { far, spans } = turns(clip("en-far"), 3);
    const voice = joined(new Float64Array(far.length), clip("en-near"), silence(0.5));
    const system = joined(far, new Float64Array(voice.length - far.length));
    const echo = echoOf(system, 45);
    const floor = noise(voice.length, 5);
    const microphone = added(voice, echo, floor);
    const pcm = (signal: Float64Array, from: number) =>
      Int16Array.from(signal.subarray(from, from + RATE / 10), (value) => Math.round(value));

    const { id } = await core.createMeeting();
    await core.startMeeting({ meetingId: id, systemAudio: true });
    for (let at = 0; at + RATE / 10 <= microphone.length; at += RATE / 10) {
      clock.advance(100);
      (sink as AudioSink | null)?.audio("microphone", pcm(microphone, at));
      (sink as AudioSink | null)?.audio("system", pcm(system, at));
    }
    await core.stopMeeting(id);
    await until(core, id, (transcript) => transcript.state === "saved");

    // What the transcriber heard, less the computer's sound, is what the microphone added.
    const mixed = Float64Array.from(heard.flatMap((samples) => Array.from(samples)));
    const fromMicrophone = Float64Array.from(mixed, (value, i) => value - (system[i] as number));
    const left = Float64Array.from(
      fromMicrophone,
      (value, i) => value - (voice[i] as number) - (floor[i] as number),
    );
    const [from, to] = spans[2] as readonly [number, number];
    expect(below(echo, left, from, to)).toBeGreaterThan(15);
    expect(below(voice, left, far.length, mixed.length)).toBeGreaterThan(30);
  });
});

describe("The canceller's licence", () => {
  test("the notice of the Speex code it is derived from ships with the app", () => {
    const root = join(__dirname, "..", "..");
    const notice = readFileSync(join(root, "resources", "notices", "speexdsp.txt"), "utf8");
    expect(notice).toContain("Copyright (C) 2003-2008 Jean-Marc Valin");
    expect(notice).toContain(
      "Redistributions of source code must retain the above copyright notice",
    );
    expect(notice).toContain("POSSIBILITY OF SUCH DAMAGE.");
    const builder = readFileSync(join(root, "electron-builder.yml"), "utf8");
    expect(builder).toMatch(/- from: resources\/notices\n\s+to: notices\n/);
  });
});

describe("The canceller's FFT", () => {
  test("gives the DFT, divided by its size, and the inverse gives the samples back", () => {
    const fft = createRealFft(32);
    const signal = Float64Array.from({ length: 32 }, (_, i) => Math.sin(i * 0.7) + (i % 5) - 2);
    const re = new Float64Array(17);
    const im = new Float64Array(17);
    fft.forward(signal, re, im);
    for (let k = 0; k <= 16; k++) {
      let [dftRe, dftIm] = [0, 0];
      for (let n = 0; n < 32; n++) {
        dftRe += (signal[n] as number) * Math.cos((2 * Math.PI * k * n) / 32);
        dftIm -= (signal[n] as number) * Math.sin((2 * Math.PI * k * n) / 32);
      }
      expect(re[k]).toBeCloseTo(dftRe / 32, 12);
      expect(im[k]).toBeCloseTo(dftIm / 32, 12);
    }
    const back = new Float64Array(32);
    fft.inverse(re, im, back);
    for (let n = 0; n < 32; n++) expect(back[n]).toBeCloseTo(signal[n] as number, 12);
  });
});
