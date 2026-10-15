/**
 * Removes the computer's sound from the microphone (#846). On laptop
 * speakers the microphone hears the others again, 20 to 80 ms late, quieter
 * and coloured by the room; Chromium's own canceller can't help, because it
 * only removes sound the page itself plays (ADR-0015).
 *
 * Derived from libspeexdsp's mdf.c (BSD-3-Clause, Copyright (C) 2003-2008
 * Jean-Marc Valin), whose notice is in resources/notices/speexdsp.txt and
 * ships with the app. It is Speex's MDF (Valin, "On Adjusting the Learning
 * Rate in Frequency Domain Echo Cancellation With Double-Talk", 2007): a
 * partitioned-block frequency-domain adaptive filter whose learning rate
 * follows how much residual echo is left, so it slows down while the User
 * talks over the others, with a background filter that adapts and a
 * foreground filter that takes its weights only once they do better. With
 * headphones there is no echo to learn, the filter stays near zero and the
 * microphone passes through.
 *
 * The two sources are timestamped apart (./index), so where the echo sits
 * against its reference moves: the filter looks a little ahead of the
 * microphone as well as behind it, its window moves when the echo's
 * strongest partition nears either end, and when the two clocks drift apart
 * the reference is read later or earlier at the rate the echo slides.
 */
import { createRealFft } from "./fft";

/** 16 ms at 16 kHz: the filter works a block at a time. */
export const ECHO_BLOCK = 256;
const WINDOW = 2 * ECHO_BLOCK;
const BINS = ECHO_BLOCK + 1;
/** 320 ms of filter: the echo's delay, wherever it sits, and the room's tail after it. */
const PARTITIONS = 20;
/** Where the filter starts, in blocks of reference ahead of the microphone: an echo may be stamped before its source. */
const LEAD = 5;
/** The most reference the filter reads ahead of the microphone: the mixer holds audio 300 ms. */
const MAX_LEAD = 12;
/** The partition the echo's strongest part is moved to when it nears an end. */
const TARGET = LEAD;
/** Blocks in a row the echo's strongest part must sit near an end before the window moves. */
const MOVE_AFTER = 40;
/** Reference kept before the oldest block the filter reads, for moving its window. */
const HISTORY = MAX_LEAD * ECHO_BLOCK + 16_000;
/** How often, in blocks, the echo's position is compared with where it was, to measure drift. */
const DRIFT_EVERY = 32;
/** How alike the two impulse responses must be for their offset to count: otherwise the filter was still learning. */
const SAME_SHAPE = 0.9;
/** How much of the drift measured each time is corrected. */
const DRIFT_GAIN = 0.3;
/** The most drift corrected, as a share of a sample per sample: clocks a little apart differ by about 1e-4. */
const MAX_DRIFT = 1e-3;
/** The reference is read between its samples, once drift is corrected, through a windowed sinc. */
const SINC_TAPS = 16;
const SINC_PHASES = 256;

// Speex's constants (libspeexdsp/mdf.c), for 16 kHz and 16 ms blocks.
const PREEMPHASIS = 0.9;
const MIN_LEAK = 0.005;
const SMOOTHING = 0.35 / PARTITIONS;
const SPEC_AVERAGE = ECHO_BLOCK / 16_000;
const BETA0 = (2 * ECHO_BLOCK) / 16_000;
const BETA_MAX = (0.5 * ECHO_BLOCK) / 16_000;
const VAR1_SMOOTH = 0.36;
const VAR2_SMOOTH = 0.7225;
const VAR1_UPDATE = 0.5;
const VAR2_UPDATE = 0.25;
const VAR_BACKTRACK = 4;

const spectra = (count: number) => Array.from({ length: count }, () => new Float64Array(BINS));

/** For each fraction of a sample, the taps that read the reference that much later than a sample. */
const SINC = Array.from({ length: SINC_PHASES }, (_, phase) => {
  const fraction = phase / SINC_PHASES;
  const half = SINC_TAPS / 2;
  const taps = Float64Array.from({ length: SINC_TAPS }, (_, t) => {
    const x = t - (half - 1) - fraction;
    const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    return sinc * (0.5 + 0.5 * Math.cos((Math.PI * x) / (half + 0.5)));
  });
  const sum = taps.reduce((total, tap) => total + tap, 0);
  return taps.map((tap) => tap / sum);
});

export function createEchoCanceller() {
  const fft = createRealFft(WINDOW);
  const hann = Float64Array.from(
    { length: WINDOW },
    (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / WINDOW),
  );

  let referenceFrom = 0;
  let reference = new Float64Array(4 * 16_000);
  /** One past the newest reference sample heard. */
  let referenceEnd = 0;

  /** The reference's spectra, newest first; one more than the partitions, for adapting on the block before. */
  const xRe = spectra(PARTITIONS + 1);
  const xIm = spectra(PARTITIONS + 1);
  const wRe = spectra(PARTITIONS);
  const wIm = spectra(PARTITIONS);
  const fRe = spectra(PARTITIONS);
  const fIm = spectra(PARTITIONS);
  const eRe = new Float64Array(BINS);
  const eIm = new Float64Array(BINS);
  const yRe = new Float64Array(BINS);
  const yIm = new Float64Array(BINS);
  const accRe = new Float64Array(BINS);
  const accIm = new Float64Array(BINS);
  const time = new Float64Array(WINDOW);
  /** The reference samples one window reads, with the sinc's reach either side. */
  const raw = new Float64Array(WINDOW + 1 + SINC_TAPS);
  const foregroundEcho = new Float64Array(ECHO_BLOCK);
  const backgroundEcho = new Float64Array(ECHO_BLOCK);
  const estimate = new Float64Array(ECHO_BLOCK);
  const microphone = new Float64Array(ECHO_BLOCK);
  const xf = new Float64Array(BINS);
  const rf = new Float64Array(BINS);
  const yf = new Float64Array(BINS);
  const power = new Float64Array(BINS);
  const power1 = new Float64Array(BINS).fill(1);
  const eh = new Float64Array(BINS);
  const yh = new Float64Array(BINS);
  const prop = new Float64Array(PARTITIONS);

  let lead = LEAD;
  let nextAt: number | null = null;
  let canAdapt = false;
  let adapted = false;
  let sumAdapt = 0;
  let pey = 1;
  let pyy = 1;
  let leak = 0;
  let davg1 = 0;
  let davg2 = 0;
  let dvar1 = 0;
  let dvar2 = 0;
  let saturated = 0;
  let screwedUp = 0;
  let count = 0;
  let nearEnd = 0;
  let strongest = 0;
  let lastMicrophone = 0;
  let lastEcho = 0;
  /** How much later than its stamps the reference is read, in samples, to make up for drift. */
  let tau = 0;
  /** How much `tau` grows each sample. */
  let drift = 0;
  /** The echo's impulse response around its strongest partition when drift was last measured. */
  let snapshot: { from: number; lead: number; taps: Float64Array } | null = null;
  let farBlocks = 0;

  function initialProp(): void {
    const decay = Math.exp(-2.4 / PARTITIONS);
    let value = 0.7;
    let sum = 0;
    for (let j = 0; j < PARTITIONS; j++) {
      prop[j] = value;
      sum += value;
      value *= decay;
    }
    for (let j = 0; j < PARTITIONS; j++) prop[j] = (0.8 * (prop[j] as number)) / sum;
  }
  initialProp();

  function reset(): void {
    for (const each of [...wRe, ...wIm, ...fRe, ...fIm, ...xRe, ...xIm]) each.fill(0);
    for (const each of [power, eh, yh, eRe, eIm]) each.fill(0);
    power1.fill(1);
    adapted = false;
    sumAdapt = 0;
    pey = 1;
    pyy = 1;
    leak = 0;
    davg1 = davg2 = dvar1 = dvar2 = 0;
    saturated = 0;
    screwedUp = 0;
    canAdapt = false;
    nextAt = null;
    snapshot = null;
    initialProp();
  }

  const referenceAt = (sample: number) => {
    const index = sample - referenceFrom;
    return index >= 0 && index < reference.length && sample < referenceEnd
      ? (reference[index] as number)
      : 0;
  };

  /**
   * The pre-emphasised reference from `start`, a window long and read `tau`
   * samples late, into `time`, then its spectrum into partition `slot`.
   */
  function referenceSpectrum(start: number, slot: number): void {
    const late = start - tau;
    let base = Math.floor(late);
    let phase = Math.round((late - base) * SINC_PHASES);
    if (phase === SINC_PHASES) {
      base++;
      phase = 0;
    }
    const reach = phase === 0 ? 0 : SINC_TAPS / 2;
    const first = base - 1 - reach;
    for (let i = 0; i < raw.length; i++) raw[i] = referenceAt(first + i);
    const taps = SINC[phase] as Float64Array;
    let previous = 0;
    for (let i = -1; i < WINDOW; i++) {
      let value: number;
      if (phase === 0) {
        value = raw[i + 1] as number;
      } else {
        value = 0;
        const from = i + 1 + reach - (SINC_TAPS / 2 - 1);
        for (let t = 0; t < SINC_TAPS; t++)
          value += (taps[t] as number) * (raw[from + t] as number);
      }
      if (i >= 0) time[i] = value - PREEMPHASIS * previous;
      previous = value;
    }
    fft.forward(time, xRe[slot] as Float64Array, xIm[slot] as Float64Array);
  }

  const windowStart = (at: number, partition: number) =>
    at + lead * ECHO_BLOCK - ECHO_BLOCK - partition * ECHO_BLOCK;

  function filterOutput(re: Float64Array[], im: Float64Array[], out: Float64Array): void {
    accRe.fill(0);
    accIm.fill(0);
    for (let j = 0; j < PARTITIONS; j++) {
      const ar = re[j] as Float64Array;
      const ai = im[j] as Float64Array;
      const br = xRe[j] as Float64Array;
      const bi = xIm[j] as Float64Array;
      for (let k = 0; k < BINS; k++) {
        const a = ar[k] as number;
        const b = ai[k] as number;
        const c = br[k] as number;
        const d = bi[k] as number;
        accRe[k] = (accRe[k] as number) + a * c - b * d;
        accIm[k] = (accIm[k] as number) + a * d + b * c;
      }
    }
    fft.inverse(accRe, accIm, time);
    for (let i = 0; i < ECHO_BLOCK; i++) out[i] = time[ECHO_BLOCK + i] as number;
  }

  /** Keeps a partition's impulse response a block long, so the blocks don't wrap round. */
  function constrain(j: number): void {
    fft.inverse(wRe[j] as Float64Array, wIm[j] as Float64Array, time);
    time.fill(0, ECHO_BLOCK);
    fft.forward(time, wRe[j] as Float64Array, wIm[j] as Float64Array);
  }

  function adjustProp(): void {
    let max = 1;
    for (let j = 0; j < PARTITIONS; j++) {
      let sum = 1;
      const re = wRe[j] as Float64Array;
      const im = wIm[j] as Float64Array;
      for (let k = 0; k < BINS; k++) sum += (re[k] as number) ** 2 + (im[k] as number) ** 2;
      prop[j] = Math.sqrt(sum);
      max = Math.max(max, prop[j] as number);
    }
    let total = 1;
    for (let j = 0; j < PARTITIONS; j++) {
      prop[j] = (prop[j] as number) + 0.1 * max;
      total += prop[j] as number;
    }
    for (let j = 0; j < PARTITIONS; j++) prop[j] = (0.99 * (prop[j] as number)) / total;
  }

  /**
   * Moves the filter's window when the echo's strongest partition has sat near
   * either end of it for a while: the delay between the sources drifted.
   */
  function follow(at: number): void {
    if (!adapted) return;
    strongest = 0;
    let peak = 0;
    let total = 0;
    for (let j = 0; j < PARTITIONS; j++) {
      const re = fRe[j] as Float64Array;
      const im = fIm[j] as Float64Array;
      let energy = 0;
      for (let k = 0; k < BINS; k++) energy += (re[k] as number) ** 2 + (im[k] as number) ** 2;
      total += energy;
      if (energy > peak) {
        peak = energy;
        strongest = j;
      }
    }
    const nearAnEnd = strongest <= 1 || strongest >= PARTITIONS - 6;
    nearEnd = peak > 0.2 * total && nearAnEnd ? nearEnd + 1 : 0;
    if (nearEnd < MOVE_AFTER) return;
    nearEnd = 0;
    const moved = Math.min(MAX_LEAD, lead - (strongest - TARGET)) - lead;
    if (moved === 0) return;
    lead += moved;
    const shift = -moved;
    for (const filter of [wRe, wIm, fRe, fIm]) {
      const old = filter.slice();
      for (let j = 0; j < PARTITIONS; j++) {
        const from = j + shift;
        const target = filter[j] as Float64Array;
        if (from >= 0 && from < PARTITIONS) target.set(old[from] as Float64Array);
        else target.fill(0);
      }
    }
    reload(at);
    davg1 = davg2 = dvar1 = dvar2 = 0;
  }

  /**
   * Compares the echo's impulse response around its strongest partition with
   * where it was `DRIFT_EVERY` blocks ago. When the two clocks run a little
   * apart, the echo slides along the filter faster than it can follow; the
   * reference is then read later or earlier at the rate it slid.
   */
  function measureDrift(): void {
    const from = Math.max(0, Math.min(PARTITIONS - 3, strongest - 1));
    const taps = new Float64Array(3 * ECHO_BLOCK);
    for (let j = 0; j < 3; j++) {
      fft.inverse(fRe[from + j] as Float64Array, fIm[from + j] as Float64Array, time);
      taps.set(time.subarray(0, ECHO_BLOCK), j * ECHO_BLOCK);
    }
    const before = snapshot;
    snapshot = { from, lead, taps };
    const active = farBlocks >= DRIFT_EVERY / 2;
    farBlocks = 0;
    if (!before || before.from !== from || before.lead !== lead || !active) return;
    const correlation = (lag: number) => {
      let sum = 0;
      for (let t = Math.max(0, lag); t < taps.length + Math.min(0, lag); t++) {
        sum += (taps[t] as number) * (before.taps[t - lag] as number);
      }
      return sum;
    };
    const lags = [-3, -2, -1, 0, 1, 2, 3].map(correlation);
    let best = 3;
    for (let i = 0; i < lags.length; i++)
      if ((lags[i] as number) > (lags[best] as number)) best = i;
    const energyOf = (signal: Float64Array) => signal.reduce((sum, tap) => sum + tap * tap, 0);
    const alike = (lags[best] as number) / Math.sqrt(energyOf(taps) * energyOf(before.taps));
    if (best === 0 || best === lags.length - 1 || alike < SAME_SHAPE) return;
    const left = lags[best - 1] as number;
    const middle = lags[best] as number;
    const right = lags[best + 1] as number;
    const curvature = left - 2 * middle + right;
    const slid = best - 3 + (curvature < 0 ? (left - right) / (2 * curvature) : 0);
    drift += (DRIFT_GAIN * slid) / (DRIFT_EVERY * ECHO_BLOCK);
    drift = Math.max(-MAX_DRIFT, Math.min(MAX_DRIFT, drift));
  }

  /** The reference's spectra as the block at `at` expects to find them, before it adds its own. */
  function reload(at: number): void {
    for (let j = 0; j < PARTITIONS; j++) referenceSpectrum(windowStart(at, j + 1), j);
    canAdapt = false;
  }

  function forget(before: number): void {
    const drop = Math.min(before - referenceFrom, referenceEnd - referenceFrom);
    if (drop <= 0) return;
    reference.copyWithin(0, drop);
    reference.fill(0, reference.length - drop);
    referenceFrom += drop;
  }

  return {
    /** The computer's sound, its first sample at `at`, counted in samples of the Meeting's time. */
    reference(samples: Int16Array, at: number): void {
      if (at + samples.length <= referenceFrom) return;
      const needed = at + samples.length - referenceFrom;
      if (needed > reference.length) {
        const grown = new Float64Array(Math.max(needed, reference.length * 2));
        grown.set(reference);
        reference = grown;
      }
      const skip = Math.max(0, referenceFrom - at);
      for (let i = skip; i < samples.length; i++) {
        reference[at - referenceFrom + i] = samples[i] as number;
      }
      referenceEnd = Math.max(referenceEnd, at + samples.length);
    },

    /**
     * A block (`ECHO_BLOCK` samples) of the microphone, its first sample at
     * `at`: returns it with the echo of the computer's sound taken out.
     * Blocks come in order.
     */
    cancel(block: Int16Array, at: number): Int16Array {
      count++;
      if (nextAt !== at) reload(at);
      nextAt = at + ECHO_BLOCK;
      let clipped = false;
      for (let i = 0; i < ECHO_BLOCK; i++) {
        const input = block[i] as number;
        if (input <= -32_000 || input >= 32_000) clipped = true;
        microphone[i] = input - PREEMPHASIS * lastMicrophone;
        lastMicrophone = input;
      }

      tau += drift * ECHO_BLOCK;
      // A whole block of drift is the same as reading from one block further along.
      if (tau >= ECHO_BLOCK) {
        tau -= ECHO_BLOCK;
        lead--;
      } else if (tau <= -ECHO_BLOCK && lead < MAX_LEAD) {
        tau += ECHO_BLOCK;
        lead++;
      }
      for (let j = PARTITIONS; j > 0; j--) {
        (xRe[j] as Float64Array).set(xRe[j - 1] as Float64Array);
        (xIm[j] as Float64Array).set(xIm[j - 1] as Float64Array);
      }
      const start = windowStart(at, 0);
      referenceSpectrum(start, 0);
      let sxx = 0;
      for (let i = ECHO_BLOCK; i < WINDOW; i++) sxx += (time[i] as number) ** 2;
      const heard = referenceEnd >= start + WINDOW - Math.min(0, Math.floor(tau)) + SINC_TAPS / 2;
      if (sxx > (WINDOW * 1000) / 64) farBlocks++;

      filterOutput(fRe, fIm, foregroundEcho);
      let sff = 0;
      for (let i = 0; i < ECHO_BLOCK; i++) {
        sff += ((microphone[i] as number) - (foregroundEcho[i] as number)) ** 2;
      }

      if (adapted) adjustProp();
      if (saturated === 0 && canAdapt) {
        for (let j = 0; j < PARTITIONS; j++) {
          const p = prop[j] as number;
          const ar = xRe[j + 1] as Float64Array;
          const ai = xIm[j + 1] as Float64Array;
          const re = wRe[j] as Float64Array;
          const im = wIm[j] as Float64Array;
          for (let k = 0; k < BINS; k++) {
            const w = p * (power1[k] as number);
            const xr = ar[k] as number;
            const xi = ai[k] as number;
            const er = eRe[k] as number;
            const ei = eIm[k] as number;
            re[k] = (re[k] as number) + w * (xr * er + xi * ei);
            im[k] = (im[k] as number) + w * (xr * ei - xi * er);
          }
        }
      } else if (saturated > 0) {
        saturated--;
      }
      for (let j = 0; j < PARTITIONS; j++) {
        if (j === 0 || count % (PARTITIONS - 1) === j - 1) constrain(j);
      }

      filterOutput(wRe, wIm, backgroundEcho);
      let dbf = 10;
      let see = 0;
      for (let i = 0; i < ECHO_BLOCK; i++) {
        const fg = foregroundEcho[i] as number;
        const bg = backgroundEcho[i] as number;
        dbf += (fg - bg) ** 2;
        see += ((microphone[i] as number) - bg) ** 2;
      }

      davg1 = 0.6 * davg1 + 0.4 * (sff - see);
      davg2 = 0.85 * davg2 + 0.15 * (sff - see);
      dvar1 = VAR1_SMOOTH * dvar1 + 0.16 * sff * dbf;
      dvar2 = VAR2_SMOOTH * dvar2 + 0.0225 * sff * dbf;
      const difference = sff - see;
      const better =
        difference * Math.abs(difference) > sff * dbf ||
        davg1 * Math.abs(davg1) > VAR1_UPDATE * dvar1 ||
        davg2 * Math.abs(davg2) > VAR2_UPDATE * dvar2;
      const echo = estimate;
      echo.set(foregroundEcho);
      if (better) {
        davg1 = davg2 = dvar1 = dvar2 = 0;
        for (let j = 0; j < PARTITIONS; j++) {
          (fRe[j] as Float64Array).set(wRe[j] as Float64Array);
          (fIm[j] as Float64Array).set(wIm[j] as Float64Array);
        }
        for (let i = 0; i < ECHO_BLOCK; i++) {
          echo[i] =
            (hann[i + ECHO_BLOCK] as number) * (foregroundEcho[i] as number) +
            (hann[i] as number) * (backgroundEcho[i] as number);
        }
      } else if (
        -difference * Math.abs(difference) > VAR_BACKTRACK * sff * dbf ||
        -davg1 * Math.abs(davg1) > VAR_BACKTRACK * dvar1 ||
        -davg2 * Math.abs(davg2) > VAR_BACKTRACK * dvar2
      ) {
        for (let j = 0; j < PARTITIONS; j++) {
          (wRe[j] as Float64Array).set(fRe[j] as Float64Array);
          (wIm[j] as Float64Array).set(fIm[j] as Float64Array);
        }
        backgroundEcho.set(foregroundEcho);
        see = sff;
        davg1 = davg2 = dvar1 = dvar2 = 0;
      }

      const cleaned = new Int16Array(ECHO_BLOCK);
      for (let i = 0; i < ECHO_BLOCK; i++) {
        lastEcho = (echo[i] as number) + PREEMPHASIS * lastEcho;
        const value = Math.round((block[i] as number) - lastEcho);
        cleaned[i] = Math.max(-32_768, Math.min(32_767, value));
      }
      if (clipped && saturated === 0) saturated = 1;

      let sey = 0;
      let syy = 0;
      let sdd = 0;
      time.fill(0, 0, ECHO_BLOCK);
      for (let i = 0; i < ECHO_BLOCK; i++) {
        const bg = backgroundEcho[i] as number;
        const error = (microphone[i] as number) - bg;
        time[ECHO_BLOCK + i] = error;
        sey += error * bg;
        syy += bg * bg;
        sdd += (microphone[i] as number) ** 2;
      }
      fft.forward(time, eRe, eIm);
      for (let i = 0; i < ECHO_BLOCK; i++) time[ECHO_BLOCK + i] = backgroundEcho[i] as number;
      time.fill(0, 0, ECHO_BLOCK);
      fft.forward(time, yRe, yIm);
      for (let k = 0; k < BINS; k++) {
        rf[k] = (eRe[k] as number) ** 2 + (eIm[k] as number) ** 2;
        yf[k] = (yRe[k] as number) ** 2 + (yIm[k] as number) ** 2;
        xf[k] =
          ((xRe[0] as Float64Array)[k] as number) ** 2 +
          ((xIm[0] as Float64Array)[k] as number) ** 2;
      }

      const limit = WINDOW * 1e9;
      if (!(syy >= 0 && sxx >= 0 && see >= 0) || !(sff < limit && syy < limit && sxx < limit)) {
        reset();
        return block.slice();
      }
      screwedUp = sff / 4 > sdd + (WINDOW * 10_000) / 64 ? screwedUp + 1 : 0;
      if (screwedUp >= 50) {
        reset();
        return cleaned;
      }

      see = Math.max(see, (WINDOW * 100) / 64);
      for (let k = 0; k < BINS; k++) {
        power[k] = (1 - SMOOTHING) * (power[k] as number) + 1 + SMOOTHING * (xf[k] as number);
      }
      let localPey = 1;
      let localPyy = 1;
      for (let k = BINS - 1; k >= 0; k--) {
        const e = (rf[k] as number) - (eh[k] as number);
        const y = (yf[k] as number) - (yh[k] as number);
        localPey += e * y;
        localPyy += y * y;
        eh[k] = (1 - SPEC_AVERAGE) * (eh[k] as number) + SPEC_AVERAGE * (rf[k] as number);
        yh[k] = (1 - SPEC_AVERAGE) * (yh[k] as number) + SPEC_AVERAGE * (yf[k] as number);
      }
      localPyy = Math.sqrt(localPyy);
      localPey /= localPyy;
      const alpha = Math.min(BETA0 * syy, BETA_MAX * see) / see;
      pey = (1 - alpha) * pey + alpha * localPey;
      pyy = Math.max(1, (1 - alpha) * pyy + alpha * localPyy);
      pey = Math.min(pyy, Math.max(MIN_LEAK * pyy, pey));
      leak = pey / pyy;

      let rer = (0.0001 * sxx + 3 * leak * syy) / see;
      rer = Math.min(0.5, Math.max(rer, (sey * sey) / (1 + see * syy)));

      if (!adapted && sumAdapt > PARTITIONS && leak > 0.03) adapted = true;
      if (adapted) {
        for (let k = 0; k < BINS; k++) {
          const e = (rf[k] as number) + 1;
          let r = Math.min(leak * (yf[k] as number), 0.5 * e);
          r = 0.7 * r + 0.3 * rer * e;
          power1[k] = r / (e * ((power[k] as number) + 10));
        }
      } else {
        let rate = 0;
        if (sxx > (WINDOW * 1000) / 64) rate = Math.min(0.25 * sxx, see / 4) / see;
        for (let k = 0; k < BINS; k++) power1[k] = rate / ((power[k] as number) + 10);
        sumAdapt += rate;
      }
      canAdapt = heard;

      follow(at + ECHO_BLOCK);
      if (!adapted) snapshot = null;
      else if (count % DRIFT_EVERY === 0) measureDrift();
      forget(windowStart(at + ECHO_BLOCK, PARTITIONS) - HISTORY);
      return cleaned;
    },
  };
}

export type EchoCanceller = ReturnType<typeof createEchoCanceller>;
