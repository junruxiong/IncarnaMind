/**
 * Removes the computer's sound from the microphone (#846). On laptop
 * speakers the microphone hears the others again, 20 to 80 ms late, quieter
 * and coloured by the room; Chromium's own canceller can't help, because it
 * only removes sound the page itself plays (ADR-0015).
 *
 * This is Speex's MDF (Valin, "On Adjusting the Learning Rate in Frequency
 * Domain Echo Cancellation With Double-Talk", 2007): a partitioned-block
 * frequency-domain adaptive filter whose learning rate follows how much
 * residual echo is left, so it slows down while the User talks over the
 * others, with a background filter that adapts and a foreground filter that
 * takes its weights only once they do better. With headphones there is no
 * echo to learn, the filter stays near zero and the microphone passes
 * through.
 *
 * The two sources are timestamped apart (./index), so where the echo sits
 * against its reference moves: the filter looks a little ahead of the
 * microphone as well as behind it, and when the echo's strongest partition
 * nears either end of the filter, the filter's window moves to keep it in.
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

const PREEMPHASIS = 0.9;
const MIN_LEAK = 0.005;
const SPEC_AVERAGE = ECHO_BLOCK / 16_000;
const BETA0 = (2 * ECHO_BLOCK) / 16_000;
const BETA_MAX = (0.5 * ECHO_BLOCK) / 16_000;
const VAR1_SMOOTH = 0.36;
const VAR2_SMOOTH = 0.7225;
const VAR1_UPDATE = 0.5;
const VAR2_UPDATE = 0.25;
const VAR_BACKTRACK = 4;

const spectra = (count: number) => Array.from({ length: count }, () => new Float64Array(BINS));

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
  const foregroundEcho = new Float64Array(ECHO_BLOCK);
  const backgroundEcho = new Float64Array(ECHO_BLOCK);
  const estimate = new Float64Array(ECHO_BLOCK);
  const errorSignal = new Float64Array(ECHO_BLOCK);
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
  let lastMicrophone = 0;
  let lastEcho = 0;

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
    initialProp();
  }

  const referenceAt = (sample: number) => {
    const index = sample - referenceFrom;
    return index >= 0 && index < reference.length && sample < referenceEnd
      ? (reference[index] as number)
      : 0;
  };

  /** The pre-emphasised reference from `start`, a window long, into `time`, then its spectrum into partition `slot`. */
  function referenceSpectrum(start: number, slot: number): void {
    let previous = referenceAt(start - 1);
    for (let i = 0; i < WINDOW; i++) {
      const value = referenceAt(start + i);
      time[i] = value - PREEMPHASIS * previous;
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
    let strongest = 0;
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
      const ss = 0.35 / PARTITIONS;
      if (nextAt !== at) reload(at);
      nextAt = at + ECHO_BLOCK;
      let clipped = false;
      for (let i = 0; i < ECHO_BLOCK; i++) {
        const input = block[i] as number;
        if (input <= -32_000 || input >= 32_000) clipped = true;
        microphone[i] = input - PREEMPHASIS * lastMicrophone;
        lastMicrophone = input;
      }

      for (let j = PARTITIONS; j > 0; j--) {
        (xRe[j] as Float64Array).set(xRe[j - 1] as Float64Array);
        (xIm[j] as Float64Array).set(xIm[j - 1] as Float64Array);
      }
      const start = windowStart(at, 0);
      referenceSpectrum(start, 0);
      let sxx = 0;
      for (let i = ECHO_BLOCK; i < WINDOW; i++) sxx += (time[i] as number) ** 2;
      const heard = referenceEnd >= start + WINDOW;

      filterOutput(fRe, fIm, foregroundEcho);
      let sff = 0;
      for (let i = 0; i < ECHO_BLOCK; i++) {
        errorSignal[i] = (microphone[i] as number) - (foregroundEcho[i] as number);
        sff += (errorSignal[i] as number) ** 2;
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
        power[k] = (1 - ss) * (power[k] as number) + 1 + ss * (xf[k] as number);
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
      forget(windowStart(at + ECHO_BLOCK, PARTITIONS) - HISTORY);
      return cleaned;
    },
  };
}

export type EchoCanceller = ReturnType<typeof createEchoCanceller>;
