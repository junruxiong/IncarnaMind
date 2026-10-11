/**
 * The capture window's AudioWorklet (ADR-0015): it collects a track's samples,
 * mixed to mono and resampled to the context's 16 kHz by Web Audio, and posts
 * them to the page in frames of 100 ms. "flush" posts what it holds, then
 * "flushed".
 */

declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
}
declare function registerProcessor(
  name: string,
  processor: new () => AudioWorkletProcessor & {
    process(inputs: Float32Array[][]): boolean;
  },
): void;

/** 100 ms at 16 kHz. */
const FRAME = 1600;

class PcmFrames extends AudioWorkletProcessor {
  private frame = new Float32Array(FRAME);
  private filled = 0;

  constructor() {
    super();
    this.port.onmessage = () => {
      if (this.filled > 0) this.port.postMessage(this.frame.slice(0, this.filled));
      this.filled = 0;
      this.port.postMessage("flushed");
    };
  }

  process(inputs: Float32Array[][]): boolean {
    const samples = inputs[0]?.[0];
    if (!samples) return true;
    let at = 0;
    while (at < samples.length) {
      const taken = Math.min(samples.length - at, FRAME - this.filled);
      this.frame.set(samples.subarray(at, at + taken), this.filled);
      this.filled += taken;
      at += taken;
      if (this.filled === FRAME) {
        this.port.postMessage(this.frame, [this.frame.buffer]);
        this.frame = new Float32Array(FRAME);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-frames", PcmFrames);

export {};
