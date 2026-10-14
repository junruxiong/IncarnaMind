/**
 * What the main process and the hidden capture window say to each other
 * while a Meeting records (ADR-0015) or reads its shared screen (ADR-0024):
 * the channels, and the bridge the window's preload gives its page.
 */
import type { AudioSource } from "../core/api";

/** How wide the shared window's thumbnails are: about a 20th of a 1080p frame, enough to see a slide change. */
export const SCREEN_THUMBNAIL_WIDTH = 96;

/** The capture window's page reaches its preload as `window.incarnamindCapture`. */
export const CAPTURE_BRIDGE_KEY = "incarnamindCapture";

/** The in-memory session only the capture window uses: the one place media is allowed. */
export const CAPTURE_PARTITION = "meeting-capture";

export const CAPTURE_CHANNELS = {
  /** Main → page: start capturing (`CaptureStart`). */
  start: "capture:start",
  /** Main → page: stop, send what is left, then say "stopped". */
  stop: "capture:stop",
  /** Page → main: (source, Int16Array) of 16 kHz mono samples. */
  audio: "capture:audio",
  /** Page → main: (source, `CaptureState`). */
  state: "capture:state",
  /** Page → main: everything captured has been sent. */
  stopped: "capture:stopped",
  /** Main → page: capture the shared window (ADR-0024), which the next display request gives. */
  screenStart: "capture:screen-start",
  /** Page → main: `ScreenStarted`. */
  screenStarted: "capture:screen-started",
  /** Page → main: (width, height, pixels) of a small grey thumbnail, about once a second. */
  screenSample: "capture:screen-sample",
  /** Main → page: (id) send the window as it is now, at full size. */
  screenGrab: "capture:screen-grab",
  /** Page → main: (id, `ScreenGrabbed`). */
  screenFrame: "capture:screen-frame",
  /** Page → main: the shared window's capture ended by itself: closed, or stopped from outside. */
  screenEnded: "capture:screen-ended",
  /** Main → page: stop capturing the shared window. */
  screenStop: "capture:screen-stop",
} as const;

export interface CaptureStart {
  /** Open the microphone: false when the OS has already refused it. */
  microphone: boolean;
  /** Record the computer's sound too. */
  systemAudio: boolean;
}

/** How a source is doing, as the capture window sees it (`CaptureState` in the core's adapters). */
export type CaptureWindowState = "recording" | "denied" | "unavailable" | "ended";

/** How the shared window's capture started: its name, or why it didn't. */
export type ScreenStarted =
  | { started: true }
  | { started: false; reason: "whole-screen" | "denied" | "closed"; message: string };

/** The shared window as it was when asked for: grey to read, and a JPEG to show. */
export type ScreenGrabbed =
  | { width: number; height: number; pixels: Uint8Array; image: Uint8Array }
  | { error: string };

export interface CaptureBridge {
  onStart(listener: (request: CaptureStart) => void): void;
  onStop(listener: () => void): void;
  audio(source: AudioSource, samples: Int16Array): void;
  state(source: AudioSource, state: CaptureWindowState): void;
  stopped(): void;
  onScreenStart(listener: () => void): void;
  onScreenGrab(listener: (id: number) => void): void;
  onScreenStop(listener: () => void): void;
  screenStarted(started: ScreenStarted): void;
  screenSample(width: number, height: number, pixels: Uint8Array): void;
  screenFrame(id: number, frame: ScreenGrabbed): void;
  screenEnded(): void;
}

/** Web Audio's samples (-1 to 1) as 16-bit PCM, clipped. */
export function toPcm16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i] ?? 0));
    out[i] = sample < 0 ? Math.round(sample * 0x8000) : Math.round(sample * 0x7fff);
  }
  return out;
}
