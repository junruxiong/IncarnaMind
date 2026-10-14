/**
 * The shared window, in the hidden capture window's page (ADR-0024). Asked to
 * start, it calls `getDisplayMedia` for video only, which the main process
 * answers with the window the User chose; a whole screen is refused. Once a second it draws the window small, in grey, and
 * sends that thumbnail; asked for a frame, it draws the window at full size,
 * in grey to read and as a JPEG to show. Nothing is kept here: each picture
 * is drawn when it is asked for and sent at once.
 */
import {
  type CaptureBridge,
  SCREEN_THUMBNAIL_WIDTH,
  type ScreenStarted,
} from "../../shared/capture";

/** How often the window is sampled. */
const SAMPLE_MS = 1000;
/** The largest side read: the text recogniser reads no more (PP-OCRv5's detection cap). */
const MAX_READ_SIDE = 2560;
/** The width of the picture a Citation's card shows. */
const SHOWN_WIDTH = 1280;
const SHOWN_QUALITY = 0.8;
/** At most this many frames a second: what is read changes slowly, and fewer cost less. */
const MAX_FRAME_RATE = 2;

let stream: MediaStream | null = null;
let video: HTMLVideoElement | null = null;
let sampling: ReturnType<typeof setInterval> | null = null;

/** The video as it is now, drawn at `width` × `height`. */
function draw(width: number, height: number): OffscreenCanvas {
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context || !video) throw new Error("The shared window can't be drawn.");
  context.drawImage(video, 0, 0, width, height);
  return canvas;
}

/** The canvas in grey, one byte a pixel, as the text recogniser reads it. */
function grey(canvas: OffscreenCanvas): Uint8Array {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("The shared window can't be read.");
  const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = new Uint8Array(canvas.width * canvas.height);
  for (let at = 0; at < pixels.length; at++) {
    const base = at * 4;
    pixels[at] = Math.round(
      0.299 * (data[base] ?? 0) + 0.587 * (data[base + 1] ?? 0) + 0.114 * (data[base + 2] ?? 0),
    );
  }
  return pixels;
}

/** The size the video is drawn at, its longer side at most `longest` and its width at most `widest`. */
function scaled(longest: number, widest = Number.POSITIVE_INFINITY) {
  const width = video?.videoWidth ?? 0;
  const height = video?.videoHeight ?? 0;
  const ratio = Math.min(1, longest / Math.max(width, height, 1), widest / Math.max(width, 1));
  return {
    width: Math.max(1, Math.round(width * ratio)),
    height: Math.max(1, Math.round(height * ratio)),
  };
}

function stop(): void {
  if (sampling) clearInterval(sampling);
  sampling = null;
  for (const track of stream?.getTracks() ?? []) {
    track.onended = null;
    track.stop();
  }
  stream = null;
  if (video) video.srcObject = null;
  video = null;
}

/** Why `getDisplayMedia` failed, as the main process is told: the system refused, or the window went. */
function failure(error: unknown): ScreenStarted {
  const name = error instanceof DOMException ? error.name : "";
  const message = error instanceof Error ? error.message : String(error);
  return { started: false, reason: name === "NotAllowedError" ? "denied" : "closed", message };
}

async function start(bridge: CaptureBridge): Promise<void> {
  stop();
  let captured: MediaStream;
  try {
    captured = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { max: MAX_FRAME_RATE } },
      audio: false,
    });
  } catch (error) {
    bridge.screenStarted(failure(error));
    return;
  }
  const [track] = captured.getVideoTracks();
  const surface = track?.getSettings().displaySurface;
  if (!track || surface === "monitor") {
    for (const each of captured.getTracks()) each.stop();
    bridge.screenStarted({
      started: false,
      reason: "whole-screen",
      message: "A whole screen was chosen.",
    });
    return;
  }
  const playing = document.createElement("video");
  playing.muted = true;
  playing.srcObject = captured;
  await new Promise<void>((resolve) => {
    playing.onloadedmetadata = () => resolve();
  });
  await playing.play();
  stream = captured;
  video = playing;
  track.onended = () => {
    stop();
    bridge.screenEnded();
  };
  bridge.screenStarted({ started: true });
  const sample = () => {
    if (!video || video.videoWidth === 0) return;
    const { width, height } = scaled(Number.POSITIVE_INFINITY, SCREEN_THUMBNAIL_WIDTH);
    bridge.screenSample(width, height, grey(draw(width, height)));
  };
  sample();
  sampling = setInterval(sample, SAMPLE_MS);
}

async function grab(bridge: CaptureBridge, id: number): Promise<void> {
  try {
    if (!video || video.videoWidth === 0)
      throw new Error("The shared window isn't being captured.");
    const read = scaled(MAX_READ_SIDE);
    const pixels = grey(draw(read.width, read.height));
    const shown = scaled(Number.POSITIVE_INFINITY, SHOWN_WIDTH);
    const blob = await draw(shown.width, shown.height).convertToBlob({
      type: "image/jpeg",
      quality: SHOWN_QUALITY,
    });
    const image = new Uint8Array(await blob.arrayBuffer());
    bridge.screenFrame(id, { width: read.width, height: read.height, pixels, image });
  } catch (error) {
    bridge.screenFrame(id, { error: error instanceof Error ? error.message : String(error) });
  }
}

/** Answers the main process's requests about the shared window. */
export function serveScreen(bridge: CaptureBridge): void {
  bridge.onScreenStart(() => {
    start(bridge).catch((error: unknown) => {
      stop();
      bridge.screenStarted(failure(error));
    });
  });
  bridge.onScreenGrab((id) => void grab(bridge, id));
  bridge.onScreenStop(stop);
}
