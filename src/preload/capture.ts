/**
 * The hidden capture window's preload (ADR-0015): its page gets start and
 * stop from the main process, and sends back audio and how each source is
 * doing, and the shared window's thumbnails and frames (ADR-0024). Nothing
 * else: no core, no files.
 */
import { contextBridge, ipcRenderer } from "electron";
import {
  CAPTURE_BRIDGE_KEY,
  CAPTURE_CHANNELS,
  type CaptureBridge,
  type CaptureStart,
} from "../shared/capture";

const bridge: CaptureBridge = {
  onStart: (listener) => {
    ipcRenderer.on(CAPTURE_CHANNELS.start, (_event, request: CaptureStart) => listener(request));
  },
  onStop: (listener) => {
    ipcRenderer.on(CAPTURE_CHANNELS.stop, () => listener());
  },
  audio: (source, samples) => ipcRenderer.send(CAPTURE_CHANNELS.audio, source, samples),
  state: (source, state) => ipcRenderer.send(CAPTURE_CHANNELS.state, source, state),
  stopped: () => ipcRenderer.send(CAPTURE_CHANNELS.stopped),
  onScreenStart: (listener) => {
    ipcRenderer.on(CAPTURE_CHANNELS.screenStart, () => listener());
  },
  onScreenGrab: (listener) => {
    ipcRenderer.on(CAPTURE_CHANNELS.screenGrab, (_event, id: number) => listener(id));
  },
  onScreenStop: (listener) => {
    ipcRenderer.on(CAPTURE_CHANNELS.screenStop, () => listener());
  },
  screenStarted: (started) => ipcRenderer.send(CAPTURE_CHANNELS.screenStarted, started),
  screenSample: (width, height, pixels) =>
    ipcRenderer.send(CAPTURE_CHANNELS.screenSample, width, height, pixels),
  screenFrame: (id, frame) => ipcRenderer.send(CAPTURE_CHANNELS.screenFrame, id, frame),
  screenEnded: () => ipcRenderer.send(CAPTURE_CHANNELS.screenEnded),
};

contextBridge.exposeInMainWorld(CAPTURE_BRIDGE_KEY, bridge);
