/**
 * The desktop app's `ScreenCapture` (ADR-0024): the one window the User
 * shares during a Meeting, chosen in IncarnaMind's own picker from the open
 * windows (`desktopCapturer`, never IncarnaMind's own), and captured as video
 * only in the meeting capture window (./captureWindow), which sends a small
 * grey thumbnail about once a second and, when the core asks, the window at
 * full size. On macOS this needs the Screen Recording permission, which macOS
 * asks for the first time the windows are listed.
 *
 * Test builds can share a test page instead (INCARNAMIND_TEST_SHARED_WINDOW):
 * a hidden window showing that HTML file is the one window the picker lists,
 * captured as a tab is, so no screen is ever captured in tests.
 */
import {
  BrowserWindow,
  desktopCapturer,
  type IpcMainEvent,
  ipcMain,
  systemPreferences,
  type WebContents,
} from "electron";
import {
  type Logger,
  type PagePicture,
  type ScreenCapture,
  ScreenCaptureError,
  type ScreenFrame,
  type ScreenPermission,
  type ScreenSession,
  type ScreenSink,
} from "../../core";
import type { ShareableWindow } from "../../core/api";
import { CAPTURE_CHANNELS, type ScreenGrabbed, type ScreenStarted } from "../../shared/capture";
import { type CaptureWindows, leaveOut, type ScreenChoice } from "./captureWindow";

/** How long a frame asked for may take before reading it gives up. */
const GRAB_TIMEOUT_MS = 10_000;
/** The pictures IncarnaMind's picker shows of each window. */
const THUMBNAIL = { width: 320, height: 200 };
/** The test page's id in IncarnaMind's picker. */
const TEST_WINDOW = "test-window";

/** Whether IncarnaMind may capture a window: macOS's Screen Recording; nothing to ask elsewhere. */
function screenPermission(): ScreenPermission {
  if (process.platform !== "darwin") return "granted";
  const status = systemPreferences.getMediaAccessStatus("screen");
  if (status === "granted") return "granted";
  return status === "not-determined" ? "not-asked" : "denied";
}

/** Bytes from the capture window, whatever view IPC delivered them in. */
function bytesOf(value: unknown): Uint8Array | null {
  if (value instanceof Uint8Array) return value;
  if (!ArrayBuffer.isView(value)) return null;
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

/** A picture from the capture window, checked. */
function pictureOf(width: unknown, height: unknown, pixels: unknown): PagePicture | null {
  const bytes = bytesOf(pixels);
  if (typeof width !== "number" || typeof height !== "number" || !bytes) return null;
  return bytes.length === width * height ? { width, height, pixels: bytes } : null;
}

export interface ElectronScreenCaptureOptions {
  windows: CaptureWindows;
  log: Logger;
  /** A test build's page to share instead of a window (INCARNAMIND_TEST_SHARED_WINDOW). */
  testPage?: string;
}

export function createElectronScreenCapture({
  windows,
  log,
  testPage,
}: ElectronScreenCaptureOptions): ScreenCapture {
  /** The windows IncarnaMind's picker listed last, by id: their names. */
  let listed = new Map<string, string>();
  let testWindow: BrowserWindow | null = null;

  /** The test page's hidden window, opened the first time it is asked for. */
  async function openTestWindow(page: string): Promise<BrowserWindow> {
    if (testWindow && !testWindow.isDestroyed()) return testWindow;
    const window = new BrowserWindow({
      show: false,
      width: 1280,
      height: 720,
      webPreferences: { backgroundThrottling: false, sandbox: true, contextIsolation: true },
    });
    leaveOut(window);
    await window.loadFile(page);
    testWindow = window;
    // The tests change what it shows through this.
    (globalThis as { incarnamindTestScreen?: WebContents }).incarnamindTestScreen =
      window.webContents;
    return window;
  }

  async function listWindows(): Promise<ShareableWindow[]> {
    if (testPage) {
      const window = await openTestWindow(testPage);
      const picture = await window.webContents.capturePage();
      listed = new Map([[TEST_WINDOW, window.getTitle()]]);
      return [{ id: TEST_WINDOW, name: window.getTitle(), thumbnail: picture.toDataURL() }];
    }
    if (screenPermission() === "denied") {
      throw new ScreenCaptureError("denied", "IncarnaMind may not record the screen.");
    }
    const own = new Set(BrowserWindow.getAllWindows().map((window) => window.getMediaSourceId()));
    // On macOS the first listing asks for Screen Recording; without it, the windows can't be read.
    const sources = await desktopCapturer.getSources({
      types: ["window"],
      thumbnailSize: THUMBNAIL,
      fetchWindowIcons: false,
    });
    if (screenPermission() !== "granted") {
      throw new ScreenCaptureError("denied", "IncarnaMind may not record the screen.");
    }
    const shown = sources.filter((source) => !own.has(source.id) && source.name.trim() !== "");
    listed = new Map(shown.map((source) => [source.id, source.name]));
    return shown.map((source) => ({
      id: source.id,
      name: source.name,
      thumbnail: source.thumbnail.isEmpty() ? "" : source.thumbnail.toDataURL(),
    }));
  }

  async function choiceFor(windowId: string): Promise<ScreenChoice> {
    if (testPage && windowId === TEST_WINDOW) {
      return { kind: "page", frame: (await openTestWindow(testPage)).webContents.mainFrame };
    }
    return { kind: "window", source: { id: windowId, name: listed.get(windowId) ?? "" } };
  }

  return {
    permission: () => (testPage ? "granted" : screenPermission()),

    windows: listWindows,

    async start(windowId: string, sink: ScreenSink): Promise<ScreenSession> {
      const choice = await choiceFor(windowId);
      const use = windows.use();
      const contents = use.contents;
      const fromWindow = (event: IpcMainEvent) => event.sender === contents;
      const grabs = new Map<
        number,
        { resolve(frame: ScreenFrame): void; reject(error: Error): void }
      >();
      let nextGrab = 1;
      let ended = false;
      let answer: (started: ScreenStarted) => void = () => {};
      const started = new Promise<ScreenStarted>((resolve) => {
        answer = resolve;
      });

      const onStarted = (event: IpcMainEvent, value: ScreenStarted) => {
        if (fromWindow(event)) answer(value);
      };
      const onSample = (event: IpcMainEvent, width: unknown, height: unknown, pixels: unknown) => {
        if (!fromWindow(event) || ended) return;
        const thumbnail = pictureOf(width, height, pixels);
        if (thumbnail) sink.sample(thumbnail);
      };
      const onFrame = (event: IpcMainEvent, id: unknown, frame: ScreenGrabbed) => {
        if (!fromWindow(event) || typeof id !== "number") return;
        const waiting = grabs.get(id);
        grabs.delete(id);
        if (!waiting) return;
        if ("error" in frame) {
          waiting.reject(new Error(frame.error));
          return;
        }
        const picture = pictureOf(frame.width, frame.height, frame.pixels);
        const image = bytesOf(frame.image);
        if (!picture || !image)
          waiting.reject(new Error("The shared window's frame couldn't be read."));
        else waiting.resolve({ picture, image });
      };
      const finish = () => {
        if (ended) return;
        ended = true;
        ipcMain.off(CAPTURE_CHANNELS.screenStarted, onStarted);
        ipcMain.off(CAPTURE_CHANNELS.screenSample, onSample);
        ipcMain.off(CAPTURE_CHANNELS.screenFrame, onFrame);
        ipcMain.off(CAPTURE_CHANNELS.screenEnded, onEnded);
        for (const waiting of grabs.values()) {
          waiting.reject(new Error("The shared window is no longer captured."));
        }
        grabs.clear();
        use.release();
      };
      function onEnded(event: IpcMainEvent) {
        if (!fromWindow(event) || ended) return;
        finish();
        sink.ended();
      }
      ipcMain.on(CAPTURE_CHANNELS.screenStarted, onStarted);
      ipcMain.on(CAPTURE_CHANNELS.screenSample, onSample);
      ipcMain.on(CAPTURE_CHANNELS.screenFrame, onFrame);
      ipcMain.on(CAPTURE_CHANNELS.screenEnded, onEnded);
      use.onGone(() => {
        answer({ started: false, reason: "closed", message: "The capture window stopped." });
        if (ended) return;
        finish();
        sink.ended();
      });

      let result: ScreenStarted;
      try {
        await use.loaded;
        windows.chooseScreen(choice);
        contents.send(CAPTURE_CHANNELS.screenStart);
        result = await started;
      } catch (error) {
        result = {
          started: false,
          reason: "closed",
          message: error instanceof Error ? error.message : String(error),
        };
      }
      if (!result.started) {
        finish();
        log.warn("meeting.screenNotStarted", { reason: result.reason, kind: choice.kind });
        throw new ScreenCaptureError(result.reason, result.message);
      }
      const name = choice.kind === "window" ? choice.source.name : (testWindow?.getTitle() ?? "");

      return {
        name,
        grab() {
          if (ended || contents.isDestroyed()) {
            return Promise.reject(new Error("The shared window is no longer captured."));
          }
          const id = nextGrab++;
          return new Promise<ScreenFrame>((resolve, reject) => {
            const timer = setTimeout(() => {
              grabs.delete(id);
              reject(new Error("The shared window's frame didn't come."));
            }, GRAB_TIMEOUT_MS);
            grabs.set(id, {
              resolve: (frame) => {
                clearTimeout(timer);
                resolve(frame);
              },
              reject: (error) => {
                clearTimeout(timer);
                reject(error);
              },
            });
            contents.send(CAPTURE_CHANNELS.screenGrab, id);
          });
        },
        stop() {
          if (ended) return;
          if (!contents.isDestroyed()) contents.send(CAPTURE_CHANNELS.screenStop);
          finish();
        },
      };
    },
  };
}
