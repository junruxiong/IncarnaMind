/**
 * A page loaded out of sight for work of the app's own: printing a PDF
 * export, drawing a picture, keeping a web page as it looks. It is a
 * `WebContentsView` in a `BaseWindow`, never a `BrowserWindow`, so it is
 * never one of the app's windows, which hear the core's events and come to
 * the front. It is sandboxed, with no Node and no preload, in the session
 * given; pop-ups, web views and navigation away from the page are refused,
 * and it makes no sound. It is destroyed when the work is done or its time
 * is up.
 */
import { BaseWindow, type Session, type WebContents, WebContentsView } from "electron";

export interface HiddenPageOptions<T> {
  session: Session;
  width: number;
  height: number;
  limitMs: number;
  /** What the work comes to when its time is up: a value, or an error to throw. */
  onLimit(): T | Error;
}

export async function onHiddenPage<T>(
  options: HiddenPageOptions<T>,
  work: (contents: WebContents) => Promise<T>,
): Promise<T> {
  const window = new BaseWindow({ show: false, width: options.width, height: options.height });
  const view = new WebContentsView({
    webPreferences: {
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      nodeIntegrationInSubFrames: false,
      nodeIntegrationInWorker: false,
      webviewTag: false,
      session: options.session,
      backgroundThrottling: false,
      spellcheck: false,
      disableDialogs: true,
    },
  });
  window.contentView.addChildView(view);
  view.setBounds({ x: 0, y: 0, width: options.width, height: options.height });
  const contents = view.webContents;
  contents.setAudioMuted(true);
  contents.setWindowOpenHandler(() => ({ action: "deny" }));
  contents.on("will-navigate", (event) => event.preventDefault());
  contents.on("will-attach-webview", (event) => event.preventDefault());
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const done = work(contents);
    done.catch(() => undefined);
    const timedOut = new Promise<T>((resolve, reject) => {
      timer = setTimeout(() => {
        const outcome = options.onLimit();
        if (outcome instanceof Error) reject(outcome);
        else resolve(outcome);
      }, options.limitMs);
    });
    return await Promise.race([done, timedOut]);
  } finally {
    clearTimeout(timer);
    if (!contents.isDestroyed()) contents.close();
    if (!window.isDestroyed()) window.destroy();
  }
}
