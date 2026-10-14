/**
 * Keeps a web page as it looks (#756): the core's `PageLook`. Each page is
 * loaded once, alone, in a hidden window of a session held only in memory and
 * cleared after it: no Node and no preload, no cookies or storage kept, no
 * permissions, pop-ups or downloads, no navigation away from the page, and
 * nothing on this computer or its network but the page's own host. Once it has
 * loaded, or stopped waiting for it, the page is printed as the screen shows
 * it, and the window is destroyed: all of it within the time limit.
 */
import { BrowserWindow, type Session, session } from "electron";
import type { Logger, PageLook, WebPageLookFailure } from "../core";

/** Not "persist:": the session lives in memory only. */
const PARTITION = "page-look";
const LIMIT_MS = 30_000;
/** How long to wait for the page to finish loading before printing what has. */
const LOAD_MS = 20_000;
/** For what a page draws after it has loaded. */
const SETTLE_MS = 1_000;
const WIDTH = 1280;
const HEIGHT = 1600;
/** Inches, as printToPDF takes them: room for the date, title and address it prints. */
const MARGIN = 0.4;

const PRIVATE_HOST =
  /^(?:localhost|.+\.localhost|.+\.local|127(?:\.\d+){3}|10(?:\.\d+){3}|192\.168(?:\.\d+){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d+){2}|169\.254(?:\.\d+){2}|0\.0\.0\.0|\[::1?\]|\[f[cd][0-9a-f]{0,2}:.*\]|\[fe[89ab][0-9a-f]:.*\])$/i;

/**
 * What the page may load: the web, and data it makes itself. Never a file,
 * the app's own schemes, or a host on this computer or its network, unless
 * the page itself is there.
 */
function allowed(url: string, pageHost: string): boolean {
  const parsed = URL.parse(url);
  if (!parsed) return false;
  if (parsed.protocol === "data:" || parsed.protocol === "blob:") return true;
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return false;
  return parsed.hostname === pageHost || !PRIVATE_HOST.test(parsed.hostname);
}

const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

type Made = Awaited<ReturnType<PageLook["render"]>>;

const failed = (failure: WebPageLookFailure, message: string): Made => ({ failure, message });

function prepare(): Session {
  const lookSession = session.fromPartition(PARTITION, { cache: false });
  lookSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  lookSession.setPermissionCheckHandler(() => false);
  lookSession.setDevicePermissionHandler(() => false);
  lookSession.setSpellCheckerEnabled(false);
  lookSession.on("will-download", (event) => event.preventDefault());
  return lookSession;
}

export function createPageLook({
  log,
  limitMs = LIMIT_MS,
}: {
  log: Logger;
  /** Test builds shorten it to see a page that never loads closed. */
  limitMs?: number;
}): PageLook {
  let lookSession: Session | null = null;
  let previous: Promise<unknown> = Promise.resolve();

  async function render(address: string): Promise<Made> {
    const pageHost = URL.parse(address)?.hostname;
    if (!pageHost) return failed("unreachable", "Not a web address.");
    lookSession ??= prepare();
    const own = lookSession;
    own.webRequest.onBeforeRequest((details, callback) =>
      callback({ cancel: !allowed(details.url, pageHost) }),
    );
    const window = new BrowserWindow({
      show: false,
      width: WIDTH,
      height: HEIGHT,
      title: "IncarnaMind page look",
      webPreferences: {
        sandbox: true,
        contextIsolation: true,
        nodeIntegration: false,
        nodeIntegrationInSubFrames: false,
        nodeIntegrationInWorker: false,
        webviewTag: false,
        session: own,
        backgroundThrottling: false,
        spellcheck: false,
        disableDialogs: true,
      },
    });
    const contents = window.webContents;
    contents.setAudioMuted(true);
    contents.setWindowOpenHandler(() => ({ action: "deny" }));
    contents.on("will-navigate", (event) => event.preventDefault());
    contents.on("will-attach-webview", (event) => event.preventDefault());
    // Some sites turn away a browser that names itself Electron.
    contents.setUserAgent(contents.getUserAgent().replace(/\s(?:incarnamind|Electron)\/\S+/gi, ""));
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const made = (async (): Promise<Made> => {
        const loading = contents.loadURL(address).then(
          () => null,
          (error: unknown) => (error instanceof Error ? error : new Error(String(error))),
        );
        const loaded = await Promise.race([loading, wait(LOAD_MS).then(() => null)]);
        if (loaded) return failed("unreachable", loaded.message);
        await wait(SETTLE_MS);
        try {
          contents.debugger.attach("1.3");
          await contents.debugger.sendCommand("Emulation.setEmulatedMedia", { media: "screen" });
        } catch (error) {
          // Printed with the page's print styles instead, which may hide some of it.
          log.warn("webPages.lookScreenMedia", {
            message: error instanceof Error ? error.message : String(error),
          });
        }
        const pdf = await contents.printToPDF({
          printBackground: true,
          displayHeaderFooter: true,
          pageSize: {
            width: WIDTH / 96 + 2 * MARGIN,
            height: HEIGHT / 96 + 2 * MARGIN,
          },
          margins: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
        });
        return { pdf: new Uint8Array(pdf) };
      })();
      made.catch(() => undefined);
      const timedOut = new Promise<Made>((resolve) => {
        timer = setTimeout(
          () => resolve(failed("timed-out", `Not printed in ${limitMs} ms.`)),
          limitMs,
        );
      });
      return await Promise.race([made, timedOut]);
    } catch (error) {
      return failed("failed", error instanceof Error ? error.message : String(error));
    } finally {
      clearTimeout(timer);
      if (!window.isDestroyed()) window.destroy();
      await Promise.all([own.clearStorageData(), own.clearCache(), own.clearAuthCache()]).catch(
        (error: unknown) =>
          log.error("webPages.lookClear", {
            message: error instanceof Error ? error.message : String(error),
          }),
      );
    }
  }

  return {
    render(address) {
      // One at a time: the pages share the session, cleared between them.
      const next = previous.then(() => render(address));
      previous = next.catch(() => undefined);
      return next;
    },
  };
}
