/**
 * The core's `Pictures` (#436): each SVG is drawn on a canvas in a hidden
 * window whose session refuses every request but the picture itself, so
 * nothing reaches the network.
 */
import { BrowserWindow, type Session, session } from "electron";
import type { Logger, Pictures } from "../core";

const PICTURES_PARTITION = "pictures";

/** Far longer than a chart takes, so a page that hangs fails the export's picture instead. */
const PICTURE_TIMEOUT_MS = 30_000;

/** Kept as text: it runs in the page. */
const DRAW = `(svg) => new Promise((resolve, reject) => {
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    if (!context) return reject(new Error("The page has no canvas to draw on."));
    context.drawImage(image, 0, 0);
    resolve(canvas.toDataURL("image/png"));
  };
  image.onerror = () => reject(new Error("The picture's SVG couldn't be read."));
  image.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
})`;

function prepareSession(): Session {
  const pictures = session.fromPartition(PICTURES_PARTITION, { cache: false });
  pictures.webRequest.onBeforeRequest((details, callback) => {
    callback({ cancel: !details.url.startsWith("data:") });
  });
  pictures.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  pictures.setPermissionCheckHandler(() => false);
  return pictures;
}

export function createElectronPictures(log: Logger): Pictures {
  let picturesSession: Session | null = null;

  return {
    async png(svg) {
      picturesSession ??= prepareSession();
      const window = new BrowserWindow({
        show: false,
        width: 320,
        height: 240,
        title: "IncarnaMind picture",
        webPreferences: {
          contextIsolation: true,
          nodeIntegration: false,
          sandbox: true,
          session: picturesSession,
          backgroundThrottling: false,
          spellcheck: false,
        },
      });
      const contents = window.webContents;
      contents.setWindowOpenHandler(() => ({ action: "deny" }));
      contents.on("will-navigate", (event) => event.preventDefault());
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        const drawn = (async () => {
          await window.loadURL(
            "data:text/html;charset=utf-8,<!doctype html><title>picture</title>",
          );
          const url: unknown = await contents.executeJavaScript(
            `(${DRAW})(${JSON.stringify(svg)})`,
          );
          if (typeof url !== "string" || !url.startsWith("data:image/png;base64,")) {
            throw new Error("The page gave back no PNG.");
          }
          return new Uint8Array(Buffer.from(url.slice(url.indexOf(",") + 1), "base64"));
        })();
        drawn.catch(() => undefined);
        const timedOut = new Promise<never>((_resolve, reject) => {
          timer = setTimeout(
            () => reject(new Error("The picture took too long to draw.")),
            PICTURE_TIMEOUT_MS,
          );
        });
        return await Promise.race([drawn, timedOut]);
      } catch (error) {
        log.error("export.pictureFailed", {
          message: error instanceof Error ? error.message : String(error),
        });
        throw error;
      } finally {
        clearTimeout(timer);
        if (!window.isDestroyed()) window.destroy();
      }
    },
  };
}
