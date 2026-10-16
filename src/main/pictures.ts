/**
 * The core's `Pictures` (#436): each SVG is drawn on a canvas in a hidden
 * window whose session refuses every request but the picture itself, so
 * nothing reaches the network.
 */
import { type Session, session } from "electron";
import type { Logger, Pictures } from "../core";
import { onHiddenPage } from "./hiddenPage";

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
      try {
        return await onHiddenPage(
          {
            session: picturesSession,
            width: 320,
            height: 240,
            limitMs: PICTURE_TIMEOUT_MS,
            onLimit: () => new Error("The picture took too long to draw."),
          },
          async (contents) => {
            await contents.loadURL(
              "data:text/html;charset=utf-8,<!doctype html><title>picture</title>",
            );
            const url: unknown = await contents.executeJavaScript(
              `(${DRAW})(${JSON.stringify(svg)})`,
            );
            if (typeof url !== "string" || !url.startsWith("data:image/png;base64,")) {
              throw new Error("The page gave back no PNG.");
            }
            return new Uint8Array(Buffer.from(url.slice(url.indexOf(",") + 1), "base64"));
          },
        );
      } catch (error) {
        log.error("export.pictureFailed", {
          message: error instanceof Error ? error.message : String(error),
        });
        throw error;
      }
    },
  };
}
