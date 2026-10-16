/**
 * Prints PDF exports (#100): the core's `Printer`. Each print loads the print page
 * (src/renderer/print.html) in a hidden window, hands it the document to lay out in pages, and
 * prints it with Chromium. The window's session refuses every request but the app's own files
 * (and the dev server's, in development), so a Mind's links and pictures never reach the
 * network, and it asks for nothing.
 */
import { join } from "node:path";
import { type Session, session } from "electron";
import type { Logger, Printer } from "../core";
import { PRINT_ENTRY } from "../shared/printDocument";
import { onHiddenPage } from "./hiddenPage";

/** An in-memory session of its own: nothing it loads is kept, or shared with the app's window. */
const PRINT_PARTITION = "print";

/** Longer than any Mind takes to lay out, so a page that hangs fails the export instead. */
const PRINT_TIMEOUT_MS = 120_000;

export interface PrinterOptions {
  log: Logger;
  /** The dev server's address in development; the built page otherwise. */
  rendererUrl: string | undefined;
}

function prepareSession(rendererUrl: string | undefined): Session {
  const printSession = session.fromPartition(PRINT_PARTITION, { cache: false });
  const devOrigin = rendererUrl ? new URL(rendererUrl).origin : null;
  printSession.webRequest.onBeforeRequest((details, callback) => {
    const url = new URL(details.url);
    const allowed =
      url.protocol === "file:" ||
      url.protocol === "data:" ||
      url.protocol === "devtools:" ||
      (devOrigin !== null && url.origin === devOrigin);
    callback({ cancel: !allowed });
  });
  printSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  printSession.setPermissionCheckHandler(() => false);
  return printSession;
}

export function createElectronPrinter({ log, rendererUrl }: PrinterOptions): Printer {
  let printSession: Session | null = null;

  return {
    async print(document) {
      printSession ??= prepareSession(rendererUrl);
      try {
        const pdf = await onHiddenPage(
          {
            session: printSession,
            width: 900,
            height: 1200,
            limitMs: PRINT_TIMEOUT_MS,
            onLimit: () => new Error("The PDF took too long to lay out."),
          },
          async (contents) => {
            if (rendererUrl) await contents.loadURL(new URL("print.html", `${rendererUrl}/`).href);
            else await contents.loadFile(join(__dirname, "../renderer/print.html"));
            await contents.executeJavaScript(`window.${PRINT_ENTRY}(${JSON.stringify(document)})`);
            return contents.printToPDF({
              preferCSSPageSize: true,
              printBackground: true,
              margins: { top: 0, bottom: 0, left: 0, right: 0 },
              generateTaggedPDF: true,
              generateDocumentOutline: true,
            });
          },
        );
        return new Uint8Array(pdf.buffer, pdf.byteOffset, pdf.byteLength);
      } catch (error) {
        log.error("export.printFailed", {
          message: error instanceof Error ? error.message : String(error),
        });
        throw error;
      }
    },
  };
}
