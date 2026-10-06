/**
 * pdf.js in the renderer, loaded on first use so the app starts without it.
 *
 * Its worker is a separate file Vite copies into the build (`?url`), so parsing
 * and rendering commands run off the main thread. Its data files (character
 * maps for CJK fonts, standard fonts, image decoders) are served under
 * `pdfjs/` next to the page; see `pdfjsData` in electron.vite.config.ts.
 */
import type { PDFDocumentLoadingTask } from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { documentFileUrl } from "../../../shared/documentViewer";

export type PdfJs = typeof import("pdfjs-dist");

let loading: Promise<PdfJs> | undefined;

export function loadPdfJs(): Promise<PdfJs> {
  loading ??= import("pdfjs-dist").then((pdfjs) => {
    pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
    return pdfjs;
  });
  return loading;
}

const dataUrl = (folder: string) => new URL(`pdfjs/${folder}/`, document.baseURI).href;

/** Starts loading a Document's PDF, streamed from the main process (never over IPC). */
export function openPdf(pdfjs: PdfJs, documentId: string): PDFDocumentLoadingTask {
  return pdfjs.getDocument({
    url: documentFileUrl(documentId),
    cMapUrl: dataUrl("cmaps"),
    cMapPacked: true,
    standardFontDataUrl: dataUrl("standard_fonts"),
    wasmUrl: dataUrl("wasm"),
    enableXfa: false,
  });
}

/** True if loading failed because the main process has no such Document (deleted, or its file is gone). */
export function isMissingDocumentError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "status" in error &&
    (error as { status: unknown }).status === 404
  );
}
