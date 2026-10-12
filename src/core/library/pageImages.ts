import type { DrawnPage } from "../../shared/citedPage";
import startWorker from "./pageImageWorker?nodeWorker";
import type { DocumentPageImage } from "./pdfImages";

/** Runs one job on a disposable worker thread, off Electron's main thread, within a minute. */
function onWorker<T>(
  workerData: { file: string; contentHash: string; pages?: number[] },
  signal: AbortSignal,
  take: (result: { images?: DocumentPageImage[]; page?: DrawnPage }) => T | undefined,
  timeout: string,
): Promise<T> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const worker = startWorker({ name: "incarnamind-page-images", workerData });
    const stop = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      void worker.terminate();
    };
    const abort = () => {
      stop();
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      stop();
      reject(new Error(timeout));
    }, 60_000);
    signal.addEventListener("abort", abort, { once: true });
    worker.once(
      "message",
      (result: { images?: DocumentPageImage[]; page?: DrawnPage; error?: string }) => {
        stop();
        const value = result.error ? undefined : take(result);
        if (value === undefined) reject(new Error(result.error ?? "The worker gave nothing back."));
        else resolve(value);
      },
    );
    worker.once("error", (error) => {
      stop();
      reject(error);
    });
    worker.once("exit", () => {
      stop();
      reject(new Error("PDF page preview worker stopped."));
    });
  });
}

/** A bounded, cancellable preview job, kept off Electron's main thread. */
export function documentPageImages(
  file: string,
  contentHash: string,
  signal: AbortSignal,
): Promise<DocumentPageImage[]> {
  return onWorker(
    { file, contentHash },
    signal,
    (result) => result.images,
    "PDF page previews took too long. Turn off page images to classify its text.",
  );
}

/** A Citation's cited pages of a PDF, drawn as one picture to save (#133); cancellable. */
export function documentCitedPages(
  file: string,
  contentHash: string,
  pages: readonly number[],
  signal: AbortSignal,
): Promise<DrawnPage> {
  return onWorker(
    { file, contentHash, pages: [...pages] },
    signal,
    (result) => result.page,
    "Drawing the cited PDF pages took too long.",
  );
}
