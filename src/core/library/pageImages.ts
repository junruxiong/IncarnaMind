import startWorker from "./pageImageWorker?nodeWorker";
import type { DocumentPageImage } from "./pdfImages";

/** A bounded, cancellable preview job, kept off Electron's main thread. */
export function documentPageImages(
  file: string,
  contentHash: string,
  signal: AbortSignal,
): Promise<DocumentPageImage[]> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const worker = startWorker({
      name: "incarnamind-page-images",
      workerData: { file, contentHash },
    });
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
      reject(
        new Error("PDF page previews took too long. Turn off page images to classify its text."),
      );
    }, 60_000);
    signal.addEventListener("abort", abort, { once: true });
    worker.once("message", (result: { images: DocumentPageImage[]; error?: string }) => {
      stop();
      if (result.error) reject(new Error(result.error));
      else resolve(result.images);
    });
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
