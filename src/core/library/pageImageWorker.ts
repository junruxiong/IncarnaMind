import { parentPort, workerData } from "node:worker_threads";
import { renderPdfImages } from "./pdfImages";

const { file, contentHash } = workerData as { file: string; contentHash: string };
void renderPdfImages(file, contentHash).then(
  (images) => parentPort?.postMessage({ images }),
  (error) =>
    parentPort?.postMessage({ error: error instanceof Error ? error.message : String(error) }),
);
