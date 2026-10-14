import { parentPort, workerData } from "node:worker_threads";
import { drawPdfPages, renderPdfImages } from "./pdfImages";

const { file, contentHash, pages } = workerData as {
  file: string;
  contentHash: string;
  /** A Citation's pages to draw (#133); without, previews for classification. */
  pages?: number[];
};
const work = pages
  ? drawPdfPages(file, contentHash, pages).then((page) => ({ page }))
  : renderPdfImages(file, contentHash).then((images) => ({ images }));
void work.then(
  (result) => parentPort?.postMessage(result),
  (error) =>
    parentPort?.postMessage({ error: error instanceof Error ? error.message : String(error) }),
);
