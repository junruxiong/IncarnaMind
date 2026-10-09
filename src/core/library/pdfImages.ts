import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import type { PDFPageProxy } from "pdfjs-dist/types/src/display/api";

export interface DocumentPageImage {
  page: number;
  /** Bare base64, as Ollama's System One endpoint requires. */
  data: string;
}

export const PDF_IMAGE_LIMITS = {
  pages: 3,
  scanPages: 12,
  edge: 1600,
  fileBytes: 100 * 1024 * 1024,
  // Accommodates a 300 dpi A3 scan (~17.4 MP), with a bounded raster allocation.
  imagePixels: 20_000_000,
};

type Canvas = NonNullable<Parameters<PDFPageProxy["render"]>[0]["canvas"]> & {
  toBuffer(mime: "image/jpeg", quality: number): Buffer;
};
type Surface = { canvas: Canvas };

/** Runs in a disposable worker; no previews are written to disk. */
export async function renderPdfImages(
  file: string,
  contentHash: string,
): Promise<DocumentPageImage[]> {
  if ((await stat(file)).size > PDF_IMAGE_LIMITS.fileBytes)
    throw new Error(
      "PDF page previews support files up to 100 MB. Turn off page images to classify its text.",
    );
  const bytes = await readFile(file);
  if (bytes.length > PDF_IMAGE_LIMITS.fileBytes)
    throw new Error("The PDF is too large for page previews.");
  if (createHash("sha256").update(bytes).digest("hex") !== contentHash)
    throw new Error(
      "The Document changed. Wait for it to finish processing, then classify it again.",
    );
  const { getDocument, OPS, PDFWorker } = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const packageDir = dirname(createRequire(import.meta.url).resolve("pdfjs-dist/package.json"));
  const worker = new PDFWorker({ verbosity: 0 });
  let task: ReturnType<typeof getDocument> | undefined;
  let workerError: Error | undefined;
  // PDF.js 6.4 can resolve a partial operator list before rejecting its stream error.
  // Observe the dedicated worker's error envelope so a dropped scan cannot be sent
  // as a blank preview. Keep the oversized first/second-page regressions on upgrades.
  const onMessage = (event: { data: { reason?: { message?: unknown } } }) => {
    if (typeof event.data.reason?.message === "string")
      workerError ??= new Error(event.data.reason.message);
  };
  const checkWorker = () => {
    if (workerError) throw workerError;
  };
  try {
    await worker.promise;
    worker.port.addEventListener("message", onMessage);
    task = getDocument({
      worker,
      data: new Uint8Array(bytes),
      cMapUrl: `${join(packageDir, "cmaps")}/`,
      cMapPacked: true,
      standardFontDataUrl: `${join(packageDir, "standard_fonts")}/`,
      wasmUrl: `${join(packageDir, "wasm")}/`,
      useSystemFonts: false,
      maxImageSize: PDF_IMAGE_LIMITS.imagePixels,
      // PDF.js otherwise silently drops oversized images, turning scans into blank previews.
      stopAtErrors: true,
      verbosity: 0,
    });
    const document = await task.promise;
    checkWorker();
    const factory = document.canvasFactory as {
      create(width: number, height: number): Surface;
      destroy(surface: Surface): void;
    };
    // Always include the opening page; favour illustrations/charts in the next eleven.
    const candidates: { page: number; score: number }[] = [];
    for (
      let number = 2;
      number <= Math.min(document.numPages, PDF_IMAGE_LIMITS.scanPages);
      number++
    ) {
      const page = await document.getPage(number);
      try {
        const operators = await page.getOperatorList();
        checkWorker();
        const score = operators.fnArray.reduce(
          (sum, op) =>
            sum +
            (op === OPS.paintImageXObject ||
            op === OPS.paintInlineImageXObject ||
            op === OPS.paintImageMaskXObject
              ? 10
              : op === OPS.constructPath
                ? 1
                : 0),
          0,
        );
        candidates.push({ page: number, score });
      } finally {
        page.cleanup();
      }
    }
    const selected = [
      1,
      ...candidates
        .sort((a, b) => b.score - a.score || a.page - b.page)
        .slice(0, PDF_IMAGE_LIMITS.pages - 1)
        .map(({ page }) => page),
    ].sort((a, b) => a - b);
    const images: DocumentPageImage[] = [];
    for (const number of selected) {
      const page = await document.getPage(number);
      const original = page.getViewport({ scale: 1 });
      const viewport = page.getViewport({
        scale: PDF_IMAGE_LIMITS.edge / Math.max(original.width, original.height),
      });
      const surface = factory.create(Math.ceil(viewport.width), Math.ceil(viewport.height));
      try {
        await page.render({ canvas: surface.canvas, viewport, background: "white" }).promise;
        checkWorker();
        const jpeg = surface.canvas.toBuffer("image/jpeg", 80);
        if (jpeg.length > 2 * 1024 * 1024)
          throw new Error(
            "A PDF page preview is too large. Turn off page images to classify its text.",
          );
        images.push({ page: number, data: jpeg.toString("base64") });
      } finally {
        factory.destroy(surface);
        page.cleanup();
      }
    }
    checkWorker();
    return images;
  } finally {
    worker.port?.removeEventListener("message", onMessage);
    try {
      await task?.destroy();
    } finally {
      worker.destroy();
    }
  }
}
