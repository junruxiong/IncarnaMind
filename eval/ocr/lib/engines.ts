/**
 * The OCR engines compared (eval/ocr/README.md), each reading a page's
 * picture file and giving its text in reading order, with the time it took.
 *
 * - PaddleOCR's models run with the app's own code (`createOnnxTextRecognizer`
 *   and `scannedUnit`), on onnxruntime-node, which the app already ships.
 * - macOS Vision runs through a small Swift helper (./../vision.swift), built
 *   once into the cache with `swiftc`. Its lines are put in reading order by
 *   the app's `readingOrder`, as the app would.
 * - Windows.Media.Ocr runs through PowerShell (./../windows-ocr.ps1), on Windows only.
 * - Tesseract is the `tesseract` command, with its own page layout, and the
 *   tessdata models downloaded into the cache. The app would ship
 *   tesseract.js, the same engine compiled to WebAssembly, which is slower.
 *
 * Every engine reads English, Simplified and Traditional Chinese at once, as
 * the app must: it doesn't know a scan's language before reading it.
 */
import { execFile } from "node:child_process";
import { mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { promisify } from "node:util";
import { createCanvas, loadImage } from "@napi-rs/canvas";
import type { EmbeddingModelSource, PagePicture, RecognizedLine } from "../../../src/core/adapters";
import { readingOrder, scannedUnit } from "../../../src/core/documents/scan/layout";
import { BUILT_IN_TEXT_RECOGNITION_MODEL } from "../../../src/core/documents/scan/model";
import { createOnnxTextRecognizer } from "../../../src/core/documents/scan/onnx";
import { downloadModel, isModelDownloaded } from "../../../src/core/embedding/download";
import type { Log } from "../../lib/log";

const run = promisify(execFile);

export interface Engine {
  id: string;
  name: string;
  /** What the app would ship or download for it, for the report. */
  size: string;
  licence: string;
  /** Null when it can run here, else why not. */
  unavailable(): Promise<string | null>;
  /** Prepares it (downloads, builds), once. */
  open(cacheDir: string, log: Log): Promise<void>;
  read(file: string): Promise<{ text: string; ms: number }>;
  close(): void;
}

/** A page picture file in shades of grey, as the app renders a page. */
export async function greyPicture(file: string): Promise<PagePicture> {
  const image = await loadImage(await readFile(file));
  const canvas = createCanvas(image.width, image.height);
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);
  const { data } = context.getImageData(0, 0, image.width, image.height);
  const pixels = new Uint8Array(image.width * image.height);
  for (let index = 0; index < pixels.length; index++) {
    pixels[index] = Math.round(
      0.299 * (data[index * 4] as number) +
        0.587 * (data[index * 4 + 1] as number) +
        0.114 * (data[index * 4 + 2] as number),
    );
  }
  return { width: image.width, height: image.height, pixels };
}

/** Downloads a model's files into the cache once, checked against their pinned hashes. */
async function ensureFiles(
  directory: string,
  source: EmbeddingModelSource,
  name: string,
  log: Log,
): Promise<void> {
  if (isModelDownloaded(directory, source.files)) return;
  log(`Downloading ${name}…`);
  await downloadModel({ directory, source, signal: new AbortController().signal, onProgress() {} });
}

const HF = "https://huggingface.co/";

/** A PaddleOCR detection and recognition pair, pinned. */
interface PaddleModels {
  id: string;
  name: string;
  folder: string;
  detector: { path: string; url: string; size: number; sha256: string };
  recognizer: { path: string; url: string; size: number; sha256: string };
}

const V5_MOBILE = BUILT_IN_TEXT_RECOGNITION_MODEL.source.files;
const v5Detector = V5_MOBILE[0] as PaddleModels["detector"];
const v5Recognizer = V5_MOBILE[1] as PaddleModels["recognizer"];
/** RapidOCR's ONNX conversions of PP-OCRv4, on Hugging Face (the same bytes as its ModelScope release). */
const RAPIDOCR_V4 = "SWHL/RapidOCR/resolve/1cfba2e90fc938db55889873735088de210cc173/PP-OCRv4/";

export const PADDLE_MODELS: readonly PaddleModels[] = [
  {
    id: "ppocr-v5-mobile",
    name: "PP-OCRv5 mobile",
    folder: BUILT_IN_TEXT_RECOGNITION_MODEL.folder,
    detector: v5Detector,
    recognizer: v5Recognizer,
  },
  {
    id: "ppocr-v4-mobile",
    name: "PP-OCRv4 mobile",
    folder: "pp-ocrv4-mobile",
    detector: {
      path: "PP-OCRv4_mobile_det.onnx",
      url: `${RAPIDOCR_V4}ch_PP-OCRv4_det_infer.onnx`,
      size: 4_745_517,
      sha256: "d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9",
    },
    recognizer: {
      path: "PP-OCRv4_mobile_rec.onnx",
      url: `${RAPIDOCR_V4}ch_PP-OCRv4_rec_infer.onnx`,
      size: 10_857_958,
      sha256: "48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b",
    },
  },
  {
    id: "ppocr-v5-server",
    name: "PP-OCRv5 server",
    folder: "pp-ocrv5-server",
    detector: {
      path: "PP-OCRv5_server_det.onnx",
      url: "breezedeus/cnstd-ppocr-ch_PP-OCRv5_det_server/resolve/5902daf0bdba2e7809cc760f555abcf4f233eeee/ch_PP-OCRv5_det_server_infer.onnx",
      size: 88_118_768,
      sha256: "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad",
    },
    recognizer: {
      path: "PP-OCRv5_server_rec.onnx",
      url: "breezedeus/cnocr-ppocr-ch_PP-OCRv5_server/resolve/cc74722f9daf71db351161c74cbc0e6bc2396c3c/ch_PP-OCRv5_server_rec_infer.onnx",
      size: 84_577_022,
      sha256: "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
    },
  },
];

const megabytes = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

function paddleEngine(models: PaddleModels): Engine {
  const recognizer = createOnnxTextRecognizer();
  let directory = "";
  return {
    id: models.id,
    name: models.name,
    size: `${megabytes(models.detector.size + models.recognizer.size)} download; onnxruntime-node already ships`,
    licence: "Apache-2.0",
    unavailable: async () => null,
    async open(cacheDir, log) {
      directory = join(cacheDir, "models", models.folder);
      await ensureFiles(
        directory,
        { baseUrl: HF, files: [models.detector, models.recognizer] },
        models.name,
        log,
      );
      await recognizer.load({
        detector: join(directory, models.detector.path),
        recognizer: join(directory, models.recognizer.path),
      });
    },
    async read(file) {
      const picture = await greyPicture(file);
      const started = performance.now();
      const lines = await recognizer.read(picture, 1);
      const ms = performance.now() - started;
      return { text: scannedUnit(1, lines, picture, models.name).text, ms };
    },
    close: () => recognizer.close(),
  };
}

/** Vision's line: its text and its box, normalised, origin at the bottom left. */
interface VisionLine {
  text: string;
  confidence: number;
  box: [number, number, number, number];
}

function visionEngine(root: string): Engine {
  let helper = "";
  return {
    id: "vision",
    name: "macOS Vision (accurate)",
    size: "a signed Swift helper of about 100 KB; macOS 10.15 or later, macOS only",
    licence: "Apple's system framework",
    unavailable: async () => (process.platform === "darwin" ? null : "macOS only"),
    async open(cacheDir) {
      await mkdir(join(cacheDir, "bin"), { recursive: true });
      helper = join(cacheDir, "bin", "vision-ocr");
      await run("swiftc", ["-O", join(root, "eval/ocr/vision.swift"), "-o", helper]);
    },
    async read(file) {
      const { stdout } = await run(helper, [file], { maxBuffer: 64 * 1024 * 1024 });
      const page = JSON.parse(stdout) as { milliseconds: number; lines: VisionLine[] };
      const picture = await loadImage(await readFile(file));
      const { width, height } = picture;
      const lines: RecognizedLine[] = page.lines.map((line) => {
        const [left, bottom, right, top] = line.box;
        const x0 = left * width;
        const x1 = right * width;
        const y0 = (1 - top) * height;
        const y1 = (1 - bottom) * height;
        return {
          text: line.text,
          quad: [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
          ],
          confidence: line.confidence,
          characters: [],
        };
      });
      const text = readingOrder(lines)
        .map((row) => row.map((line) => line.text).join(" "))
        .join("\n");
      return { text, ms: page.milliseconds };
    },
    close() {},
  };
}

function windowsEngine(root: string): Engine {
  return {
    id: "windows",
    name: "Windows.Media.Ocr",
    size: "a PowerShell or .NET helper; Windows 10 or later, with each language's OCR pack installed",
    licence: "Microsoft's system API",
    unavailable: async () => (process.platform === "win32" ? null : "Windows only"),
    async open() {},
    async read(file) {
      const started = performance.now();
      const { stdout } = await run(
        "powershell.exe",
        [
          "-NoProfile",
          "-ExecutionPolicy",
          "Bypass",
          "-File",
          join(root, "eval/ocr/windows-ocr.ps1"),
          file,
        ],
        { maxBuffer: 64 * 1024 * 1024 },
      );
      return { text: stdout.trim(), ms: performance.now() - started };
    },
    close() {},
  };
}

/** Tesseract's models, pinned to a commit of tessdata_fast or tessdata_best. */
const TESSDATA = {
  fast: {
    commit: "87416418657359cb625c412a48b6e1d6d41c29bd",
    files: [
      {
        path: "eng.traineddata",
        size: 4_113_088,
        sha256: "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
      },
      {
        path: "chi_sim.traineddata",
        size: 2_469_156,
        sha256: "a5fcb6f0db1e1d6d8522f39db4e848f05984669172e584e8d76b6b3141e1f730",
      },
      {
        path: "chi_tra.traineddata",
        size: 2_366_642,
        sha256: "529c5b5797d64b126065cd55f2bb4c7fd7b15790798091b1ff259941a829330b",
      },
    ],
  },
  best: {
    commit: "e12c65a915945e4c28e237a9b52bc4a8f39a0cec",
    files: [
      {
        path: "eng.traineddata",
        size: 15_400_601,
        sha256: "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba",
      },
      {
        path: "chi_sim.traineddata",
        size: 13_077_423,
        sha256: "4fef2d1306c8e87616d4d3e4c6c67faf5d44be3342290cf8f2f0f6e3aa7e735b",
      },
      {
        path: "chi_tra.traineddata",
        size: 12_985_735,
        sha256: "1aa60488574cafa69486d919284f079ca9b68fcc7f6ad8dc1ff1b318dfd97028",
      },
    ],
  },
} as const;

function tesseractEngine(kind: keyof typeof TESSDATA): Engine {
  const data = TESSDATA[kind];
  let directory = "";
  const total = data.files.reduce((sum, file) => sum + file.size, 0);
  return {
    id: `tesseract-${kind}`,
    name: `Tesseract 5 (tessdata_${kind})`,
    size: `${megabytes(total)} of models; tesseract.js adds about 4 MB of WebAssembly`,
    licence: "Apache-2.0",
    async unavailable() {
      try {
        await run("tesseract", ["--version"]);
        return null;
      } catch {
        return "the tesseract command isn't installed";
      }
    },
    async open(cacheDir, log) {
      directory = join(cacheDir, "tessdata", kind);
      await ensureFiles(
        directory,
        {
          baseUrl: `https://raw.githubusercontent.com/tesseract-ocr/tessdata_${kind}/${data.commit}/`,
          files: [...data.files],
        },
        `tessdata_${kind}`,
        log,
      );
    },
    async read(file) {
      const started = performance.now();
      const { stdout } = await run(
        "tesseract",
        [file, "-", "--tessdata-dir", directory, "-l", "eng+chi_sim+chi_tra", "--psm", "3"],
        { maxBuffer: 64 * 1024 * 1024 },
      );
      return { text: stdout.trim(), ms: performance.now() - started };
    },
    close() {},
  };
}

export function engines(root: string): Engine[] {
  return [
    ...PADDLE_MODELS.map(paddleEngine),
    visionEngine(root),
    windowsEngine(root),
    tesseractEngine("fast"),
    tesseractEngine("best"),
  ];
}
