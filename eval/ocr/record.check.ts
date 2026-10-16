/**
 * `INCARNAMIND_OCR_RECORD=1 npm run eval:ocr`: reads the scanned-page set
 * with the built-in model and writes what it read, line by line with each
 * line's box, to tests/fixtures/scans/read.json. The smoke tests' stand-in
 * for the model reads those lines (`FakeScanScript`), so a quote is washed
 * where the real model would put it on the page's picture.
 */
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";
import type { FakeScanScript } from "../../src/core/documents/scan/fake";
import { unsureWords } from "../../src/core/documents/scan/layout";
import { BUILT_IN_TEXT_RECOGNITION_MODEL } from "../../src/core/documents/scan/model";
import { createOnnxTextRecognizer } from "../../src/core/documents/scan/onnx";
import { downloadModel, isModelDownloaded } from "../../src/core/embedding/download";
import { evalCache } from "../lib/cache";
import { greyPicture } from "./lib/engines";
import type { SetPage } from "./lib/score";

const root = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const setDir = join(root, "tests/fixtures/scans");
const cacheDir = evalCache();
const round = (value: number) => Math.round(value);

test.runIf(process.env.INCARNAMIND_OCR_RECORD === "1")("record what the model reads", async () => {
  const set = (JSON.parse(await readFile(join(setDir, "set.json"), "utf8")) as { pages: SetPage[] })
    .pages;
  const model = BUILT_IN_TEXT_RECOGNITION_MODEL;
  const directory = join(cacheDir, "models", model.folder);
  if (!isModelDownloaded(directory, model.source.files)) {
    await downloadModel({
      directory,
      source: model.source,
      signal: new AbortController().signal,
      onProgress() {},
    });
  }
  const recognizer = createOnnxTextRecognizer();
  await recognizer.load({
    detector: join(directory, model.files.detector),
    recognizer: join(directory, model.files.recognizer),
  });
  const read: Record<string, FakeScanScript["pages"][string]> = {};
  for (const page of set) {
    const picture = await greyPicture(join(setDir, page.file));
    const lines = await recognizer.read(picture, 1);
    read[page.id] = {
      lines: lines.map((line) => ({
        text: line.text,
        // As fractions of the picture, so the stand-in places it on a picture of any size.
        quad: line.quad.flatMap(([x, y]) => [
          round((x / picture.width) * 10_000) / 10_000,
          round((y / picture.height) * 10_000) / 10_000,
        ]),
        unsure: unsureWords(line).map(([start, end]) => line.text.slice(start, end)),
      })),
    };
  }
  recognizer.close();
  await writeFile(
    join(setDir, "read.json"),
    `${JSON.stringify({ reader: model.reader, pages: read }, null, 1)}\n`,
  );
  expect(Object.keys(read)).toHaveLength(set.length);
});
