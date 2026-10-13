import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";
import { readManifest, WORD_DIR } from "../../eval/word/corpus";
import { checkEdits, checkExport } from "../../eval/word/exports";
import { checkImport } from "../../eval/word/imports";

const files = readManifest().map((entry) => entry.file);

test.each(files)(
  "%s imports as a Mind the editor takes, its text all kept and what it kept counted",
  async (file) => {
    const outcome = await checkImport(readFileSync(join(WORD_DIR, file)));
    expect(outcome.ok, outcome.detail).toBe(true);
  },
  60_000,
);

test.each(files)(
  "%s exports back into itself: unchanged byte for byte, edited with the rest as it was",
  async (file) => {
    const outcome = await checkExport(readFileSync(join(WORD_DIR, file)));
    expect(outcome.ok, outcome.detail).toBe(true);
  },
  60_000,
);

test.each(files)(
  "%s takes edits of every kind: the export is a package Word opens and imports again",
  async (file) => {
    const outcome = await checkEdits(readFileSync(join(WORD_DIR, file)));
    expect(outcome.ok, outcome.detail).toBe(true);
  },
  60_000,
);
