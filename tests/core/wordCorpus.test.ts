import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";
import { readManifest, WORD_DIR } from "../../eval/word/corpus";
import { checkImport } from "../../eval/word/imports";

test.each(readManifest().map((entry) => entry.file))(
  "%s imports as a Mind the editor takes, its text all kept and what it kept counted",
  async (file) => {
    const outcome = await checkImport(readFileSync(join(WORD_DIR, file)));
    expect(outcome.ok, outcome.detail).toBe(true);
  },
);
