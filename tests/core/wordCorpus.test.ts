import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";
import { readManifest, WORD_DIR } from "../../eval/word/corpus";
import { checkImport } from "../../eval/word/imports";

test("every file of the Word corpus imports as a Mind the editor takes, its text all kept and what it kept counted", async () => {
  const failures: string[] = [];
  for (const entry of readManifest()) {
    const outcome = await checkImport(readFileSync(join(WORD_DIR, entry.file)));
    if (!outcome.ok) failures.push(`${entry.file}: ${outcome.detail}`);
  }
  expect(failures).toEqual([]);
});
