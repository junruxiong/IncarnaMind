/**
 * `npm run eval:word` (eval/word/README.md): for each file of the Word corpus, the checks that
 * can run today. Each is a step in `STEPS`; when the Word export (#105) exists, its round-trip
 * becomes more steps after "opens".
 *
 * INCARNAMIND_WORD_REQUIRE_LIBREOFFICE=1 (CI sets it) fails the run when LibreOffice is missing,
 * instead of skipping that step.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { type CorpusEntry, inspectPackage, readManifest, WORD_DIR } from "./corpus";
import { type Features, featureKinds } from "./features";
import { convertToPdf, findLibreOffice } from "./libreoffice";

interface Outcome {
  ok: boolean;
  detail: string;
}

interface Step {
  name: string;
  run(entry: CorpusEntry, bytes: Uint8Array): Promise<Outcome>;
}

const soffice = findLibreOffice();
const required = process.env.INCARNAMIND_WORD_REQUIRE_LIBREOFFICE === "1";

const STEPS: Step[] = [
  {
    name: "package",
    async run(entry, bytes) {
      const { features, problems } = await inspectPackage(bytes);
      if (problems.length > 0) return { ok: false, detail: problems.slice(0, 3).join("; ") };
      const drift = driftIn(entry.features, features);
      return drift
        ? { ok: false, detail: `manifest.json is out of date: ${drift}` }
        : { ok: true, detail: "" };
    },
  },
  {
    name: "opens in LibreOffice",
    async run(entry) {
      if (!soffice) return { ok: true, detail: "skipped: LibreOffice not found" };
      const result = await convertToPdf(soffice, join(WORD_DIR, entry.file));
      return result.opened
        ? { ok: true, detail: `${result.pdfBytes} bytes of PDF` }
        : { ok: false, detail: result.problem ?? "did not open" };
    },
  },
];

function driftIn(listed: Features, found: Features): string | undefined {
  const keys = Object.keys(found) as (keyof Features)[];
  const wrong = keys.filter((key) => JSON.stringify(listed[key]) !== JSON.stringify(found[key]));
  return wrong.length > 0 ? wrong.join(", ") : undefined;
}

describe("the Word corpus", () => {
  const manifest = readManifest();

  it("lists 30 documents and 20 files with tracked changes, each once", () => {
    expect(manifest.filter((entry) => entry.set === "documents")).toHaveLength(30);
    expect(manifest.filter((entry) => entry.set === "tracked-changes")).toHaveLength(20);
    expect(manifest.filter((entry) => entry.set === "saved").length).toBeLessThanOrEqual(6);
    expect(new Set(manifest.map((entry) => entry.file)).size).toBe(manifest.length);
  });

  it.skipIf(!required)("has LibreOffice when CI asks for it", () => {
    expect(soffice).toBeDefined();
  });

  it("opens every file, and reports each step", async () => {
    const failures: string[] = [];
    const lines: string[] = [];
    for (const entry of manifest) {
      const bytes = readFileSync(join(WORD_DIR, entry.file));
      const cells: string[] = [];
      for (const step of STEPS) {
        const outcome = await step.run(entry, bytes);
        cells.push(`${outcome.ok ? "ok" : "FAIL"} ${step.name}`);
        if (!outcome.ok) {
          const line = `${entry.file}: ${step.name}: ${outcome.detail}`;
          if (entry.known) lines.push(`  (known: ${entry.known}) ${line}`);
          else failures.push(line);
        }
      }
      lines.push(`${entry.file}  ${featureKinds(entry.features)} kinds  ${cells.join(", ")}`);
    }
    console.log(
      `${lines.join("\n")}\n${manifest.length} files; ${failures.length} failed${soffice ? "" : "; LibreOffice not found, so no file was opened in it"}`,
    );
    expect(failures).toEqual([]);
  });
});
