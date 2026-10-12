/**
 * `npm run eval:word` (eval/word/README.md): for each file of the Word corpus, the checks that
 * can run today. Each is a step in `STEPS`; when the Word export (#105) exists, its round-trip
 * becomes more steps after "opens".
 *
 * INCARNAMIND_WORD_SET picks the files: "core" (the committed files, the default), "extended"
 * (the cache that `npm run eval:word:fetch` fills) or "all".
 * INCARNAMIND_WORD_REQUIRE_LIBREOFFICE=1 (CI sets it) fails the run when LibreOffice is missing,
 * instead of skipping that step.
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  type CorpusEntry,
  extendedCache,
  inspectPackage,
  readExtended,
  readManifest,
  WORD_DIR,
} from "./corpus";
import { type Features, featureKinds } from "./features";
import { convertToPdf, findLibreOffice } from "./libreoffice";

interface Outcome {
  ok: boolean;
  detail: string;
}

/** A file to check, wherever it lives. */
interface Subject {
  name: string;
  path: string;
  features: Features;
  known?: string;
}

interface Step {
  name: string;
  run(subject: Subject, bytes: Uint8Array): Promise<Outcome>;
}

const soffice = findLibreOffice();
const required = process.env.INCARNAMIND_WORD_REQUIRE_LIBREOFFICE === "1";
const wanted = process.env.INCARNAMIND_WORD_SET ?? "core";

const STEPS: Step[] = [
  {
    name: "package",
    async run(subject, bytes) {
      const { features, problems } = await inspectPackage(bytes);
      if (problems.length > 0) return { ok: false, detail: problems.slice(0, 3).join("; ") };
      const drift = driftIn(subject.features, features);
      return drift
        ? { ok: false, detail: `its listed features are out of date: ${drift}` }
        : { ok: true, detail: "" };
    },
  },
  {
    name: "opens in LibreOffice",
    async run(subject) {
      if (!soffice) return { ok: true, detail: "skipped: LibreOffice not found" };
      const result = await convertToPdf(soffice, subject.path);
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

function subjects(): Subject[] {
  const found: Subject[] = [];
  if (wanted === "core" || wanted === "all") {
    for (const entry of readManifest()) {
      found.push({
        name: entry.file,
        path: join(WORD_DIR, entry.file),
        features: entry.features,
        known: entry.known,
      });
    }
  }
  if (wanted === "extended" || wanted === "all") {
    const cache = extendedCache();
    for (const entry of readExtended()) {
      found.push({
        name: `extended/${entry.name}`,
        path: join(cache, entry.name),
        features: entry.features,
        known: entry.known,
      });
    }
  }
  return found;
}

describe("the Word corpus", () => {
  const manifest = readManifest();

  it("lists 30 documents, 20 files with tracked changes and at most 6 saved by hand, each once", () => {
    expect(manifest.filter((entry) => entry.set === "documents")).toHaveLength(30);
    expect(manifest.filter((entry) => entry.set === "tracked-changes")).toHaveLength(20);
    expect(manifest.filter((entry) => entry.set === "saved").length).toBeLessThanOrEqual(6);
    expect(new Set(manifest.map((entry) => entry.file)).size).toBe(manifest.length);
  });

  it("keeps the committed files small", () => {
    const bytes = manifest.reduce(
      (sum, entry: CorpusEntry) => sum + readFileSync(join(WORD_DIR, entry.file)).byteLength,
      0,
    );
    expect(bytes).toBeLessThan(25_000_000);
  });

  it.skipIf(!required)("has LibreOffice when CI asks for it", () => {
    expect(soffice).toBeDefined();
  });

  it(`opens every file of the "${wanted}" set, and reports each step`, async () => {
    const all = subjects();
    const missing = all.filter((subject) => !existsSync(subject.path));
    expect(
      missing.map((subject) => subject.name),
      "fetch the extended set first: npm run eval:word:fetch",
    ).toEqual([]);

    const failures: string[] = [];
    const lines: string[] = [];
    for (const subject of all) {
      const bytes = readFileSync(subject.path);
      const cells: string[] = [];
      for (const step of STEPS) {
        const outcome = await step.run(subject, bytes);
        cells.push(`${outcome.ok ? "ok" : "FAIL"} ${step.name}`);
        if (!outcome.ok) {
          const line = `${subject.name}: ${step.name}: ${outcome.detail}`;
          if (subject.known) lines.push(`  (known: ${subject.known}) ${line}`);
          else failures.push(line);
        }
      }
      lines.push(`${subject.name}  ${featureKinds(subject.features)} kinds  ${cells.join(", ")}`);
    }
    console.log(
      `${lines.join("\n")}\n${all.length} files; ${failures.length} failed${soffice ? "" : "; LibreOffice not found, so no file was opened in it"}`,
    );
    expect(failures).toEqual([]);
  });
});
