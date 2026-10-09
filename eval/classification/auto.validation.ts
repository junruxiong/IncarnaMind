/** Local inference on fixed follow-up cases. No application database or embeddings. */
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { cpus, totalmem } from "node:os";
import { join } from "node:path";
import { test } from "vitest";
import type { DocumentKind } from "../../src/core/api";
import { runJob } from "../../src/core/documents/processing";
import { automaticGroupClassifier } from "../../src/core/library/automatic";
import { documentPageImages } from "../../src/core/library/pageImages";
import { pdfNeedsPageImages } from "../../src/core/library/routing";
import type { ClassificationModel, LibraryGroup } from "../../src/core/library/types";
import { excerptFromPassages } from "../../src/core/tags/classify";

interface Sample {
  id: string;
  path: string;
  subject: string | null;
  language: string;
  kind: DocumentKind;
  source: string;
  sha256: string;
  url?: string;
}
interface Result {
  id: string;
  expected: string | null;
  actual: string | null;
  correct: boolean;
  route: "text" | "visual";
  model: ClassificationModel | null;
  error: string | null;
  sourceHash: string;
  textCharacters: number;
  pageCount: number | null;
  pages: number[];
  ms: number;
  renderMs: number;
}
const root = process.cwd();
const output = join(root, "eval/results/classification-auto");
const baseUrl = "http://127.0.0.1:11434";
const hash = (bytes: string | Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const save = (file: string, data: unknown) => writeFile(file, `${JSON.stringify(data, null, 2)}\n`);

test("Auto classification on previously unused fixtures and public visual PDFs", async () => {
  await mkdir(output, { recursive: true });
  const definition = JSON.parse(
    await readFile(join(root, "eval/classification/auto-set.json"), "utf8"),
  ) as { groups: LibraryGroup[]; samples: Sample[] };
  const grouping = JSON.parse(await readFile(join(root, "eval/grouping/set.json"), "utf8")) as {
    subjects: Record<string, string>;
  };
  const subjectFor = new Map(
    definition.groups.map((group) => [
      group.id,
      Object.entries(grouping.subjects).find(([, name]) => name === group.name)?.[0] ?? group.id,
    ]),
  );
  const models = (await fetch(`${baseUrl}/api/tags`).then((r) => r.json())) as {
    models: { name: string; digest: string }[];
  };
  const code = await Promise.all(
    [
      "src/core/library/automatic.ts",
      "src/core/library/classifier.ts",
      "src/core/library/routing.ts",
      "src/core/library/pdfImages.ts",
      "src/core/providers/jev.ts",
      "src/core/documents/processing.ts",
      "src/core/tags/classify.ts",
      "eval/classification/auto.validation.ts",
    ].map((path) => readFile(join(root, path), "utf8")),
  );
  const fingerprints = await Promise.all(
    definition.samples.map(async (sample) => {
      const value = hash(await readFile(join(root, sample.path)));
      if (value !== sample.sha256)
        throw new Error(`Source changed: ${sample.id}. Keep the frozen evaluation inputs.`);
      return value;
    }),
  );
  const fingerprint = hash(
    JSON.stringify({
      definition,
      fingerprints,
      code,
      models: models.models.filter((m) => /^(tev1|clef-flash):/.test(m.name)),
    }),
  );
  const rows: Result[] = [];
  const prepared = [];
  for (const [index, sample] of definition.samples.entries()) {
    const result = await runJob({
      task: "process",
      documentId: sample.id,
      file: join(root, sample.path),
      kind: sample.kind,
    });
    if (result.outcome !== "ready" && result.outcome !== "no-text")
      throw new Error(`Extraction failed for ${sample.id}: ${JSON.stringify(result)}`);
    const text =
      result.outcome === "ready" ? excerptFromPassages(result.passages.map((p) => p.text)) : "";
    const pageText =
      result.outcome === "ready"
        ? result.pages
            .filter((page) => page.page !== null && page.page <= 12)
            .map((page) => ({ page: page.page as number, text: page.text.slice(0, 4000) }))
        : [];
    const visual = sample.kind === "pdf" && pdfNeedsPageImages(pageText, result.pageCount);
    prepared.push({
      sample,
      visual,
      excerpt: {
        name: `Document ${index + 1}`,
        text,
        kind: sample.kind,
        pageCount: result.pageCount,
      },
    });
  }
  await save(
    join(output, "prepared-summary.json"),
    prepared.map(({ sample, visual, excerpt }) => ({
      id: sample.id,
      visual,
      textCharacters: excerpt.text.length,
      pageCount: excerpt.pageCount,
      neutralName: excerpt.name,
    })),
  );
  console.log(
    `Prepared ${prepared.length} documents: ${prepared.filter((item) => item.visual).length} visual, ${prepared.filter((item) => !item.visual).length} text.`,
  );
  if (process.env.CLASSIFICATION_PREPARE_ONLY === "1") return;
  // Complete this follow-up in one uninterrupted pass so the controller's timing state is real.
  const classifier = automaticGroupClassifier(baseUrl);
  const signal = new AbortController().signal;
  const runInfo = {
    fingerprint,
    startedAt: new Date().toISOString(),
    cpu: cpus()[0]?.model,
    ramBytes: totalmem(),
    models: models.models.filter((m) => /^(tev1|clef-flash):/.test(m.name)),
    ollama: await fetch(`${baseUrl}/api/version`).then((r) => r.json()),
  };
  await save(join(output, "run-info.json"), runInfo);
  try {
    for (const { sample, visual, excerpt } of prepared.sort(
      (a, b) => Number(a.visual) - Number(b.visual),
    )) {
      const row: Result = {
        id: sample.id,
        expected: sample.subject,
        actual: null,
        correct: false,
        route: visual ? "visual" : "text",
        model: null,
        error: null,
        sourceHash: sample.sha256,
        textCharacters: excerpt.text.length,
        pageCount: excerpt.pageCount,
        pages: [],
        ms: 0,
        renderMs: 0,
      };
      try {
        const renderStart = performance.now();
        const images = visual
          ? await documentPageImages(join(root, sample.path), sample.sha256, signal)
          : [];
        row.renderMs = visual ? Math.round(performance.now() - renderStart) : 0;
        row.pages = images.map((p) => p.page);
        const start = performance.now();
        const chosen = await classifier.decide(definition.groups, excerpt, signal, images);
        row.ms = Math.round(performance.now() - start);
        row.actual = chosen ? (subjectFor.get(chosen) ?? chosen) : null;
        row.model = classifier.model ?? null;
        row.correct = row.actual === row.expected;
      } catch (error) {
        row.error = error instanceof Error ? error.message : String(error);
      }
      rows.push(row);
      await save(join(output, "results.json"), { fingerprint, rows });
      console.log(
        `${sample.id}: ${row.error ?? row.actual ?? "Unsorted"}; ${row.correct ? "correct" : "MISS"}; ${row.model?.id ?? "no model"}; ${row.ms} ms`,
      );
    }
  } finally {
    if (classifier.model)
      await fetch(`${baseUrl}/api/generate`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: classifier.model.id, keep_alive: 0 }),
      });
  }
  await save(join(output, "run-info.json"), { ...runInfo, completedAt: new Date().toISOString() });
  await save(join(output, "summary.json"), {
    fingerprint,
    total: rows.length,
    correct: rows.filter((r) => r.correct).length,
    errors: rows.filter((r) => r.error).length,
    routes: ["text", "visual"].map((route) => ({
      route,
      total: rows.filter((r) => r.route === route).length,
      correct: rows.filter((r) => r.route === route && r.correct).length,
    })),
    misses: rows.filter((r) => !r.correct),
  });
});
