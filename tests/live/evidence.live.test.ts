/**
 * What checks may use as evidence (#80), with the real built-in reranking
 * model, through the core as the app uses it: a Mind exported to Word and to
 * Markdown into a Linked folder beside the User's own PDF, Word file,
 * spreadsheet and Chinese notes, then searched as a check searches, in
 * English and in Chinese, and as a verify of the PDF, whose copy sits in
 * another folder.
 *
 * Skipped unless INCARNAMIND_REAL_RERANKER=1: the model is a 136 MB download
 * from Hugging Face. INCARNAMIND_RERANKER_MODEL_DIR reuses one downloaded
 * before, as in tests/core/realReranker.test.ts. Set INCARNAMIND_LIVE_REPORT
 * to a file to keep what each search found.
 */
import { appendFile, cp, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import { BUILT_IN_RERANKING_MODEL, type Core, type Document } from "../../src/core";
import { createOnnxCrossEncoder } from "../../src/core/reranking/onnx";
import { createTempDataFolder, startCore } from "../helpers/core";
import {
  createSourceFolder,
  documentAt,
  linkAndProcess,
  waitForDocuments,
  waitForProcessing,
} from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { heading, note, writeMind } from "../helpers/minds";
import { docxOf, xlsxOf } from "../helpers/office";
import { buildPdf } from "../helpers/pdf";

const report = async (line: string) => {
  const file = process.env.INCARNAMIND_LIVE_REPORT;
  if (file) await appendFile(file, `${line}\n`);
};

/** Saves the Mind's export into the Output location and waits until the Linked folder has indexed it. */
async function exported(core: Core, mindId: string, format: "docx" | "markdown") {
  const saved = await core.saveMindExport(mindId, { format });
  if (saved.status !== "saved") throw new Error(`The ${format} export wasn't saved.`);
  await core.reconcileDocuments();
  const all = await waitForDocuments(core, (documents) => documentAt(documents, saved.file.path));
  const [document] = await waitForProcessing(core, [documentAt(all, saved.file.path).id], 60_000);
  if (!document) throw new Error(`The ${format} export wasn't indexed.`);
  return document;
}

describe.runIf(process.env.INCARNAMIND_REAL_RERANKER === "1")(
  "Checks' evidence with the real reranking model",
  { timeout: 30 * 60_000 },
  () => {
    test("exports never count, the User's own files in every format do, and a copy of the PDF being verified is left out", async () => {
      const dataDir = await createTempDataFolder();
      const cached = process.env.INCARNAMIND_RERANKER_MODEL_DIR;
      if (cached) {
        await cp(
          join(cached, BUILT_IN_RERANKING_MODEL.id),
          join(dataDir, "models", BUILT_IN_RERANKING_MODEL.folder),
          { recursive: true },
        );
      }
      const core = startCore(dataDir, {
        crossEncoder: createOnnxCrossEncoder(),
        rerankingModelSource: BUILT_IN_RERANKING_MODEL.source,
      });

      const papers = await createSourceFolder();
      const pdf = buildPdf([
        { lines: ["Tide tables", "Spring tides happen at new moon and at full moon."] },
        { lines: ["Neap tides", "Neap tides come at the first and last quarter moons."] },
      ]);
      await writeFile(join(papers, "Tide tables.pdf"), pdf);
      await writeFile(
        join(papers, "Harbour report.docx"),
        docxOf([
          { text: "Harbour report", heading: 1 },
          { text: "The harbour wall was raised in 2015 after the spring tides flooded the quay." },
        ]),
      );
      await writeFile(
        join(papers, "Tide heights.xlsx"),
        xlsxOf([
          {
            name: "Heights",
            rows: [
              ["Tide", "Height (m)"],
              ["Spring tide", 5.9],
              ["Neap tide", 3.1],
            ],
          },
        ]),
      );
      await writeFile(
        join(papers, "潮汐笔记.md"),
        "# 潮汐笔记\n\n大潮出现在新月和满月前后，这时太阳和月亮的引力叠加在一起。\n",
      );
      const own = await linkAndProcess(core, papers);
      expect(own.map((each) => each.status)).toEqual(["ready", "ready", "ready", "ready"]);
      const [linked] = await core.listLinkedFolders();
      if (!linked) throw new Error("The folder isn't linked.");
      await core.setLinkedFolderWriting(linked.id, true);
      await core.setOutputLocation(papers);

      const mind = await core.createMind({ title: "My tide notes" });
      writeMind(await connectToMind(core, mind.id), [
        heading(1, "Tides"),
        note("Spring tides come twice a month, at new moon and at full moon."),
        note("The harbour wall was raised in 2015 after a flood."),
        note("大潮每月两次，出现在新月和满月。"),
      ]);
      const word = await exported(core, mind.id, "docx");
      const markdown = await exported(core, mind.id, "markdown");
      expect(word.ownFile?.kind).toBe("export");
      expect(markdown.ownFile?.kind).toBe("export");

      await core.downloadRerankingModel();
      await vi.waitFor(
        async () => expect((await core.getRerankSettings()).model.state).toBe("ready"),
        { timeout: 20 * 60_000, interval: 1000 },
      );

      const exports = new Set([word.id, markdown.id]);
      const names = (documents: { documentName: string }[]) => [
        ...new Set(documents.map((each) => each.documentName)),
      ];
      for (const query of [
        "When do spring tides happen?",
        "When was the harbour wall raised?",
        "How high is a spring tide?",
        "大潮什么时候出现？",
      ]) {
        const found = await core.searchEvidence(query);
        const ordinary = await core.searchPassages(query);
        await report(
          `${query} | evidence: ${names(found.passages).join(", ")} | left out: ${names(found.excluded).join(", ")} | ordinary: ${names(ordinary).join(", ")}`,
        );
        expect(found.passages.length).toBeGreaterThan(0);
        expect(found.passages.some((each) => exports.has(each.documentId))).toBe(false);
        expect(found.excluded.every((each) => exports.has(each.documentId))).toBe(true);
        expect(found.excluded.length).toBeGreaterThan(0);
      }

      // Verifying the PDF: it and its copy elsewhere never support it; the User's other files do.
      const sent = join(await createSourceFolder(), "Sent");
      await mkdir(sent);
      await writeFile(join(sent, "Tide tables (sent).pdf"), pdf);
      const [copy] = await linkAndProcess(core, sent);
      const original = own.find((each: Document) => each.name === "Tide tables");
      if (!copy || !original) throw new Error("The PDFs weren't indexed.");
      const verified = await core.searchEvidence("Spring tides happen at new moon and full moon", {
        checked: { documentIds: [original.id] },
      });
      await report(
        `verify the PDF | evidence: ${names(verified.passages).join(", ")} | left out: ${verified.excluded.map((each) => `${each.documentName} (${each.reason})`).join(", ")}`,
      );
      const verifiedIds = new Set(verified.passages.map((each) => each.documentId));
      expect(verifiedIds.has(original.id) || verifiedIds.has(copy.id)).toBe(false);
      expect(
        verified.excluded
          .filter((each) => each.reason === "checked-document")
          .map((each) => each.documentId)
          .sort(),
      ).toEqual([original.id, copy.id].sort());
    });
  },
);
