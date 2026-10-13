import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import type { Core, Document, ExportFormat } from "../../src/core";
import {
  createSourceFolder,
  documentAt,
  linkAndProcess,
  waitForDocuments,
  waitForProcessing,
  writeSourceFile,
} from "./documents";
import { connectToMind } from "./mindClient";
import { note, writeMind } from "./minds";

/**
 * A Linked folder "Papers" holding the files, indexed, with the Output
 * location inside it ("Papers/Exports") and new files there allowed: where
 * the exports IncarnaMind writes become Documents of its own (#80).
 */
export async function linkWithExports(
  core: Core,
  files: readonly { name: string; contents: string }[],
): Promise<{ papers: string; documents: Document[] }> {
  const papers = await createSourceFolder();
  for (const file of files) await writeSourceFile(papers, file.name, file.contents);
  const documents = await linkAndProcess(core, papers);
  const [linked] = await core.listLinkedFolders();
  if (!linked) throw new Error("Papers isn't linked.");
  await core.setLinkedFolderWriting(linked.id, true);
  await mkdir(join(papers, "Exports"));
  await core.setOutputLocation(join(papers, "Exports"));
  return { papers, documents };
}

/** A Mind holding one Note, exported into the Output location and indexed there as a Document. */
export async function exportMind(
  core: Core,
  title: string,
  text: string,
  format: ExportFormat = "docx",
): Promise<Document> {
  const mind = await core.createMind({ title });
  const client = await connectToMind(core, mind.id);
  writeMind(client, [note(text)]);
  const saved = await core.saveMindExport(mind.id, { format });
  if (saved.status !== "saved") throw new Error(`The export wasn't saved: ${saved.status}.`);
  return indexed(core, saved.file.path);
}

/** The Document at `path` once the Linked folder has picked it up and it is processed. */
export async function indexed(core: Core, path: string): Promise<Document> {
  await core.reconcileDocuments();
  const documents = await waitForDocuments(core, (all) => documentAt(all, path));
  const [document] = await waitForProcessing(core, [documentAt(documents, path).id]);
  if (!document) throw new Error(`Nothing was indexed at ${path}.`);
  return document;
}
