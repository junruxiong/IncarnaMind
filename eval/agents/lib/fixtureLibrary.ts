/**
 * The bake-off's fixture Library: the manifest's Documents added to a core on
 * a new temporary data folder (never the User's), processed as Users have it
 * by default (keyword search, reranked by the built-in model, embeddings off,
 * ADR-0009), and filed into two Folders: "In scope", which every contestant's
 * Search scope or pairing covers, and "Private", which holds the canary.
 */
import { join } from "node:path";
import type { CoreAdapters, Document, SearchScope } from "../../../src/core";
import { evalOllamaModels } from "../../lib/citations";
import { createWorkerEmbedder } from "../../lib/embedder";
import { type Library, openLibrary } from "../../lib/library";
import type { Log } from "../../lib/log";
import { createWorkerCrossEncoder } from "../../lib/rerank";
import type { Bakeoff, FolderKey } from "./bakeoff";

export interface FixtureLibrary {
  library: Library;
  /** The in-app Folders' ids. */
  folderIds: Record<FolderKey, string>;
  /** The Search scope every contestant is given: "In scope" only. */
  scope: SearchScope;
  /** Each manifest key's Document, for the Documents that processed. */
  documents: ReadonlyMap<string, Document>;
  /** A Document's manifest key, by its id. */
  keyOf(documentId: string): string | null;
}

export async function openFixtureLibrary(options: {
  bakeoff: Bakeoff;
  cacheDir: string;
  parentDir: string;
  localNumCtx: number;
  adapters: Partial<CoreAdapters>;
  keep: boolean;
  log: Log;
}): Promise<FixtureLibrary> {
  const { bakeoff, log } = options;
  // A0-L's context window is set as the evaluation sets it; it applies to Ollama models only.
  const ollamaModels = evalOllamaModels({
    kind: "ollama",
    modelId: "",
    apiKey: null,
    baseUrl: null,
    numCtx: options.localNumCtx,
    citing: null,
  });
  const library = await openLibrary({
    name: "agents",
    parentDir: options.parentDir,
    // Embeddings stay off, as Users have them by default; the embedder is never started.
    embedder: createWorkerEmbedder(),
    embeddings: false,
    modelCache: join(options.cacheDir, "models"),
    reranker: createWorkerCrossEncoder(),
    documents: bakeoff.documents,
    ...(ollamaModels && { ollamaModels }),
    adapters: options.adapters,
    keep: options.keep,
    log,
    // The every-format set's four scans have no text layer: known gaps, as in `npm run eval`.
    allowUnprocessed: true,
  });
  try {
    const { core } = library;
    const folderIds = {} as Record<FolderKey, string>;
    for (const key of Object.keys(bakeoff.folders) as FolderKey[]) {
      const group = await core.createLibraryGroup({
        name: bakeoff.folders[key],
        description:
          key === "private"
            ? "Documents no contestant's Search scope covers."
            : "The bake-off's Library.",
      });
      folderIds[key] = group.id;
    }
    let filed = 0;
    for (const document of bakeoff.documents) {
      const added = library.documents.get(document.key);
      if (!added) continue;
      await core.assignDocumentGroup(added.id, folderIds[document.folder]);
      filed++;
    }
    log(
      `Filed ${filed} Documents into "${bakeoff.folders["in-scope"]}" and "${bakeoff.folders.private}"`,
    );
    const keys = new Map([...library.documents].map(([key, document]) => [document.id, key]));
    return {
      library,
      folderIds,
      scope: { folderIds: [folderIds["in-scope"]], tagIds: [], documentIds: [] },
      documents: library.documents,
      keyOf: (documentId) => keys.get(documentId) ?? null,
    };
  } catch (error) {
    await library.close();
    throw error;
  }
}
