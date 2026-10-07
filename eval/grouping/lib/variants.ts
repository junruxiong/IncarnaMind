/**
 * The three ways of building a Document's vector that the grouping check
 * compares (R3 in docs/designs/library-structure-view.md):
 *
 * 1. names included: the mean of its Passage vectors as stored, which carry
 *    the Document's name (`passage: <name>\n<text>`), from the vector index's
 *    documentMeans (R2);
 * 2. names removed: that mean with the direction of the name's own vector
 *    (`passage: <name>`, embedded once per Document) taken out;
 * 3. name-free: each Passage embedded again without the name, for the check
 *    only, and averaged.
 *
 * All three use the built-in model through the evaluation's embedding worker.
 */
import { join } from "node:path";
import {
  BUILT_IN_EMBEDDING_MODEL,
  DATABASE_FILE,
  type Document,
  type DocumentKind,
  type Embedder,
} from "../../../src/core";
import { createVectorIndex } from "../../../src/core/documents/vectors";
import { type Database, openDatabase } from "../../../src/core/storage";
import { meanDirection, normalised, withoutDirection } from "../../../src/core/topics/grouping";
import { normaliseText } from "../../../src/shared/text";
import type { Log } from "../../lib/log";

export type VariantId = "names-included" | "names-removed" | "name-free";

export interface Variant {
  id: VariantId;
  label: string;
  /** Extra cost over the vectors search already has, lowest first: it decides near-ties. */
  rank: number;
  cost: string;
}

export const VARIANTS: readonly Variant[] = [
  {
    id: "names-included",
    label: "Passage means, names included",
    rank: 0,
    cost: "none: the Passage vectors search already uses",
  },
  {
    id: "names-removed",
    label: "Passage means with the name's direction removed",
    rank: 1,
    cost: "one embedding of each Document's name",
  },
  {
    id: "name-free",
    label: "Name-free Passage means",
    rank: 2,
    cost: "every Passage embedded again, without the name",
  },
];

/** A Document as the check reads it from the data folder. */
export interface CorpusDocument {
  /** The core's id. */
  id: string;
  /** Its key in the set (or its path relative to the folder, for the founder's library). */
  key: string;
  name: string;
  kind: DocumentKind;
  pageCount: number | null;
  /** Its live Passages' text, in order. */
  passages: string[];
}

export interface Corpus {
  documents: CorpusDocument[];
  db: Database;
  close(): void;
}

/** The ready Documents' names and Passages, read from the data folder's database (WAL: alongside the core). */
export function readCorpus(dataDir: string, documents: ReadonlyMap<string, Document>): Corpus {
  const db = openDatabase(join(dataDir, DATABASE_FILE));
  const corpus: CorpusDocument[] = [];
  for (const [key, document] of documents) {
    if (document.status !== "ready") continue;
    const passages = db
      .all<{ text: string }>(
        "SELECT text FROM passages WHERE document_id = ? AND deleted_at IS NULL ORDER BY position",
        [document.id],
      )
      .map((row) => row.text);
    corpus.push({
      id: document.id,
      key,
      name: document.name,
      kind: document.kind,
      pageCount: document.pageCount,
      passages,
    });
  }
  return { documents: corpus, db, close: () => db.close() };
}

/** Text to its unit vector, with the built-in model, as is: the caller adds the prefix. */
export type TextEmbedder = (text: string) => Promise<Float32Array>;

/** What the model reads for a text embedded as a Passage, with no Document name: as the core writes it, less the name. */
export const passageText = (text: string) =>
  `${BUILT_IN_EMBEDDING_MODEL.passagePrefix}${normaliseText(text)}`;

/** The built-in model, through the embedder the core used (loaded already, or loaded now from the data folder). */
export async function builtInTextEmbedder(
  embedder: Embedder,
  dataDir: string,
): Promise<TextEmbedder> {
  const model = BUILT_IN_EMBEDDING_MODEL;
  const folder = join(dataDir, "models", model.folder);
  await embedder.load({
    model: join(folder, model.files.model),
    tokenizer: join(folder, model.files.tokenizer),
    tokenizerConfig: join(folder, model.files.tokenizerConfig),
    maxTokens: model.maxTokens,
  });
  return async (text) => {
    const vector = normalised(await embedder.embed(text));
    if (!vector) throw new Error("The embedding model returned an empty vector.");
    return vector;
  };
}

export interface VariantVectors {
  id: VariantId;
  /** By Document key. */
  vectors: Map<string, Float32Array>;
  /** Time spent building them, embeddings included. */
  seconds: number;
  /** Texts embedded for this variant alone. */
  embeddings: number;
}

const seconds = (started: number) => (performance.now() - started) / 1000;

/** 1. The stored Passage vectors' means, from the vector index (R2). */
export function namesIncluded(corpus: Corpus): VariantVectors {
  const started = performance.now();
  const means = createVectorIndex(corpus.db, { id: BUILT_IN_EMBEDDING_MODEL.id }).documentMeans(
    BUILT_IN_EMBEDDING_MODEL.dimensions,
  );
  const keyOf = new Map(corpus.documents.map((document) => [document.id, document.key]));
  const vectors = new Map<string, Float32Array>();
  means.ids.forEach((id, index) => {
    const key = keyOf.get(id);
    if (key === undefined) return;
    vectors.set(key, means.data.slice(index * means.dimensions, (index + 1) * means.dimensions));
  });
  return { id: "names-included", vectors, seconds: seconds(started), embeddings: 0 };
}

/** 2. Each mean with its Document name's direction removed (one name embedding per Document). */
export async function namesRemoved(
  corpus: Corpus,
  base: VariantVectors,
  embed: TextEmbedder,
): Promise<VariantVectors> {
  const started = performance.now();
  const vectors = new Map<string, Float32Array>();
  let embeddings = 0;
  for (const document of corpus.documents) {
    const mean = base.vectors.get(document.key);
    if (!mean) continue;
    const name = await embed(passageText(document.name));
    embeddings++;
    vectors.set(document.key, withoutDirection(mean, name) ?? mean);
  }
  return { id: "names-removed", vectors, seconds: seconds(started), embeddings };
}

/** 3. Each Passage embedded again without the name, and averaged. */
export async function nameFree(
  corpus: Corpus,
  embed: TextEmbedder,
  log: Log,
): Promise<VariantVectors> {
  const started = performance.now();
  const total = corpus.documents.reduce((sum, document) => sum + document.passages.length, 0);
  const vectors = new Map<string, Float32Array>();
  let embeddings = 0;
  let logged = Date.now();
  for (const document of corpus.documents) {
    const passages: Float32Array[] = [];
    for (const text of document.passages) {
      passages.push(await embed(passageText(text)));
      embeddings++;
      if (Date.now() - logged > 15_000) {
        logged = Date.now();
        log(`Name-free embeddings: ${embeddings} of ${total} Passages`);
      }
    }
    const mean = meanDirection(passages);
    if (mean) vectors.set(document.key, mean);
  }
  return { id: "name-free", vectors, seconds: seconds(started), embeddings };
}
