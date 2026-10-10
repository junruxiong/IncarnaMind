/**
 * The built-in reranking model as the rest of the core sees it: its download
 * state (as the built-in embedding model's), loading it into the
 * `CrossEncoder`, and reordering search hits by its scores. Nothing it reads
 * leaves this computer; only its files are downloaded, once.
 */
import { normaliseText } from "../../shared/text";
import type { CrossEncoder, EmbeddingModelSource } from "../adapters";
import type { EmbeddingModelStatus } from "../api";
import { createDownloadableModel } from "../embedding/downloadable";
import type { RerankingModelDefinition } from "./model";

export { BUILT_IN_RERANKING_MODEL, type RerankingModelDefinition } from "./model";

/** What the model reads of a hit: its Document's name and its text. */
export interface Rerankable {
  documentName: string;
  text: string;
}

export interface RerankingModel {
  readonly definition: RerankingModelDefinition;
  status(): EmbeddingModelStatus;
  /** Downloaded and checked. */
  isReady(): boolean;
  /** Starts the download if it isn't downloaded or downloading (see `DownloadableModel.ensure`). */
  ensure(): void;
  /** Starts the download, or tries again after a failure. */
  retry(): EmbeddingModelStatus;
  /**
   * The hits in the model's order, best first, each with the model's score
   * from 0 to 1; ties keep their order. Null, with nothing read, while the
   * model isn't downloaded or can't start. Rejects if the model fails.
   */
  rerank<Hit extends Rerankable>(
    query: string,
    hits: readonly Hit[],
  ): Promise<(Hit & { score: number })[] | null>;
  /** Stops the model, freeing its memory, e.g. when the User turns reranking off. */
  unload(): void;
  /** Stops a download (keeping what it got) and the model. */
  close(): void;
}

export interface RerankingModelOptions {
  definition: RerankingModelDefinition;
  /** Overrides the definition's source (tests). */
  source?: EmbeddingModelSource;
  dataDir: string;
  crossEncoder: CrossEncoder;
  emitStatus(status: EmbeddingModelStatus): void;
  reportError?: (error: unknown) => void;
}

/** What the model reads for a hit, normalised as for embedding: its Document's name, then its text. */
export const rerankText = (hit: Rerankable) =>
  `${normaliseText(hit.documentName)}\n${normaliseText(hit.text)}`;

/** A logit as a score from 0 to 1, so scores stay positive and comparable, as search's are. */
const sigmoid = (logit: number) => 1 / (1 + Math.exp(-logit));

export function createRerankingModel(options: RerankingModelOptions): RerankingModel {
  const { definition, crossEncoder } = options;
  const files = createDownloadableModel({
    name: definition.name,
    folder: definition.folder,
    files: definition.files,
    maxTokens: definition.maxTokens,
    source: options.source ?? definition.source,
    dataDir: options.dataDir,
    loadRunner: (paths) => crossEncoder.load(paths),
    emitStatus: options.emitStatus,
    reportError: options.reportError,
  });

  return {
    definition,
    status: files.status,
    isReady: files.isReady,
    ensure: files.ensure,
    retry: files.retry,
    async rerank(query, hits) {
      if (!(await files.load())) return null;
      if (hits.length === 0) return [];
      const logits = await crossEncoder.score(normaliseText(query), hits.map(rerankText));
      return hits
        .map((hit, index) => ({
          ...hit,
          score: sigmoid(logits[index] ?? Number.NEGATIVE_INFINITY),
        }))
        .map((hit, index) => ({ hit, index }))
        .sort((a, b) => b.hit.score - a.hit.score || a.index - b.index)
        .map(({ hit }) => hit);
    },
    unload() {
      crossEncoder.close();
    },
    close() {
      files.close();
      crossEncoder.close();
    },
  };
}
