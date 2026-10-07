/**
 * The built-in embedding model (ADR-0009): multilingual-e5-small, int8 ONNX
 * (Xenova/multilingual-e5-small, MIT, 384 dimensions). It found 14 of the 20
 * evaluation questions with vector search alone, one behind the best model at
 * a quarter of its download, and indexes about 25 Passages a second on a laptop CPU.
 */
import type { EmbeddingModelSource } from "../adapters";

export interface EmbeddingModelDefinition {
  /** Recorded with the vectors (documents.embedding_model), so vectors from another model are never mixed in. */
  id: string;
  /** What the User sees. */
  name: string;
  /** The model's folder under `models/` in the data folder. */
  folder: string;
  dimensions: number;
  /** The model reads at most this many tokens of a text. */
  maxTokens: number;
  /** e5 models are trained with these prefixes: "passage: " for what is searched, "query: " for searches. */
  passagePrefix: string;
  queryPrefix: string;
  /** The files the runner loads, relative to the model's folder. */
  files: { model: string; tokenizer: string; tokenizerConfig: string };
  source: EmbeddingModelSource;
}

/** A pinned revision, so the files never change under their recorded sizes and hashes. */
const REVISION = "761b726dd34fb83930e26aab4e9ac3899aa1fa78";

export const BUILT_IN_EMBEDDING_MODEL: EmbeddingModelDefinition = {
  id: "multilingual-e5-small-int8",
  name: "multilingual-e5-small",
  folder: "multilingual-e5-small",
  dimensions: 384,
  maxTokens: 512,
  passagePrefix: "passage: ",
  queryPrefix: "query: ",
  files: {
    model: "onnx/model_quantized.onnx",
    tokenizer: "tokenizer.json",
    tokenizerConfig: "tokenizer_config.json",
  },
  source: {
    baseUrl: `https://huggingface.co/Xenova/multilingual-e5-small/resolve/${REVISION}/`,
    files: [
      {
        path: "onnx/model_quantized.onnx",
        size: 118_308_185,
        sha256: "f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193",
      },
      {
        path: "tokenizer.json",
        size: 17_082_730,
        sha256: "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
      },
      {
        path: "tokenizer_config.json",
        size: 443,
        sha256: "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
      },
    ],
  },
};
