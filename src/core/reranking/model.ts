/**
 * The built-in reranking model's candidates: multilingual cross-encoders with
 * permissive licences, as int8 ONNX, each pinned to a revision with its files'
 * sizes and SHA-256 hashes. The retrieval evaluation (eval/README.md, #31)
 * compares them over what the search Tool hands a reranker (keyword search's
 * top 10 and vector search's top 10, each Passage once); the one Settings
 * offers is `BUILT_IN_RERANKING_MODEL`.
 *
 * Measured on an Apple M2 Max, one pair at a time, 20 Passages of about 500
 * tokens per query, in a Node process of its own:
 * - mmarco-mMiniLMv2-L12-H384 (Apache-2.0, 118M parameters): 136 MB, about
 *   0.5 s a query, 0.8 GB resident.
 * - gte-multilingual-reranker-base (Apache-2.0, 306M): 358 MB, about 1.6 s,
 *   1.25 GB.
 * - bge-reranker-v2-m3 (Apache-2.0, 568M): 588 MB, about 4.3 s, 1.9 GB.
 *
 * Not candidates: jina-reranker-v2-base-multilingual (CC BY-NC 4.0, not for
 * commercial use), and English-only or English-and-Chinese-only rerankers.
 */
import type { EmbeddingModelSource } from "../adapters";

export interface RerankingModelDefinition {
  /** A short id, e.g. for the evaluation's INCARNAMIND_EVAL_RERANK. */
  id: string;
  /** What the User sees. */
  name: string;
  /** The model's folder under `models/` in the data folder. */
  folder: string;
  /** The licence of the model and of the files downloaded, as an SPDX id. */
  licence: string;
  /** The query and a Passage, together, are cut to this many tokens. */
  maxTokens: number;
  /** The files the runner loads, relative to the model's folder. */
  files: { model: string; tokenizer: string; tokenizerConfig: string };
  source: EmbeddingModelSource;
}

/** cross-encoder/mmarco-mMiniLMv2-L12-H384-v1, its sentence-transformers int8 ONNX export (the same file serves every CPU). */
export const MMARCO_MINILM: RerankingModelDefinition = {
  id: "mmarco-minilm",
  name: "mmarco-mMiniLMv2-L12-H384",
  folder: "mmarco-mMiniLMv2-L12-H384-v1",
  licence: "Apache-2.0",
  maxTokens: 512,
  files: {
    model: "onnx/model_qint8_arm64.onnx",
    tokenizer: "tokenizer.json",
    tokenizerConfig: "tokenizer_config.json",
  },
  source: {
    baseUrl:
      "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/resolve/1427fd652930e4ba29e8149678df786c240d8825/",
    files: [
      {
        path: "onnx/model_qint8_arm64.onnx",
        size: 118_620_017,
        sha256: "1825907d6c1a9001ff78124780bbde20a614a8c3df3b63409cf3c72c6fe5c8b4",
      },
      {
        path: "tokenizer.json",
        size: 17_082_660,
        sha256: "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
      },
      {
        path: "tokenizer_config.json",
        size: 435,
        sha256: "e7fbfbfa6347b4e414c1cee50d142e2c2f9a895dad68b068ae83a8b564c3837e",
      },
    ],
  },
};

/** Alibaba-NLP/gte-multilingual-reranker-base, as converted to int8 ONNX by onnx-community. */
export const GTE_MULTILINGUAL_RERANKER: RerankingModelDefinition = {
  id: "gte-multilingual",
  name: "gte-multilingual-reranker-base",
  folder: "gte-multilingual-reranker-base",
  licence: "Apache-2.0",
  maxTokens: 512,
  files: {
    model: "onnx/model_quantized.onnx",
    tokenizer: "tokenizer.json",
    tokenizerConfig: "tokenizer_config.json",
  },
  source: {
    baseUrl:
      "https://huggingface.co/onnx-community/gte-multilingual-reranker-base/resolve/ee64367e35a2db0da46bb6497e13a18f8bd585cb/",
    files: [
      {
        path: "onnx/model_quantized.onnx",
        size: 340_858_200,
        sha256: "ccf51dba7f8aa9205753761cfaa68c55f741792501463a3bf25d7e5bcdac7c35",
      },
      {
        path: "tokenizer.json",
        size: 17_082_999,
        sha256: "3ffb37461c391f096759f4a9bbbc329da0f36952f88bab061fcf84940c022e98",
      },
      {
        path: "tokenizer_config.json",
        size: 1_340,
        sha256: "6f00514620aff01ba8b7291b2394e98daca5be264cb743805232d9ae27494b2a",
      },
    ],
  },
};

/** BAAI/bge-reranker-v2-m3, as converted to int8 ONNX by onnx-community. */
export const BGE_RERANKER_V2_M3: RerankingModelDefinition = {
  id: "bge-m3",
  name: "bge-reranker-v2-m3",
  folder: "bge-reranker-v2-m3",
  licence: "Apache-2.0",
  maxTokens: 512,
  files: {
    model: "onnx/model_quantized.onnx",
    tokenizer: "tokenizer.json",
    tokenizerConfig: "tokenizer_config.json",
  },
  source: {
    baseUrl:
      "https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/resolve/6f5ff65298512715a1e669753bc754d2bc8f367b/",
    files: [
      {
        path: "onnx/model_quantized.onnx",
        size: 570_727_094,
        sha256: "912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d",
      },
      {
        path: "tokenizer.json",
        size: 17_082_900,
        sha256: "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
      },
      {
        path: "tokenizer_config.json",
        size: 1_203,
        sha256: "b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80",
      },
    ],
  },
};

/** Every candidate, smallest first. */
export const RERANKING_MODEL_CANDIDATES: readonly RerankingModelDefinition[] = [
  MMARCO_MINILM,
  GTE_MULTILINGUAL_RERANKER,
  BGE_RERANKER_V2_M3,
];

/**
 * The model Settings → Reranking offers as "on this computer". Provisional
 * until the evaluation's reranked mode has been run: the smallest candidate,
 * the cheapest at every search.
 */
export const BUILT_IN_RERANKING_MODEL: RerankingModelDefinition = MMARCO_MINILM;

/** The download, in bytes. */
export const downloadSize = (definition: RerankingModelDefinition) =>
  definition.source.files.reduce((sum, file) => sum + file.size, 0);
