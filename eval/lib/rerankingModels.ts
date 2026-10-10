/**
 * The reranking models the evaluation compares with the one the app ships
 * (`BUILT_IN_RERANKING_MODEL`): multilingual cross-encoders with permissive
 * licences, as int8 ONNX, each pinned to a revision with its files' sizes and
 * SHA-256 hashes. They are for the evaluation only (eval/README.md, #31); the
 * app downloads and runs just the built-in one. The evaluation reranks what
 * the search Tool hands a reranker (keyword search's top 10 and vector search's
 * top 10, each Passage once) with each.
 *
 * Measured on an Apple M2 Max, one pair at a time, 20 Passages of about 500
 * tokens per query, in a Node process of its own (the evaluation, with other
 * work on the machine, measured 0.9 s, 3.4 s and 10.4 s a search):
 * - mmarco-mMiniLMv2-L12-H384, the built-in one (Apache-2.0, 118M parameters):
 *   136 MB, about 0.5 s a query, 0.8 GB resident.
 * - gte-multilingual-reranker-base (Apache-2.0, 306M): 358 MB, about 1.6 s,
 *   1.25 GB.
 * - bge-reranker-v2-m3 (Apache-2.0, 568M): 588 MB, about 4.3 s, 1.9 GB.
 *
 * Not candidates: jina-reranker-v2-base-multilingual (CC BY-NC 4.0, not for
 * commercial use), and English-only or English-and-Chinese-only rerankers.
 */
import { BUILT_IN_RERANKING_MODEL, type RerankingModelDefinition } from "../../src/core";

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

/** Every candidate, smallest first: the built-in model, then the evaluation-only ones. */
export const RERANKING_MODEL_CANDIDATES: readonly RerankingModelDefinition[] = [
  BUILT_IN_RERANKING_MODEL,
  GTE_MULTILINGUAL_RERANKER,
  BGE_RERANKER_V2_M3,
];

/** The download, in bytes. */
export const downloadSize = (definition: RerankingModelDefinition) =>
  definition.source.files.reduce((sum, file) => sum + file.size, 0);
