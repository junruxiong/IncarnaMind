/**
 * The built-in reranking model: a multilingual cross-encoder with a permissive
 * licence, as int8 ONNX, pinned to a revision with its files' sizes and
 * SHA-256 hashes. The app ships one, `BUILT_IN_RERANKING_MODEL`, on by
 * default. The retrieval evaluation (eval/README.md, #31) compares it with
 * two larger candidates, whose definitions live with the evaluation
 * (eval/lib/rerankingModels.ts).
 *
 * Measured on an Apple M2 Max, one pair at a time, 20 Passages of about 500
 * tokens per query, in a Node process of its own: mmarco-mMiniLMv2-L12-H384
 * (Apache-2.0, 118M parameters) is 136 MB, about 0.5 s a query, 0.8 GB resident.
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
const MMARCO_MINILM: RerankingModelDefinition = {
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

/**
 * The built-in reranking model: on by default, and the one Settings →
 * Reranking offers as "on this computer". In the evaluation's run of 2026-10-09
 * (#31), reranking with it took hybrid search from 13 to 16 of 20 English Questions and from 19 to 20 of
 * 20 Chinese ones, as bge-reranker-v2-m3 did at four times the download and
 * twelve times the time; gte-multilingual-reranker-base changed nothing. It
 * adds about 0.5 to 0.9 s to a search.
 */
export const BUILT_IN_RERANKING_MODEL: RerankingModelDefinition = MMARCO_MINILM;
