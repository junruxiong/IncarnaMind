import { describe, expect, test } from "vitest";
import { downloadSize, RERANKING_MODEL_CANDIDATES } from "../../eval/lib/rerankingModels";
import { BUILT_IN_RERANKING_MODEL } from "../../src/core";

describe("The evaluation's reranking candidates", () => {
  test("every candidate is pinned, has a permissive licence and a multilingual model", () => {
    for (const candidate of RERANKING_MODEL_CANDIDATES) {
      expect(candidate.licence).toMatch(/^(Apache-2\.0|MIT)$/);
      expect(candidate.source.baseUrl).toMatch(/\/resolve\/[0-9a-f]{40}\/$/);
      const paths = candidate.source.files.map((file) => file.path);
      expect(paths).toEqual(
        expect.arrayContaining([
          candidate.files.model,
          candidate.files.tokenizer,
          candidate.files.tokenizerConfig,
        ]),
      );
      for (const file of candidate.source.files) expect(file.sha256).toMatch(/^[0-9a-f]{64}$/);
    }
  });

  test("the built-in model comes first, once, and the candidates' ids differ", () => {
    expect(RERANKING_MODEL_CANDIDATES[0]).toBe(BUILT_IN_RERANKING_MODEL);
    const ids = RERANKING_MODEL_CANDIDATES.map((candidate) => candidate.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toEqual(["mmarco-minilm", "gte-multilingual", "bge-m3"]);
  });

  test("a download's size is its files' sizes added up", () => {
    expect(downloadSize(BUILT_IN_RERANKING_MODEL)).toBe(
      BUILT_IN_RERANKING_MODEL.source.files.reduce((sum, file) => sum + file.size, 0),
    );
    expect(downloadSize(BUILT_IN_RERANKING_MODEL)).toBeGreaterThan(100_000_000);
  });
});
