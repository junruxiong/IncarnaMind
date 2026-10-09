import { describe, expect, test } from "vitest";
import type { RerankSettings } from "../../src/core/api";
import { rerankingModelStatus } from "../../src/renderer/src/rerankStatus";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

const en = (key: MessageKey, params?: MessageParams) => translate("en", key, params);
const zh = (key: MessageKey, params?: MessageParams) => translate("zh-CN", key, params);

const MODEL = {
  name: "mmarco-mMiniLMv2-L12-H384",
  host: "huggingface.co",
  state: "ready" as const,
  downloadedBytes: 135_703_112,
  totalBytes: 135_703_112,
  error: null,
};

const settings = (
  overrides: Partial<RerankSettings> = {},
  model: Partial<RerankSettings["model"]> = {},
): RerankSettings => ({
  enabled: true,
  byDefault: true,
  kind: "built-in",
  modelId: MODEL.name,
  hasApiKey: false,
  service: null,
  paused: false,
  model: { ...MODEL, ...model },
  ...overrides,
});

describe("The reranking model's download in the sidebar's status row", () => {
  test("shows its progress while it downloads, like the search model's", () => {
    expect(
      rerankingModelStatus(settings({}, { state: "downloading", downloadedBytes: 45_200_000 }), en),
    ).toEqual({
      testId: "reranking-model-download",
      tone: "progress",
      short: "Reranking model: 45 of 136 MB",
      full: "Downloading the reranking model: 45 of 136 MB. Until it's ready, search keeps its own order.",
      retry: false,
    });
    expect(
      rerankingModelStatus(settings({}, { state: "downloading", downloadedBytes: 0 }), zh)?.short,
    ).toBe("重排序模型：0 / 136 MB");
  });

  test("says why it failed, with a retry", () => {
    expect(
      rerankingModelStatus(
        settings(
          {},
          {
            state: "failed",
            error: { kind: "network", message: "fetch failed" },
            downloadedBytes: 0,
          },
        ),
        en,
      ),
    ).toEqual({
      testId: "reranking-model-failed",
      tone: "error",
      short: "Reranking model download failed",
      full: "The reranking model couldn't be downloaded: check your internet connection.\nfetch failed",
      retry: true,
    });
    expect(
      rerankingModelStatus(
        settings({}, { state: "failed", error: { kind: "load", message: "bad model" } }),
        en,
      )?.short,
    ).toBe("Reranking model couldn't start");
  });

  test("says nothing once it is ready, before it is needed, or when it isn't what reranks", () => {
    expect(rerankingModelStatus(settings(), en)).toBeNull();
    expect(rerankingModelStatus(settings({}, { state: "not-downloaded" }), en)).toBeNull();
    expect(rerankingModelStatus(null, en)).toBeNull();
    const failed = { state: "failed" as const, error: { kind: "network" as const, message: "x" } };
    expect(rerankingModelStatus(settings({ kind: "cohere" }, failed), en)).toBeNull();
    expect(rerankingModelStatus(settings({ enabled: false, kind: null }, failed), en)).toBeNull();
  });
});
