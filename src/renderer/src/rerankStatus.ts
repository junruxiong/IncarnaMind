/**
 * The built-in reranking model's download, as the sidebar's status row shows
 * it, next to the search model's: its progress, or why it failed. Reranking
 * is on by default with it, so it downloads by itself once there are
 * Documents to search; until it is ready, search keeps its own order.
 */
import type { EmbeddingModelError, RerankSettings } from "../../core/api";
import type { MessageKey, MessageParams } from "../../shared/i18n";

/** Decimal megabytes, as the search model's download is shown. */
const MEGABYTE = 1_000_000;

const failures: Record<Exclude<EmbeddingModelError["kind"], "load">, MessageKey> = {
  network: "embedding.model.failure.network",
  integrity: "embedding.model.failure.integrity",
  storage: "embedding.model.failure.storage",
};

export interface RerankingModelStatus {
  testId: "reranking-model-download" | "reranking-model-failed";
  tone: "progress" | "error";
  /** One line. */
  short: string;
  /** All of it, for the tooltip and screen readers. */
  full: string;
  /** Offer to try the download again. */
  retry: boolean;
}

/** What to show, or null: it is ready, not needed yet, or not what reranks. */
export function rerankingModelStatus(
  rerank: RerankSettings | null,
  t: (key: MessageKey, params?: MessageParams) => string,
): RerankingModelStatus | null {
  if (!rerank?.enabled || rerank.kind !== "built-in") return null;
  const { model } = rerank;
  if (model.state === "downloading") {
    const downloaded = Math.floor(model.downloadedBytes / MEGABYTE);
    const total = Math.ceil(model.totalBytes / MEGABYTE);
    return {
      testId: "reranking-model-download",
      tone: "progress",
      short: t("status.rerankDownloading", { downloaded, total }),
      full: t("rerank.settings.downloading", { downloaded, total }),
      retry: false,
    };
  }
  if (model.state === "failed" && model.error) {
    const { kind, message } = model.error;
    return {
      testId: "reranking-model-failed",
      tone: "error",
      short: t(kind === "load" ? "status.rerankModelFailed" : "status.rerankDownloadFailed"),
      full: `${
        kind === "load"
          ? t("rerank.settings.loadFailed")
          : t("rerank.settings.failed", { reason: t(failures[kind]) })
      }\n${message}`,
      retry: true,
    };
  }
  return null;
}
