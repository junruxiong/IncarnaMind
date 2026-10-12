/**
 * The built-in text recognition model's download (#397), as the sidebar's
 * status row shows it, next to the search and reranking models': its
 * progress, or why it failed, with Retry. It downloads by itself the first
 * time a scanned page needs reading; until then there is nothing to say.
 */
import { create } from "zustand";
import type { EmbeddingModelError, EmbeddingModelStatus } from "../../core/api";
import type { MessageKey, MessageParams } from "../../shared/i18n";
import { core } from "./core";
import { useAppStore } from "./store";

/** Decimal megabytes, as the other models' downloads are shown. */
const MEGABYTE = 1_000_000;

const failures: Record<Exclude<EmbeddingModelError["kind"], "load">, MessageKey> = {
  network: "embedding.model.failure.network",
  integrity: "embedding.model.failure.integrity",
  storage: "embedding.model.failure.storage",
};

interface TextRecognitionState {
  status: EmbeddingModelStatus | null;
  /** Downloads it again after a failure; the Documents waiting for it are read once it's ready. */
  download(): Promise<void>;
}

export const useTextRecognition = create<TextRecognitionState>((set) => ({
  status: null,
  async download() {
    try {
      set({ status: await core.downloadTextRecognitionModel() });
    } catch (error) {
      useAppStore.getState().reportError(error);
    }
  },
}));

core.on("textRecognitionModel.status", (status) => useTextRecognition.setState({ status }));

export interface TextRecognitionStatus {
  testId: "text-recognition-download" | "text-recognition-failed";
  tone: "progress" | "error";
  short: string;
  full: string;
  retry: boolean;
}

/** What the status row says of the model, or null: ready, or not needed yet. */
export function textRecognitionStatus(
  model: EmbeddingModelStatus | null,
  t: (key: MessageKey, params?: MessageParams) => string,
): TextRecognitionStatus | null {
  if (model?.state === "downloading") {
    const downloaded = Math.floor(model.downloadedBytes / MEGABYTE);
    const total = Math.ceil(model.totalBytes / MEGABYTE);
    return {
      testId: "text-recognition-download",
      tone: "progress",
      short: t("status.scanModelDownloading", { downloaded, total }),
      full: t("scan.model.downloading", { downloaded, total }),
      retry: false,
    };
  }
  if (model?.state === "failed" && model.error) {
    const { kind, message } = model.error;
    return {
      testId: "text-recognition-failed",
      tone: "error",
      short: t(kind === "load" ? "status.scanModelFailed" : "status.scanModelDownloadFailed"),
      full: `${
        kind === "load"
          ? t("scan.model.loadFailed")
          : t("scan.model.failed", { reason: t(failures[kind]) })
      }\n${message}`,
      retry: true,
    };
  }
  return null;
}
