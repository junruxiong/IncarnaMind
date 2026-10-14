import type { EmbeddingModelStatus } from "../../core/api";
import type { MessageKey, MessageParams } from "../../shared/i18n";

/** Decimal megabytes, as every model's size is quoted. */
const MEGABYTE = 1_000_000;

/**
 * How far a model's download has come, in whole megabytes: the numbers of
 * the full sentences, and the figure the sidebar's status row keeps whole
 * ("3/22 MB").
 */
export function downloadProgress(
  model: EmbeddingModelStatus,
  t: (key: MessageKey, params?: MessageParams) => string,
) {
  const downloaded = Math.floor(model.downloadedBytes / MEGABYTE);
  const total = Math.ceil(model.totalBytes / MEGABYTE);
  return { downloaded, total, figure: t("units.mb", { size: `${downloaded}/${total}` }) };
}
