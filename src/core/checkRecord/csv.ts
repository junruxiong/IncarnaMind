import { translate } from "../../shared/i18n";
import type { CheckRecord } from "./index";
import { ORIGIN_LABELS, resultKey } from "./labels";
import type { RecordItem } from "./read";

/**
 * A cell as a spreadsheet reads it safely: quoted when it holds a comma, a quote or a line break,
 * and, where it starts as a formula would (a quote copied from a Document can), after an
 * apostrophe so nothing it says runs.
 */
const cell = (value: string | number | null): string => {
  if (value === null) return "";
  const text = typeof value === "number" ? String(value) : value;
  const safe = /^[=+\-@\t\r]/.test(text) ? `'${text}` : text;
  return /[",\r\n]/.test(safe) ? `"${safe.replaceAll('"', '""')}"` : safe;
};

const row = (cells: readonly (string | number | null)[]) => cells.map(cell).join(",");

/**
 * A record as CSV, for the User's own records: the summary, then one row for each checked
 * sentence and each Citation, in UTF-8 with the byte order mark spreadsheets need to read Chinese.
 * Times are ISO 8601 in UTC, which no setting changes.
 */
export function renderCsv(record: CheckRecord): string {
  const t = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
    translate(record.language, key, params);
  const count = (items: readonly RecordItem[]) => {
    const counts = new Map<string, number>();
    for (const item of items) {
      const label = t(resultKey(item));
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    return [...counts].map(([label, number]) => `${label} ${number}`).join("; ");
  };
  const summary = [
    [t("checkRecord.title", { name: record.mind })],
    [t("checkRecord.summary.exported"), record.exportedAt],
    [t("checkRecord.summary.folder"), record.folder ?? t("checkRecord.summary.allDocuments")],
    [
      t("checkRecord.summary.localMode"),
      t(record.localMode ? "checkRecord.summary.on" : "checkRecord.summary.off"),
    ],
    [
      t("checkRecord.summary.models"),
      record.models.length > 0 ? record.models.join("; ") : t("checkRecord.summary.none"),
    ],
    [t("checkRecord.summary.sentences"), count(record.sentences) || t("checkRecord.summary.none")],
    [t("checkRecord.summary.citations"), count(record.citations) || t("checkRecord.summary.none")],
    [t("checkRecord.modelNote")],
    [t("checkRecord.note")],
  ];
  const header = [
    "kind",
    "sentence",
    "result",
    "text",
    "document",
    "file",
    "size",
    "sha256",
    "location",
    "quote",
    "says",
    "model",
    "service",
    "checked",
  ].map((name) => t(`checkRecord.column.${name}` as Parameters<typeof translate>[1]));
  const items = [...record.sentences, ...record.citations].map((item) => [
    t(`checkRecord.kind.${item.kind}`),
    item.text,
    t(resultKey(item)),
    t(ORIGIN_LABELS[item.origin]),
    item.evidence?.name ?? null,
    item.evidence?.file ?? null,
    item.evidence?.size ?? null,
    item.evidence?.sha256 ?? null,
    item.evidence?.location ?? null,
    item.evidence?.quote ?? null,
    item.says,
    item.model?.name ?? null,
    item.model ? (item.model.service ?? t("checkRecord.thisComputer")) : null,
    item.checkedAt,
  ]);
  const lines = [...summary.map(row), "", row(header), ...items.map(row)];
  return `﻿${lines.join("\r\n")}\r\n`;
}
