import { formatDate, formatFullDateTime } from "../../shared/dates";
import { translate } from "../../shared/i18n";
import { escapeHtml } from "../exports/printHtml";
import type { CheckRecord } from "./index";
import { ORIGIN_LABELS, resultKey } from "./labels";
import type { RecordItem } from "./read";

/**
 * A record as HTML to print as a PDF (see `renderPrintHtml`): the summary, then each checked
 * sentence and each Citation as a block that stays whole on its page. Everything from the Mind
 * and its Documents is escaped: the page shows text, never markup it was given.
 */
export function renderRecordHtml(record: CheckRecord): string {
  const { language } = record;
  const t = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
    translate(language, key, params);
  const number = new Intl.NumberFormat(language === "zh-CN" ? "zh-CN" : "en").format;
  const none = t("checkRecord.summary.none");

  const count = (items: readonly RecordItem[]) => {
    const counts = new Map<string, number>();
    for (const item of items) {
      const label = t(resultKey(item));
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    return [...counts].map(([label, total]) => `${label} ${number(total)}`).join(" · ") || none;
  };
  const summary: [string, string][] = [
    [t("checkRecord.summary.mind"), record.mind],
    [t("checkRecord.summary.exported"), formatDate(record.exportedAt, language)],
    [t("checkRecord.summary.folder"), record.folder ?? t("checkRecord.summary.allDocuments")],
    [
      t("checkRecord.summary.localMode"),
      t(record.localMode ? "checkRecord.summary.on" : "checkRecord.summary.off"),
    ],
    [t("checkRecord.summary.models"), record.models.join("; ") || none],
    [t("checkRecord.summary.sentences"), count(record.sentences)],
    [t("checkRecord.summary.citations"), count(record.citations)],
  ];

  const detail = (label: string, text: string) =>
    `<p class="record-detail"><span class="record-label">${escapeHtml(label)}</span> ${text}</p>`;
  const item = (each: RecordItem): string => {
    const model = each.model
      ? `${each.model.name} (${each.model.service ?? t("checkRecord.thisComputer")})`
      : null;
    const facts = [
      `<b>${escapeHtml(t(resultKey(each)))}</b>`,
      escapeHtml(t(ORIGIN_LABELS[each.origin])),
      ...(model ? [escapeHtml(model)] : []),
      ...(each.checkedAt ? [escapeHtml(formatFullDateTime(each.checkedAt, language))] : []),
    ];
    const { evidence } = each;
    const where = evidence
      ? [evidence.name, evidence.location].filter((part) => part !== null && part !== "")
      : [];
    const separator = t("export.citation.separator");
    const file = evidence
      ? [
          evidence.file,
          evidence.size === null ? null : t("checkRecord.bytes", { count: number(evidence.size) }),
        ].filter((part) => part !== null)
      : [];
    return [
      '<div class="record-item">',
      `<p class="record-text">${escapeHtml(each.text)}</p>`,
      `<p class="record-result">${facts.join(" · ")}</p>`,
      ...(evidence
        ? [detail(t("checkRecord.column.document"), escapeHtml(where.join(separator)))]
        : []),
      ...(evidence?.quote
        ? [detail(t("checkRecord.column.quote"), `“${escapeHtml(evidence.quote)}”`)]
        : []),
      ...(each.says ? [detail(t("checkRecord.column.says"), escapeHtml(each.says))] : []),
      ...(evidence && (file.length > 0 || evidence.sha256)
        ? [
            detail(
              t("checkRecord.column.file"),
              [
                ...file.map((part) => escapeHtml(part)),
                ...(evidence.sha256
                  ? [`SHA-256 <code class="record-hash">${escapeHtml(evidence.sha256)}</code>`]
                  : []),
              ].join(" · "),
            ),
          ]
        : []),
      "</div>",
    ].join("");
  };

  const section = (heading: string, items: readonly RecordItem[], empty: string) =>
    [
      `<h2>${escapeHtml(heading)}</h2>`,
      ...(items.length === 0 ? [`<p>${escapeHtml(empty)}</p>`] : items.map(item)),
    ].join("");

  return [
    `<h1 class="mind-title">${escapeHtml(t("checkRecord.title", { name: record.mind }))}</h1>`,
    `<table class="record-summary">${summary
      .map(
        ([label, value]) =>
          `<tr><th scope="row">${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`,
      )
      .join("")}</table>`,
    `<p class="record-note">${escapeHtml(t("checkRecord.modelNote"))}</p>`,
    section(
      t("checkRecord.sentences.heading", { count: number(record.sentences.length) }),
      record.sentences,
      t("checkRecord.sentences.none"),
    ),
    section(
      t("checkRecord.citations.heading", { count: number(record.citations.length) }),
      record.citations,
      t("checkRecord.citations.none"),
    ),
  ].join("");
}
