import type { Document } from "../../core/api";
import type { Language } from "../../core/language";
import type { MessageKey } from "../../shared/i18n";

/** What a Document's kind is called, in the words of the Library's filters. */
export const KIND_LABELS: Record<Document["kind"], MessageKey> = {
  pdf: "library.format.pdf",
  docx: "library.format.docx",
  pptx: "library.format.pptx",
  xlsx: "library.format.xlsx",
  csv: "library.format.csv",
  markdown: "library.format.markdown",
  text: "library.format.text",
  web: "library.format.web",
  transcript: "library.format.transcript",
};

/** "Get info" is "Properties" on Windows and Linux, in Chinese. */
export const infoLabelKey = (platform: string): MessageKey =>
  /Mac|iPhone|iPad/.test(platform) ? "info.get" : "info.get.other";

/**
 * A file's own date as its metadata gave it, at the precision it gave ("2019",
 * "2019-03", "2019-03-04", or with a time and an offset), shown as the date it
 * names there, not the day it falls on in another time zone.
 */
export function fileDate(value: string, language: Language): string {
  const precision = /^\d{4}$/.test(value) ? "year" : /^\d{4}-\d{2}$/.test(value) ? "month" : "day";
  const day =
    precision === "year"
      ? `${value}-01-01`
      : precision === "month"
        ? `${value}-01`
        : value.slice(0, 10);
  const date = new Date(`${day}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return value;
  const options: Intl.DateTimeFormatOptions =
    precision === "year"
      ? { year: "numeric" }
      : precision === "month"
        ? { year: "numeric", month: "long" }
        : { dateStyle: "long" };
  return new Intl.DateTimeFormat(language, { ...options, timeZone: "UTC" }).format(date);
}

/** When something was added, as the day it was on this computer. */
export const addedDate = (iso: string, language: Language): string =>
  new Intl.DateTimeFormat(language, { dateStyle: "long" }).format(new Date(iso));
