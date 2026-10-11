/**
 * What the "Cite as" panel (#286) shows and reads back: each field's label
 * for the record's type, its value as text, what the User typed as a value,
 * and the fields still missing, said as a list. Pure: no store, no bridge,
 * so the tests import it.
 */
import type {
  ReferenceField,
  ReferenceFields,
  ReferenceRecord,
  ReferenceType,
} from "../../core/api";
import type { Language } from "../../core/language";
import { type MessageKey, translate } from "../../shared/i18n";
import {
  formatNames,
  missingDetails,
  normalizeDoi,
  parseIssued,
  parseTypedNames,
} from "../../shared/references";

/** A field's label, which for some fields depends on the record's type: a journal's "Journal". */
export function fieldLabel(language: Language, field: ReferenceField, type: ReferenceType): string {
  if (field === "containerTitle" && type in CONTAINER_LABELS) {
    return translate(language, CONTAINER_LABELS[type as keyof typeof CONTAINER_LABELS]);
  }
  if (field === "publisher" && type in PUBLISHER_LABELS) {
    return translate(language, PUBLISHER_LABELS[type as keyof typeof PUBLISHER_LABELS]);
  }
  return translate(language, `reference.field.${field}`);
}

const CONTAINER_LABELS = {
  "article-journal": "reference.field.containerTitle.article-journal",
  "paper-conference": "reference.field.containerTitle.paper-conference",
  chapter: "reference.field.containerTitle.chapter",
  webpage: "reference.field.containerTitle.webpage",
} as const satisfies Partial<Record<ReferenceType, MessageKey>>;

const PUBLISHER_LABELS = {
  thesis: "reference.field.publisher.thesis",
  report: "reference.field.publisher.report",
} as const satisfies Partial<Record<ReferenceType, MessageKey>>;

/** A field's value as the panel shows it, and as its field starts when edited. */
export function valueText(
  language: Language,
  field: Exclude<ReferenceField, "type">,
  fields: ReferenceFields,
): string {
  return field === "authors" ? formatNames(fields.authors, language) : fields[field];
}

/**
 * A language tag as the panel shows it: its name in the interface's
 * language, then the tag the record keeps, "English (en)", "中文（zh）".
 * A tag with no name known is shown as it is.
 */
export function languageText(language: Language, tag: string): string {
  if (tag === "") return "";
  let name: string | undefined;
  try {
    name = new Intl.DisplayNames(language, { type: "language" }).of(tag);
  } catch {
    // Not a tag Intl knows: shown as written.
  }
  if (!name || name.toLowerCase() === tag.toLowerCase()) return tag;
  return language === "zh-CN" ? `${name}（${tag}）` : `${name} (${tag})`;
}

/** What the User typed in a field, as its value, or the problem that stops it being one. */
export type TypedValue =
  | { ok: true; value: ReferenceFields[Exclude<ReferenceField, "type">] }
  | { ok: false; problem: MessageKey };

export function readTyped(field: Exclude<ReferenceField, "type">, text: string): TypedValue {
  const typed = text.trim();
  if (field === "authors") return { ok: true, value: parseTypedNames(typed) };
  if (typed === "") return { ok: true, value: "" };
  if (field === "issued") {
    const issued = parseIssued(typed);
    return issued
      ? { ok: true, value: issued }
      : { ok: false, problem: "reference.issued.invalid" };
  }
  if (field === "doi") {
    const doi = normalizeDoi(typed);
    return doi ? { ok: true, value: doi } : { ok: false, problem: "reference.doi.invalid" };
  }
  return { ok: true, value: typed };
}

/** The fields a record still needs before it can be cited, as one list in words: "title and date". */
export function missingList(
  language: Language,
  record: Pick<ReferenceRecord, "fields">,
): string | null {
  const missing = missingDetails(record);
  if (missing.length === 0) return null;
  const labels = missing.map((field) => {
    const label = fieldLabel(language, field, record.fields.type);
    return language === "en" ? label.toLocaleLowerCase() : label;
  });
  return new Intl.ListFormat(language, { type: "conjunction" }).format(labels);
}

/** Where a field's value came from, in words, for its tooltip. */
export function sourceText(language: Language, record: ReferenceRecord, field: ReferenceField) {
  const source = record.sources[field];
  return source ? translate(language, `reference.source.${source}`) : null;
}

/** A field's name inside a sentence: "the title", but "the DOI". */
export function fieldInSentence(language: Language, field: ReferenceField, type: ReferenceType) {
  const label = fieldLabel(language, field, type);
  return label === label.toUpperCase() ? label : label.toLocaleLowerCase();
}
