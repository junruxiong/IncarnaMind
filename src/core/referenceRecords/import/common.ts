/** What the three readers (CSL JSON, BibTeX, RIS) share. */
import { isCjkName, normalizeDoi, parseIssued, parseName } from "../../../shared/references";
import type { ReferenceFields, ReferenceName } from "../../api";

/**
 * A name as an export writes it ("Family, Given", "Given Family", a whole
 * name), as the record keeps it. A name in Chinese, Japanese or Korean is
 * kept whole, family name first, however the export split it ("张, 三", "张 三").
 */
export function nameOf(text: string): ReferenceName | null {
  const name = text.replace(/\s+/g, " ").trim();
  if (name === "" || /^(?:others|et al\.?)$/i.test(name)) return null;
  if (isCjkName(name)) return { literal: name.replace(/[\s,，、]+/g, "") };
  // "Family, Jr., Given": the suffix goes with the given names.
  const parts = name.split(",").map((part) => part.trim());
  if (parts.length === 3 && parts.every((part) => part !== "")) {
    return { family: parts[0] as string, given: `${parts[2]} ${parts[1]}` };
  }
  return parseName(name);
}

/** Languages' names as exports write them, as BCP 47 tags. */
const LANGUAGES: Readonly<Record<string, string>> = {
  english: "en",
  chinese: "zh",
  german: "de",
  french: "fr",
  spanish: "es",
  japanese: "ja",
  korean: "ko",
  italian: "it",
  portuguese: "pt",
  russian: "ru",
  dutch: "nl",
};

export function languageOf(text: string): string {
  const name = text.trim().toLowerCase();
  const known = LANGUAGES[name] ?? LANGUAGES[name.replace(/^.*?\s*[-–]\s*/, "")];
  if (known) return known;
  return /^[a-z]{2,3}(?:[-_][a-z0-9]{2,8})*$/i.test(name) ? name.replace(/_/g, "-") : "";
}

/** Pages as the record keeps them: "456-470" is "456–470". */
export const pagesOf = (text: string) =>
  text
    .replace(/\s*(?:-{1,3}|—|–)\s*/g, "–")
    .replace(/\s+/g, " ")
    .trim();

/** A date as written ("2010-08-04", "2010/08/04/", "2010"), kept at the precision it has. */
export const dateOf = (text: string) =>
  parseIssued(text.replace(/\/+$/, "").replace(/\/{2,}.*$/, "")) ?? "";

export const doiOf = (text: string) => normalizeDoi(text) ?? "";

/** The paths an export's attached-files field lists. */
export function attachedFiles(field: string): string[] {
  const paths: string[] = [];
  // Zotero and JabRef separate files with ";" ("\;" is a ";" in a name).
  for (const entry of field.split(/(?<!\\);/)) {
    const part = entry
      .replace(/\$\\backslash\$/g, "/")
      .replace(/\\([:;\\])/g, "$1")
      .trim();
    if (part === "") continue;
    // "Description:path:mime/type", or Mendeley's ":path:pdf".
    const listed = /^[^:]*:(.+):([A-Za-z0-9.+/-]+)$/.exec(part);
    const path = (listed ? (listed[1] as string) : part).trim();
    if (path !== "") paths.push(path.replace(/^file:\/\//i, ""));
  }
  return paths;
}

/** A field set with only the fields that have a value. */
export function withValues(entries: Partial<ReferenceFields>): Partial<ReferenceFields> {
  const found: Partial<ReferenceFields> = {};
  for (const [field, value] of Object.entries(entries) as [
    keyof ReferenceFields,
    ReferenceFields[keyof ReferenceFields],
  ][]) {
    if (value === "" || (Array.isArray(value) && value.length === 0)) continue;
    (found as Record<string, unknown>)[field] = value;
  }
  return found;
}
